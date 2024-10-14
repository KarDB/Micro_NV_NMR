use crate::linear_algebra::{intersects_chip_walls, make_rotation};
use crate::types::*;
use bvh::ray::Ray;
use hdf5::File;
use indicatif::ProgressBar;
use nalgebra::{Point3, Vector3};
use ndarray::{arr1, s, Array3};
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use rand_distr::{Distribution, Normal, NormalError};
use rayon::prelude::*;
use std::fs::OpenOptions; 


//export DYLD_LIBRARY_PATH=/usr/local/Cellar/libiconv/1.17/lib:$DYLD_LIBRARY_PATH
//cargo build --release
//nvmr config.yaml or ./target/release/nvmr config.yaml

//to commit in git:
//git status
//git add -u
//git commit -m "note to write"
//git commit --amend --author="giacomomerlo <183497957+giacomomerlo@users.noreply.github.com>"

//the magnetic field gradient mostly changes the rotation of the spin instead of the position
//in the end, we plot the amplitude of the oscillation and if there is a tumor cell (field gradient), we see a black spot



pub fn start_sim(
    m1: Vector3<f32>, 
    m2: Vector3<f32>,
    nv_depth: f32,
    n_prot: usize,
    filepath: String,
    stlfile: String,
    resolution_x: u32,
    resolution_y: u32,
    diffusion_coefficient: f32,
    frequency: f32, //use this as "|B0|" nd multiply it for m1 to obtain the B0 vector
    number_time_steps: usize,
    timestep: f32,
    scale_factor: f32,
    parallelization_level: usize,
) -> f32 {
    let n_prot = get_per_batch_proton_number(&parallelization_level, &n_prot); 
    let triangles = make_triangles(stlfile.clone()); //it creates vector of (structure) triangles from the stl file 
    let triangles_all = make_all_triangles(stlfile.clone()); //triangles in all three dimension
    let dimensions = get_dims(stlfile.clone());
    let mut pos = make_timeresolved_locations( //k-th NV location at the j-th time step of the i-th parallel run => parallelized[i][j][k]
        dimensions, 
        nv_depth,
        resolution_x,
        resolution_y,
        number_time_steps,
        scale_factor,
        &parallelization_level,
    );

    let mut rng = SmallRng::from_entropy();

    let gradient = Vector3::new(  // random magnetic field gradient [mT/mm]
        rng.gen::<f32>(),
        rng.gen::<f32>(), 
        rng.gen::<f32>(),
    );

    println!("gradient = {gradient}");

    let magnetic_field_0 = frequency * &m1; //use the input "frequency" as |B0| and compute B0 || to the nv axix

    let volume = get_chip_volume(&dimensions);
    let total_count: usize = 0;

    //let rotation_angle = get_rotation_angle(frequency, timestep);
    //let rotation_matrix = make_rotation(&m1, rotation_angle); //rotation matrix of an angle rotation_angle

    let diffusion_stepsize = get_rms_diffusion_displacement(diffusion_coefficient, timestep);
    let bar = ProgressBar::new(n_prot as u64 * parallelization_level as u64);

    pos.par_iter_mut().for_each(|nv_array| {
        let mut proton_count: usize = 0;
        let mut rng = SmallRng::from_entropy(); //random number generator

        while proton_count < n_prot { //each iteration we "create" a new proton and diffuse it
            bar.inc(1);
            let mut m2_current = m2.clone();  //clone m2
            let mut proton = make_proton_position(&mut rng, &dimensions, &mut proton_count, &triangles); //random proton position using the triangle intersection

            for time_step in 0..number_time_steps as usize {
                
                dd_for_all_pos(proton.origin, m1, m2_current, &mut nv_array[time_step]); //compute the dipole interaction between the proton and the nv centers (in nv.interaction)

                //the magnetic field changes the precession !!the field is in mT
                //let field = magnetic_field_0 + magnetic_field(&proton.origin, &gradient, &dimensions); //compute the magnetic field in the proton position [mT]
                let field = magnetic_field_0 + magnetic_field_r3(&proton.origin, &dimensions) + magnetic_field(&proton.origin, &gradient, &dimensions);
                let rotation_angle = get_rotation_angle_field(&field, timestep); //compute the rotation angle
                let rotation_matrix = make_rotation(&field, rotation_angle); //rotation matrix around the field
                m2_current = rotation_matrix * m2_current; //rotation
        
                diffuse_proton(&mut proton, &mut rng, &diffusion_stepsize, &triangles_all);   //upgrade the position of the proton 
                //diffuse_proton_field(&mut proton, &mut rng, &diffusion_stepsize, &triangles_all, &gradient); //upgrade position of the proton with the magnetic field
            }
        }
    });

    bar.finish();

    let hdf5_data = accumulate_parallelized_interactions(pos); 
    //3d array with sum of parallelizations ([timestep, N_nv, (x, y, z, interaction)])

    let metadata = Metadata {
        stl_file: stlfile.clone().parse().unwrap(),
        timestep,
        m1x: m1.x,
        m1y: m1.y,
        m1z: m1.z,
        m2x: m2.x,
        m2y: m2.y,
        m2z: m2.z,
        frequency,
        diffusion_coefficient,
        number_time_steps,
        nv_depth,
        proton_count: 10,
        resolution_x,
        resolution_y,
    };
    
    let _ = save_to_hdf5(&hdf5_data, &metadata, filepath);
    let proton_count = n_prot;
    return volume * (proton_count as f32) / (total_count as f32);  //total count is 0?
}
// To use function, create proton list like so:
// let mut proton_positions = make_timeresolved_proton_list(&n_prot, &number_time_steps);
#[allow(dead_code)]
fn track_proton_positions(
    proton_positions: &mut Array3<f32>,
    proton: &Ray<f32, 3>,
    m2_current: &Vector3<f32>,
    proton_count: usize,
    time_step: usize,
) {
    proton_positions
        .slice_mut(s![proton_count - 1, time_step, ..])
        .assign(&arr1(&[
            proton.origin.x,
            proton.origin.y,
            proton.origin.z,
            m2_current.x,
            m2_current.y,
            m2_current.z,
        ]));
}

fn accumulate_parallelized_interactions(pos: Vec<Vec<Vec<NVLocation>>>) -> Array3<f32> {
    let mut array = convert_to_array3(&pos[0]); //convert pos[0] (first parallelization) into a 3d array
    for arr in pos.iter().skip(1) { //convert the other layers into 3d vectors
        let arr_from_pos = convert_to_array3(arr);
        array = array + arr_from_pos; //sum the different parallelization ("accumulate the results")
    }
    array
}

// Check to see if number of protons can be equally distbributed across cores.
fn validate_parllelization(parallelization_level: &usize, n_prot: &usize) {
    if n_prot % parallelization_level != 0 {
        panic!(
            "The number of protons ({}) cannot be distributed equally across the requested number of cores ({}).",
            n_prot, parallelization_level
        );
    }
}

fn get_per_batch_proton_number(parallelization_level: &usize, n_prot: &usize) -> usize {
    validate_parllelization(parallelization_level, n_prot);
    n_prot / parallelization_level
}

fn get_rms_diffusion_displacement(diffusion_coefficient: f32, timestep: f32) -> f32 {
    // We are working in 3D so the dimensionality factor is 6 (6 dof?)
    // However, we are drawing the 3 components of a vector.
    // So for each component, the dimesionality factor is 2.
    let dim_factor = 2.0;
    (dim_factor * timestep * diffusion_coefficient).sqrt()
}

//the rotation angle depends also on the magnetic field
// fn get_rotation_angle(frequency: f32, timestep: f32) -> f32 {
//     frequency * 2.0 * std::f32::consts::PI * timestep
// }

fn get_rotation_angle_field(field: &Vector3<f32>, timestep: f32) -> f32 {
    let gamma = 2.675e5; //giromagnetic ratio of a proton rad / mT s
    let magnitude = field.norm(); //magnitude of the magnetic field
    let frequency = gamma * magnitude; //Larmor frequency
    frequency * timestep
}

fn generate_gaussian(rng: &mut SmallRng, std: &f32) -> f32 {
    let mean = 0.0;
    let normal_dist = Normal::new(mean, std.to_owned());
    match normal_dist {
        Ok(dist) => dist.sample(rng),
        Err(NormalError::BadVariance) => 0.0,
        Err(NormalError::MeanTooSmall) => panic!("MeanToSmall error should not exists here!"),
    }
}

// fn diffuse_proton(
//     ray: &mut Ray<f32, 3>,
//     rng: &mut SmallRng,
//     stepsize: &f32,
//     triangles: &Vec<Triangle>,
// ) {
//     let mut position_update = Vector3::new(
//         generate_gaussian(rng, stepsize),
//         generate_gaussian(rng, stepsize),
//         generate_gaussian(rng, stepsize),
//     );
//     // loop {
//     let wall_intersection = intersects_chip_walls(triangles, &ray.origin, &position_update);
//     match wall_intersection {
//         Some(intersection) => {
//             let (new_origin, new_pos_update) = reflect_on_wall(
//                 &ray.origin,
//                 &position_update,
//                 &intersection.1,
//                 &intersection.0,
//             );
//             ray.origin = new_origin + new_pos_update;
//             dbg!("Intersections Detected");
//             // position_update = new_pos_update;
//         }
//         None => {
//             ray.origin += position_update;
//             // break;
//         }
//     }
//     // }
//     // ray.origin += position_update;
// }

fn diffuse_proton(
    ray: &mut Ray<f32, 3>, //random proton vector (position+direction) using the triangle intersection
    rng: &mut SmallRng, //random number
    stepsize: &f32,  //diffusion stepsize
    triangles: &Vec<Triangle>, //all triangles
) {
    let mut position_update = Vector3::new( //direction of the diffusion
        generate_gaussian(rng, stepsize), //random number from a gaussian distribution
        generate_gaussian(rng, stepsize),
        generate_gaussian(rng, stepsize),
    );

    loop {
        let wall_intersection = intersects_chip_walls(triangles, &ray.origin, &position_update); //checks if we have an intersection and returns the triangle
        match wall_intersection {
            Some(_) => { //if we have an intersection, we create a new potential movement direction and check again the intersection
                position_update = Vector3::new(
                    generate_gaussian(rng, stepsize),
                    generate_gaussian(rng, stepsize),
                    generate_gaussian(rng, stepsize),
                );
            }
            None => { //if we don't have an intersection, we evolve the position of the proton
                ray.origin += position_update;
                break;
            }
        }
    }
}


//compute the magnetic field in a given position (suppose linear magnetic field gradient) in [mT]
fn magnetic_field(position: &Point3<f32>, gradient: &Vector3<f32>, dimensions: &ChipDimensions) -> Vector3<f32> {
    
    let x0 = (dimensions.max_x - dimensions.min_x) * 0.75 + dimensions.min_x; 
    let y0 = (dimensions.max_y - dimensions.min_y) * 0.5 + dimensions.min_y;
    let z0 = (dimensions.max_z - dimensions.min_z) * 0.33 + dimensions.min_z;

    Vector3::new(
        (position.x - x0) * gradient.x, 
        (position.y - y0) * gradient.y,
        (position.z - z0) * gradient.z,
    )
}

fn magnetic_field_r3(position: &Point3<f32>, dimensions: &ChipDimensions) -> Vector3<f32> { //magnetic field of a cancer cell?
    
    let x0 = (dimensions.max_x - dimensions.min_x) * 0.35 + dimensions.min_x; 
    let y0 = (dimensions.max_y - dimensions.min_y) * 0.45 + dimensions.min_y;
    let z0 = (dimensions.max_z - dimensions.min_z) * 0.5 + dimensions.min_z;

    let radius = (position.x - x0).powi(2) + (position.y - y0).powi(2) + (position.z - z0).powi(2); //radius^2

    let module = 1.0 / radius.powi(2);

    Vector3::new(
        (position.x - x0) * module, 
        (position.y - y0) * module,
        (position.z - z0) * module,
    )
}

//new function to diffuse the proton where we also take into account a magnetic field gradient
// fn diffuse_proton_field(
//     ray: &mut Ray<f32, 3>, //random proton vector (position+direction) using the triangle intersection
//     rng: &mut SmallRng, //random number
//     stepsize: &f32,  //diffusion stepsize
//     triangles: &Vec<Triangle>, //all triangles
//     gradient: &Vector3<f32> 
// ) {
//     let mut position_test = Vector3::new( 
//         generate_gaussian(rng, stepsize),
//         generate_gaussian(rng, stepsize),
//         generate_gaussian(rng, stepsize),
//     );

//     let mut field = magnetic_field(&position_test, &gradient);
//     let mut position_update = position_test.cross(&field);

//     loop {
//         let wall_intersection = intersects_chip_walls(triangles, &ray.origin, &position_update); //checks if we have an intersection and returns the triangle
//         match wall_intersection {
//             Some(_) => { //if we have an intersection, we create a new potential movement direction and check again the intersection
//                 position_test = Vector3::new(
//                     generate_gaussian(rng, stepsize),
//                     generate_gaussian(rng, stepsize),
//                     generate_gaussian(rng, stepsize),
//                 );
//                 field = magnetic_field(&position_test, &gradient);
//                 position_update = position_test.cross(&field);

//             }
//             None => { //if we don't have an intersection, we evolve the position of the proton
//                 ray.origin += position_update;
//                 break;
//             }
//         }
//     }
// }



fn get_chip_volume(dimensions: &ChipDimensions) -> f32 {
    let volume = (dimensions.max_x - dimensions.min_x)
        * (dimensions.max_y - dimensions.min_y)
        * (dimensions.max_z - dimensions.min_z);
    volume
}

fn make_proton_position(
    rng: &mut SmallRng,
    dimensions: &ChipDimensions,
    // total_count: &mut usize,
    proton_count: &mut usize,
    triangles: &Vec<Triangle>,
) -> Ray<f32, 3> {
    loop {
        let ray = Ray::new( 
            Point3::new(
                rng.gen::<f32>() * (dimensions.max_x - dimensions.min_x) + dimensions.min_x, //random position inside the chip
                rng.gen::<f32>() * (dimensions.max_y - dimensions.min_y) + dimensions.min_y,
                rng.gen::<f32>() * (dimensions.max_z - dimensions.min_z) + dimensions.min_z,
            ), // Origin point
            Vector3::new(0.0, 0.0, 1.0), // Direction vector (z versor)
        );
        let mut intersect_count = 0;
        for triangle in triangles {
            let intersect = ray.intersects_triangle_no_cull(&triangle.a, &triangle.b, &triangle.c);
            if intersect { //can be true or false
                intersect_count += 1;
            }
        }
        // *total_count += 1;
        if intersect_count % 2 == 0 {  //check if the proton hits even or odd times the triangles
            *proton_count += 1;
            return ray;  //even=>outside the triangle=>return the array (breaks the loop)
        }
    }
}

pub fn make_triangles(stlfile: String) -> Vec<Triangle> {
    let mut file = OpenOptions::new().read(true).open(stlfile).unwrap(); //opening the stl file
    let stl = stl_io::read_stl(&mut file).unwrap(); //reading the stl file
    let mut triangles = std::vec::Vec::new(); //inizialize the new vector
    for face in stl.faces {
        if face.normal[2] != 0.0 {
            let tr = Triangle {
                a: Point3::new( //coordinates of the vertex a
                    stl.vertices[face.vertices[0] as usize][0],
                    stl.vertices[face.vertices[0] as usize][1],
                    stl.vertices[face.vertices[0] as usize][2],
                ),
                b: Point3::new(
                    stl.vertices[face.vertices[1] as usize][0],
                    stl.vertices[face.vertices[1] as usize][1],
                    stl.vertices[face.vertices[1] as usize][2],
                ),
                c: Point3::new(
                    stl.vertices[face.vertices[2] as usize][0],
                    stl.vertices[face.vertices[2] as usize][1],
                    stl.vertices[face.vertices[2] as usize][2],
                ),
            };
            triangles.push(tr); //vector of triangles
        }
    }
    return triangles;
}

pub fn make_all_triangles(stlfile: String) -> Vec<Triangle> {
    let mut file = OpenOptions::new().read(true).open(stlfile).unwrap();
    let stl = stl_io::read_stl(&mut file).unwrap();
    let mut triangles = std::vec::Vec::new();
    for face in stl.faces {
        let tr = Triangle {
            a: Point3::new(
                stl.vertices[face.vertices[0] as usize][0],
                stl.vertices[face.vertices[0] as usize][1],
                stl.vertices[face.vertices[0] as usize][2],
            ),
            b: Point3::new(
                stl.vertices[face.vertices[1] as usize][0],
                stl.vertices[face.vertices[1] as usize][1],
                stl.vertices[face.vertices[1] as usize][2],
            ),
            c: Point3::new(
                stl.vertices[face.vertices[2] as usize][0],
                stl.vertices[face.vertices[2] as usize][1],
                stl.vertices[face.vertices[2] as usize][2],
            ),
        };
        triangles.push(tr); //similar to append in python
    }
    return triangles;
}

fn get_dims(stlfile: String) -> ChipDimensions {
    let mut file = OpenOptions::new().read(true).open(stlfile).unwrap();
    let stl = stl_io::read_stl(&mut file).unwrap();

    let mut xvec = Vec::new();
    for x in stl.vertices.iter() {
        xvec.push(x[0].clone());
    }
    let mut yvec = Vec::new();
    for x in stl.vertices.iter() {
        yvec.push(x[1].clone());
    }
    let mut zvec = Vec::new();
    for x in stl.vertices.iter() {
        zvec.push(x[2].clone());
    }
    let chip_dimensions = ChipDimensions {
        min_x: *xvec
            .iter()
            .min_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap(),
        max_x: *xvec
            .iter()
            .max_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap(),
        min_y: *yvec
            .iter()
            .min_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap(),
        max_y: *yvec
            .iter()
            .max_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap(),
        min_z: *zvec
            .iter()
            .min_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap(),
        max_z: *zvec
            .iter()
            .max_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap(),
    };
    return chip_dimensions;
}

// needed if we want to track proton positions.
#[allow(dead_code)]
fn make_timeresolved_proton_list(number_protons: &usize, time_steps: &usize) -> Array3<f32> {
    let proton_time_resolved_list =
        Array3::<f32>::zeros((number_protons.to_owned(), time_steps.to_owned(), 6));
    proton_time_resolved_list
}

fn make_timeresolved_locations( //k-th NV location at the j-th time step of the i-th parallel run => parallelized[i][j][k]
    dimensions: ChipDimensions,
    nv_depth: f32,
    resolution_x: u32,
    resolution_y: u32,
    steps: usize, //time steps
    scale_factor: f32,
    parallelization_level: &usize,
) -> Vec<Vec<Vec<NVLocation>>> {  //3d matrix where each element is a NVLocation
    let mut parallelized = std::vec::Vec::new();
    for _ in 0..parallelization_level.to_owned() {
        let mut timeresolved = std::vec::Vec::new();
        for _ in 0..steps {
            timeresolved.push(make_nv_locations( //this (make_nv_loc) is a vector with the positions of the nv centers
                dimensions,
                nv_depth,
                resolution_x,
                resolution_y,
                scale_factor,
            ));
        }
        parallelized.push(timeresolved); //this (timeresolved) is a matrix A with the position of the nv centers for each time steps
        //the positions of a given NV is the same for each time step
    }
    parallelized //this is a 3d array with different parallelizations of A
}

fn make_nv_locations(  
    dimensions: ChipDimensions,
    nv_depth: f32,
    resolution_x: u32,
    resolution_y: u32,
    scale_factor: f32,
) -> Vec<NVLocation> {  //vector of structure NVLocation
    let mut nv_locations = std::vec::Vec::new();
    let scaled_x_distance = (dimensions.max_x - dimensions.min_x) / scale_factor;
    let scaled_y_distance = (dimensions.max_y - dimensions.min_y) / scale_factor;
    let x_step = scaled_x_distance / (resolution_x - 1) as f32;
    let y_step = scaled_y_distance / (resolution_y - 1) as f32;
    for x in 0..resolution_x {
        let x_offset = dimensions.min_x + (scale_factor / 2.0 - 0.5) * scaled_x_distance;
        let x = x as f32 * x_step + x_offset;
        for y in 0..resolution_y {
            let y_offset = dimensions.min_y + (scale_factor / 2.0 - 0.5) * scaled_y_distance;
            let y = y as f32 * y_step + y_offset;
            let new_location = NVLocation {
                loc: Point3::new(x, y, -nv_depth / 1000.0),
                interaction: 0.0,
            };
            nv_locations.push(new_location); 
        }
    }
    return nv_locations;
}

fn dd_for_all_pos(
    ray_origin: Point3<f32>,
    m1: Vector3<f32>,
    m2: Vector3<f32>,
    pos: &mut Vec<NVLocation>,
) {
    pos.iter_mut().for_each(|nv| {
        nv.interaction += dipole_dipole(ray_origin - nv.loc, m1, m2);
    });
}

// The Units are completely wrong in the following!!!
// mu0 is in SI base whereas r, m1 and m2 are in mm.
fn dipole_dipole(r: Vector3<f32>, m1: Vector3<f32>, m2: Vector3<f32>) -> f32 {
    let mu0 = 0.00000125663706212; // N⋅A−2 / kg m s-2 A−2
    let k = mu0 / (4.0 * std::f32::consts::PI);
    let r_norm = r.norm(); 
    let r_unit = r / r_norm;

    let interaction = -k / r_norm.powi(3) * (3.0 * m1.dot(&r_unit) * m2.dot(&r_unit) - m1.dot(&m2));
    return interaction;
}

fn convert_to_array3(data: &Vec<Vec<NVLocation>>) -> Array3<f32> {  //(the input is the position-timestep matrix)
    let depth = data.len(); //time step
    let height = data.iter().map(|v| v.len()).max().unwrap_or(0); //nv location
    let mut array = Array3::<f32>::zeros((depth, height, 4));

    for (i, layer) in data.iter().enumerate() {
        for (j, nvloc) in layer.iter().enumerate() {
            let flat_index = [i, j, 0]; // Starting index for each NVLocation
            array[[flat_index[0], flat_index[1], 0]] = nvloc.loc.x;
            array[[flat_index[0], flat_index[1], 1]] = nvloc.loc.y;
            array[[flat_index[0], flat_index[1], 2]] = nvloc.loc.z;
            array[[flat_index[0], flat_index[1], 3]] = nvloc.interaction;
        }
    }

    array
}

fn save_to_hdf5(
    data: &Array3<f32>,
    // proton_positions: &Array3<f32>,
    metadata: &Metadata,
    filename: String,
) -> hdf5::Result<()> { //result type can be "ok" or "error"
    let file = File::create(&filename).map_err(|e| {
        eprintln!("Failed to create file '{}': {}", filename, &e);
        e
    })?;

    let dataset = file
        .new_dataset_builder()
        .with_data(data)
        .create("dataset")
        .map_err(|e| {
            eprintln!("Failed to create dataset in file '{}': {}", filename, &e);
            e
        })?;
    // let dataset_2 = file
    //     .new_dataset_builder()
    //     .with_data(proton_positions)
    //     .create("protons")
    //     .map_err(|e| {
    //         eprintln!("Failed to create dataset in file '{}': {}", filename, &e);
    //         e
    //     })?;

    let attr = dataset
        .new_attr::<hdf5::types::VarLenUnicode>()
        .create("stl_file")
        .map_err(|e| {
            eprintln!("Failed to create attribute stl_file. {}", &e);
            e
        })?;
    attr.write_scalar(&metadata.stl_file).map_err(|e| {
        eprintln!("Failed to create attribute stl_file. {}", &e);
        e
    })?;

    let attr = dataset.new_attr::<f32>().create("timestep").map_err(|e| {
        eprintln!("Failed to create attribute timestep. {}", &e);
        e
    })?;
    attr.write_scalar(&metadata.timestep).map_err(|e| {
        eprintln!("Failed to create attribute timestep. {}", &e);
        e
    })?;
    // vector m1
    let attr = dataset.new_attr::<f32>().create("m1x").map_err(|e| {
        eprintln!("Failed to create attribute m1x. {}", &e);
        e
    })?;
    attr.write_scalar(&metadata.m1x).map_err(|e| {
        eprintln!("Failed to create attribute m1x. {}", &e);
        e
    })?;
    let attr = dataset.new_attr::<f32>().create("m1y").map_err(|e| {
        eprintln!("Failed to create attribute m1y. {}", &e);
        e
    })?;
    attr.write_scalar(&metadata.m1y).map_err(|e| {
        eprintln!("Failed to create attribute m1y. {}", &e);
        e
    })?;
    let attr = dataset.new_attr::<f32>().create("m1z").map_err(|e| {
        eprintln!("Failed to create attribute m1z. {}", &e);
        e
    })?;
    attr.write_scalar(&metadata.m1z).map_err(|e| {
        eprintln!("Failed to create attribute m1z. {}", &e);
        e
    })?;
    // vector m1
    let attr = dataset.new_attr::<f32>().create("m2x").map_err(|e| {
        eprintln!("Failed to create attribute m2x. {}", &e);
        e
    })?;
    attr.write_scalar(&metadata.m2x).map_err(|e| {
        eprintln!("Failed to create attribute m2x. {}", &e);
        e
    })?;
    let attr = dataset.new_attr::<f32>().create("m2y").map_err(|e| {
        eprintln!("Failed to create attribute m2y. {}", &e);
        e
    })?;
    attr.write_scalar(&metadata.m2y).map_err(|e| {
        eprintln!("Failed to create attribute m2y. {}", &e);
        e
    })?;
    let attr = dataset.new_attr::<f32>().create("m2z").map_err(|e| {
        eprintln!("Failed to create attribute m2z. {}", &e);
        e
    })?;
    attr.write_scalar(&metadata.m2z).map_err(|e| {
        eprintln!("Failed to create attribute m2z. {}", &e);
        e
    })?;

    let attr = dataset.new_attr::<f32>().create("nv_depth").map_err(|e| {
        eprintln!("Failed to create attribute nv_depth. {}", &e);
        e
    })?;
    attr.write_scalar(&metadata.nv_depth).map_err(|e| {
        eprintln!("Failed to create attribute nv_depth. {}", &e);
        e
    })?;

    let attr = dataset
        .new_attr::<f32>()
        .create("diffusion_coefficient")
        .map_err(|e| {
            eprintln!("Failed to create attribute diffusion_coefficient. {}", &e);
            e
        })?;
    attr.write_scalar(&metadata.diffusion_coefficient)
        .map_err(|e| {
            eprintln!("Failed to create attribute diffusion_coefficient. {}", &e);
            e
        })?;

    let attr = dataset.new_attr::<f32>().create("frequency").map_err(|e| {
        eprintln!("Failed to create attribute frequency. {}", &e);
        e
    })?;
    attr.write_scalar(&metadata.frequency).map_err(|e| {
        eprintln!("Failed to create attribute frequency. {}", &e);
        e
    })?;

    let attr = dataset
        .new_attr::<usize>()
        .create("proton_count")
        .map_err(|e| {
            eprintln!("Failed to create attribute proton_count. {}", &e);
            e
        })?;
    attr.write_scalar(&metadata.proton_count).map_err(|e| {
        eprintln!("Failed to create attribute proton_count. {}", &e);
        e
    })?;

    let attr = dataset
        .new_attr::<u32>()
        .create("resolution_x")
        .map_err(|e| {
            eprintln!("Failed to create attribute resolution_x. {}", &e);
            e
        })?;
    attr.write_scalar(&metadata.resolution_x).map_err(|e| {
        eprintln!("Failed to create attribute resolution_x. {}", &e);
        e
    })?;

    let attr = dataset
        .new_attr::<u32>()
        .create("resolution_y")
        .map_err(|e| {
            eprintln!("Failed to create attribute resolution_y. {}", &e);
            e
        })?;
    attr.write_scalar(&metadata.resolution_y).map_err(|e| {
        eprintln!("Failed to create attribute resolution_y. {}", &e);
        e
    })?;

    let attr = dataset
        .new_attr::<usize>()
        .create("number_time_steps")
        .map_err(|e| {
            eprintln!("Failed to create attribute number_time_steps. {}", &e);
            e
        })?;
    attr.write_scalar(&metadata.number_time_steps)
        .map_err(|e| {
            eprintln!("Failed to create attribute number_time_steps. {}", &e);
            e
        })?;
    Ok(())
}

trait IntersectTriangle {
    fn intersects_triangle_no_cull(
        &self,
        a: &Point3<f32>,
        b: &Point3<f32>,
        c: &Point3<f32>,
    ) -> bool;
}

// Adjusted from bvh Ray.
// Originally licensed under MIT
//Moller-Trumbore algorithm
// https://github.com/svenstaro/bvh
impl IntersectTriangle for Ray<f32, 3> { //checking if a ray intersects a triangle abc
    fn intersects_triangle_no_cull(
        &self, //(vector at a random point in the z direction)
        a: &Point3<f32>,
        b: &Point3<f32>,
        c: &Point3<f32>,
    ) -> bool {
        //edges vectors
        let a_to_b = *b - *a; //vector from a to b 
        let a_to_c = *c - *a;

        // Begin calculating determinant - also used to calculate u parameter
        // u_vec lies in view plane
        // length of a_to_c in view_plane = |u_vec| = |a_to_c|*sin(a_to_c, dir)
        let u_vec = self.direction.cross(&a_to_c); 

        // If determinant is near zero, ray lies in plane of triangle
        // The determinant corresponds to the parallelepiped volume:
        // det = 0 => [dir, a_to_b, a_to_c] not linearly independant
        let det = a_to_b.dot(&u_vec);

        // Only testing positive bound, thus enabling backface culling
        // If backface culling is not desired write:
        // det < EPSILON && det > -EPSILON
        // instead of
        // det < EPSILON => changed to no backculling
        // if det < EPSILON && det > -EPSILON {
        //     return false;
        // }
        if det < f32::EPSILON && det > -f32::EPSILON {  //small floating point
            return false;  //ray and triangle almost parallel/coplanar => no intersection
        }

        let inv_det = 1.0 / det;

        // Vector from point a to ray origin
        let a_to_origin = self.origin - *a; 

        // Calculate u parameter
        let u = a_to_origin.dot(&u_vec) * inv_det;

        // Test bounds: u < 0 || u > 1 => outside of triangle
        if !(0.0..=1.0).contains(&u) {
            return false;
        }

        // Prepare to test v parameter
        let v_vec = a_to_origin.cross(&a_to_b);

        // Calculate v parameter and test bound
        let v = self.direction.dot(&v_vec) * inv_det;

        // The intersection lies outside of the triangle
        if v < 0.0 || u + v > 1.0 {
            return false;
        }

        let dist = a_to_c.dot(&v_vec) * inv_det; //distance between the ray origin and the interseption point

        if dist > f32::EPSILON {
            return true;
        } else {
            return false; //the ray is pointing away from the triangle
        }
    }
}
