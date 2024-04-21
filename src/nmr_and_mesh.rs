use bvh::ray::Ray;
use hdf5::File;
use nalgebra::{Point3, Vector3};
use ndarray::{arr1, s, Array3};
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use rand_distr::{Distribution, Normal, NormalError};
// use rayon::prelude::*;
use crate::linear_algebra::{intersects_chip_walls, make_rotation, reflect_on_wall};
use crate::types::*;
use std::fs::OpenOptions;

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
    angular_frequency: f32,
    diffusion_numer_steps: usize,
    timestep: f32,
) -> f32 {
    let triangles = make_triangles(stlfile.clone());
    let triangles_all = make_all_triangles(stlfile.clone());
    let dimensions = get_dims(stlfile.clone());
    let mut rng = SmallRng::from_entropy();
    // let mut pos = make_nv_locations(dimensions, nv_depth, resolution_x, resolution_y);
    let mut pos = make_timeresolved_locations(
        dimensions,
        nv_depth,
        resolution_x,
        resolution_y,
        diffusion_numer_steps,
    );
    let volume = get_chip_volume(&dimensions);
    let mut proton_count: usize = 0;
    let mut total_count: usize = 0;
    let rotation_angle = get_rotation_angle(angular_frequency, timestep);
    let rotation_matrix = make_rotation(&m1, rotation_angle);
    let diffusion_stepsize = get_rms_diffusion_displacement(diffusion_coefficient, timestep);
    let mut proton_positions = make_timeresolved_proton_list(&n_prot, &diffusion_numer_steps);
    while proton_count < n_prot {
        let mut m2_current = m2.clone();
        let mut proton = make_proton_position(
            &mut rng,
            &dimensions,
            &mut total_count,
            &mut proton_count,
            &triangles,
        );
        for t in 0..diffusion_numer_steps as usize {
            dd_for_all_pos(proton.origin, m1, m2_current, &mut pos[t]);
            proton_positions
                .slice_mut(s![proton_count - 1, t, ..])
                .assign(&arr1(&[
                    proton.origin.x,
                    proton.origin.y,
                    proton.origin.z,
                    m2_current.x,
                    m2_current.y,
                    m2_current.z,
                ]));
            diffuse_proton(&mut proton, &mut rng, &diffusion_stepsize, &triangles_all);
            m2_current = rotation_matrix * m2_current;
        }
    }
    // write_result(&pos, filepath);
    let hdf5_data = convert_to_array3(&pos);
    // test struct
    let metadata = Metadata {
        stl_file: stlfile.clone().parse().unwrap(),
        experiment_id: 123,
        temperature: 37.5,
    };
    let _ = save_to_hdf5(&hdf5_data, &proton_positions, &metadata, filepath);
    return volume * (proton_count as f32) / (total_count as f32);
}

fn get_rms_diffusion_displacement(diffusion_coefficient: f32, timestep: f32) -> f32 {
    // We are working in 3D so the dimensionality factor is 6
    let dim_factor = 6.0;
    (dim_factor * timestep * diffusion_coefficient).sqrt()
}

fn get_rotation_angle(angular_frequency: f32, timestep: f32) -> f32 {
    angular_frequency * timestep
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
    ray: &mut Ray<f32, 3>,
    rng: &mut SmallRng,
    stepsize: &f32,
    triangles: &Vec<Triangle>,
) {
    let mut position_update = Vector3::new(
        generate_gaussian(rng, stepsize),
        generate_gaussian(rng, stepsize),
        generate_gaussian(rng, stepsize),
    );
    let mut counter = 0;
    loop {
        let wall_intersection = intersects_chip_walls(triangles, &ray.origin, &position_update);
        match wall_intersection {
            Some(_) => {
                position_update = Vector3::new(
                    generate_gaussian(rng, stepsize),
                    generate_gaussian(rng, stepsize),
                    generate_gaussian(rng, stepsize),
                );
                counter += 1;
            }
            None => {
                ray.origin += position_update;
                break;
            }
        }
    }
    // ray.origin += position_update;
}

fn get_chip_volume(dimensions: &ChipDimensions) -> f32 {
    let volume = (dimensions.max_x - dimensions.min_x)
        * (dimensions.max_y - dimensions.min_y)
        * (dimensions.max_z - dimensions.min_z);
    volume
}

fn make_proton_position(
    rng: &mut SmallRng,
    dimensions: &ChipDimensions,
    total_count: &mut usize,
    proton_count: &mut usize,
    triangles: &Vec<Triangle>,
) -> Ray<f32, 3> {
    loop {
        let ray = Ray::new(
            Point3::new(
                rng.gen::<f32>() * (dimensions.max_x - dimensions.min_x) + dimensions.min_x,
                rng.gen::<f32>() * (dimensions.max_y - dimensions.min_y) + dimensions.min_y,
                rng.gen::<f32>() * (dimensions.max_z - dimensions.min_z) + dimensions.min_z,
            ), // Origin
            Vector3::new(0.0, 0.0, 1.0), // Direction
        );
        let mut intersect_count = 0;
        for triangle in triangles {
            let intersect = ray.intersects_triangle_no_cull(&triangle.a, &triangle.b, &triangle.c);
            if intersect {
                intersect_count += 1;
            }
        }
        *total_count += 1;
        if intersect_count % 2 == 0 {
            *proton_count += 1;
            return ray;
        }
    }
}

fn write_result(pos: &Vec<NVLocation>, filepath: String) {
    let mut wtr = csv::Writer::from_path(filepath).unwrap();
    for nv in pos {
        wtr.write_record(&[
            nv.loc[0].to_string(),
            nv.loc[1].to_string(),
            nv.loc[2].to_string(),
            nv.interaction.to_string(),
        ])
        .unwrap();
    }
    wtr.flush().unwrap();
}

pub fn make_triangles(stlfile: String) -> Vec<Triangle> {
    let mut file = OpenOptions::new().read(true).open(stlfile).unwrap();
    let stl = stl_io::read_stl(&mut file).unwrap();
    let mut triangles = std::vec::Vec::new();
    //dbg!(&stl.vertices);
    for face in stl.faces {
        if face.normal[2] != 0.0 {
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
            triangles.push(tr);
        }
    }
    return triangles;
}

pub fn make_all_triangles(stlfile: String) -> Vec<Triangle> {
    let mut file = OpenOptions::new().read(true).open(stlfile).unwrap();
    let stl = stl_io::read_stl(&mut file).unwrap();
    let mut triangles = std::vec::Vec::new();
    //dbg!(&stl.vertices);
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
        triangles.push(tr);
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
    //dbg!(&chip_dimensions);
    return chip_dimensions;
}

fn make_timeresolved_proton_list(number_protons: &usize, time_steps: &usize) -> Array3<f32> {
    let proton_time_resolved_list =
        Array3::<f32>::zeros((number_protons.to_owned(), time_steps.to_owned(), 6));
    proton_time_resolved_list
}

fn make_timeresolved_locations(
    dimensions: ChipDimensions,
    nv_depth: f32,
    resolution_x: u32,
    resolution_y: u32,
    steps: usize,
) -> Vec<Vec<NVLocation>> {
    let mut timeresolved = std::vec::Vec::new();
    for _ in 0..steps {
        timeresolved.push(make_nv_locations(
            dimensions,
            nv_depth,
            resolution_x,
            resolution_y,
        ));
    }
    timeresolved
}

fn make_nv_locations(
    dimensions: ChipDimensions,
    nv_depth: f32,
    resolution_x: u32,
    resolution_y: u32,
) -> Vec<NVLocation> {
    let mut nv_locations = std::vec::Vec::new();
    let x_step = (dimensions.max_x - dimensions.min_x) / resolution_x as f32;
    let y_step = (dimensions.max_y - dimensions.min_y) / resolution_y as f32;
    for x in 0..resolution_x {
        let x = x as f32 * x_step + dimensions.min_x;
        for y in 0..resolution_y {
            let y = y as f32 * y_step + dimensions.min_y;
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
    let mu0 = 0.00000125663706212; // N⋅A−2;
    let k = mu0 / (4.0 * std::f32::consts::PI);
    let r_norm = r.norm();
    let r_unit = r / r_norm;

    let interaction = -k / r_norm.powi(3) * (3.0 * m1.dot(&r_unit) * m2.dot(&r_unit) - m1.dot(&m2));
    return interaction;
}

fn convert_to_array3(data: &Vec<Vec<NVLocation>>) -> Array3<f32> {
    let depth = data.len();
    let height = data.iter().map(|v| v.len()).max().unwrap_or(0);
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
    proton_positions: &Array3<f32>,
    metadata: &Metadata,
    filename: String,
) -> hdf5::Result<()> {
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
    let dataset_2 = file
        .new_dataset_builder()
        .with_data(proton_positions)
        .create("protons")
        .map_err(|e| {
            eprintln!("Failed to create dataset in file '{}': {}", filename, &e);
            e
        })?;

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
// https://github.com/svenstaro/bvh
impl IntersectTriangle for Ray<f32, 3> {
    fn intersects_triangle_no_cull(
        &self,
        a: &Point3<f32>,
        b: &Point3<f32>,
        c: &Point3<f32>,
    ) -> bool {
        let a_to_b = *b - *a;
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
        if det < f32::EPSILON && det > -f32::EPSILON {
            return false;
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

        let dist = a_to_c.dot(&v_vec) * inv_det;

        if dist > f32::EPSILON {
            return true;
        } else {
            return false;
        }
    }
}
