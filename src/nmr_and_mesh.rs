use bvh::ray::Ray;
use nalgebra::{Point3, Vector3};
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use rand_distr::{Distribution, Normal, NormalError};
// use rayon::prelude::*;
use nalgebra::Rotation3;
use std::fs::OpenOptions;

pub fn start_sim(
    mut m1: Vector3<f32>,
    mut m2: Vector3<f32>,
    nv_depth: f32,
    n_prot: u32,
    filepath: String,
    stlfile: String,
    resolution_x: u32,
    resolution_y: u32,
    diffusion_stepsize: f32,
    diffusion_numer_steps: u32,
) -> f32 {
    let triangles = make_triangles(stlfile.clone());
    let dimensions = get_dims(stlfile);
    let mut rng = SmallRng::from_entropy();
    let mut pos = make_nv_locations(dimensions, nv_depth, resolution_x, resolution_y);
    let volume = get_chip_volume(&dimensions);
    let mut proton_count: u32 = 0;
    let mut total_count: u32 = 0;
    // let rotation_matrix = make_rotation(&mut m1, 12.0);
    while proton_count < n_prot {
        let mut proton = make_proton_position(
            &mut rng,
            &dimensions,
            &mut total_count,
            &mut proton_count,
            &triangles,
        );
        diffuse_proton(&mut proton, &mut rng, &diffusion_stepsize);
        dd_for_all_pos(proton.origin, m1, m2, &mut pos)
    }
    write_result(&pos, filepath);
    return volume * (proton_count as f32) / (total_count as f32);
}

// fn make_rotation(m1: nalgebra::Vector3<f32>, angle: f32) {
//     let rotation_matrix = Rotation3::new(m1.normalize() * angle);
// }

fn generate_gaussian(rng: &mut SmallRng, std: &f32) -> f32 {
    let mean = 0.0;
    let normal_dist = Normal::new(mean, std.to_owned());
    match normal_dist {
        Ok(dist) => dist.sample(rng),
        Err(NormalError::BadVariance) => 0.0,
        Err(NormalError::MeanTooSmall) => panic!("MeanToSmall error should not exists here!"),
    }
}

fn diffuse_proton(ray: &mut Ray<f32, 3>, rng: &mut SmallRng, stepsize: &f32) {
    let position_update = Vector3::new(
        generate_gaussian(rng, stepsize),
        generate_gaussian(rng, stepsize),
        generate_gaussian(rng, stepsize),
    );
    ray.origin += position_update;
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
    total_count: &mut u32,
    proton_count: &mut u32,
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

#[derive(Debug, Clone, Copy)]
struct ChipDimensions {
    min_x: f32,
    max_x: f32,
    min_y: f32,
    max_y: f32,
    min_z: f32,
    max_z: f32,
}

fn make_triangles(stlfile: String) -> Vec<Triangle> {
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

#[derive(Debug)]
struct Triangle {
    a: Point3<f32>,
    b: Point3<f32>,
    c: Point3<f32>,
}

#[derive(Debug)]
struct NVLocation {
    loc: Point3<f32>,
    interaction: f32,
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
