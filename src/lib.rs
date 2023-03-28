use bvh::ray::Ray;
use bvh::{Point3, Vector3, EPSILON};
use pyo3::prelude::*;
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use std::fs::OpenOptions;

/// A Python module implemented in Rust.
#[pymodule]
fn nmr_stl_python(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(calc_nmr, m)?)?;
    Ok(())
}

#[pyfunction]
fn calc_nmr(a: Vec<f32>, b: Vec<f32>) -> PyResult<()> {
    //let m1 = Point3::new(-1.0, 0.0, 0.0);
    //let m1 = Point3::new(0.0, -0.57728, 0.81654);
    //let m2 = Point3::new(0.0, 0.81654, 0.57728);
    let m1 = Point3::new(a[0], a[1], a[2]);
    let m2 = Point3::new(b[0], b[1], b[2]);
    let triangles = make_triangles();
    let mut rng = SmallRng::from_entropy();
    let mut pos = make_nv_locations();

    let mut proton_count = 0;
    while proton_count < 10_000 {
        let ray = Ray::new(
            Point3::new(
                rng.gen::<f32>() * 2.0 - 1.0,
                rng.gen::<f32>() * 2.0 - 1.0,
                rng.gen::<f32>() * 0.1,
            ), // Origin
            Vector3::new(1.0, 0.0, 0.0), // Direction
        );
        let mut intersect_count = 0;
        for triangle in &triangles {
            let intersect = ray.intersects_triangle_no_cull(&triangle.a, &triangle.b, &triangle.c);
            if intersect {
                intersect_count += 1;
            }
        }
        if intersect_count % 2 == 0 {
            proton_count += 1;
            dd_for_all_pos(ray.origin, m1, m2, &mut pos)
        }
    }
    write_result(&pos);
    Ok(())
}

fn write_result(pos: &Vec<NVLocation>) {
    let mut wtr = csv::Writer::from_path("out.csv").unwrap();
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

fn make_triangles() -> Vec<Triangle> {
    let mut file = OpenOptions::new()
        .read(true)
        .open("simple_chip.stl")
        .unwrap();
    let stl = stl_io::read_stl(&mut file).unwrap();
    let mut triangles = std::vec::Vec::new();
    for face in stl.faces {
        if face.normal[0] != 0.0 {
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

fn dd_for_all_pos(ray_origin: Point3, m1: Point3, m2: Point3, pos: &mut Vec<NVLocation>) {
    for nv in pos {
        nv.interaction += dipole_dipole(ray_origin - nv.loc, m1, m2);
    }
}

fn make_nv_locations() -> Vec<NVLocation> {
    let mut nv_locations = std::vec::Vec::new();
    for x in -100..100 {
        for y in -100..100 {
            for z in 1..2 {
                let new_location = NVLocation {
                    loc: Point3::new(x as f32 / 50.0, y as f32 / 50.0, z as f32 / 1000.0),
                    interaction: 0.0,
                };
                nv_locations.push(new_location);
            }
        }
    }
    return nv_locations;
}

// The Units are completely wrong in the following!!!
// mu0 is in SI base whereas r, m1 and m2 are in mm.
fn dipole_dipole(r: Point3, m1: Point3, m2: Point3) -> f32 {
    let mu0 = 0.00000125663706212; // N⋅A−2;
    let k = mu0 / (4.0 * std::f32::consts::PI);
    let r_norm = r.length();

    let interaction = -k / r_norm.powi(3) * (3.0 * m1.dot(r) * m2.dot(r) + m1.dot(m2));
    return interaction;
}

#[derive(Debug)]
struct Triangle {
    a: Point3,
    b: Point3,
    c: Point3,
}

#[derive(Debug)]
struct NVLocation {
    loc: Point3,
    interaction: f32,
}

trait IntersectTriangle {
    fn intersects_triangle_no_cull(&self, a: &Point3, b: &Point3, c: &Point3) -> bool;
}

impl IntersectTriangle for Ray {
    fn intersects_triangle_no_cull(&self, a: &Point3, b: &Point3, c: &Point3) -> bool {
        let a_to_b = *b - *a;
        let a_to_c = *c - *a;

        // Begin calculating determinant - also used to calculate u parameter
        // u_vec lies in view plane
        // length of a_to_c in view_plane = |u_vec| = |a_to_c|*sin(a_to_c, dir)
        let u_vec = self.direction.cross(a_to_c);

        // If determinant is near zero, ray lies in plane of triangle
        // The determinant corresponds to the parallelepiped volume:
        // det = 0 => [dir, a_to_b, a_to_c] not linearly independant
        let det = a_to_b.dot(u_vec);

        // Only testing positive bound, thus enabling backface culling
        // If backface culling is not desired write:
        // det < EPSILON && det > -EPSILON
        // instead of
        // det < EPSILON => changed to no backculling
        if det < EPSILON && det > -EPSILON {
            return false;
        }

        let inv_det = 1.0 / det;

        // Vector from point a to ray origin
        let a_to_origin = self.origin - *a;

        // Calculate u parameter
        let u = a_to_origin.dot(u_vec) * inv_det;

        // Test bounds: u < 0 || u > 1 => outside of triangle
        if !(0.0..=1.0).contains(&u) {
            return false;
        }

        // Prepare to test v parameter
        let v_vec = a_to_origin.cross(a_to_b);

        // Calculate v parameter and test bound
        let v = self.direction.dot(v_vec) * inv_det;
        // The intersection lies outside of the triangle
        if v < 0.0 || u + v > 1.0 {
            return false;
        }

        let dist = a_to_c.dot(v_vec) * inv_det;

        if dist > EPSILON {
            return true;
        } else {
            return false;
        }
    }
}
