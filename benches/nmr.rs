use bvh::Point3;
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;
use std::sync::Mutex;
use std::vec::Vec;

pub fn start(m1: Point3, m2: Point3, pos: &Vec<NVLocation>) {
    let mut rng = SmallRng::from_entropy();
    let mut proton_count = 0;
    let mut protons = Vec::new();
    while proton_count < 10_000 {
        let ray_origin = Point3::new(rng.gen::<f32>(), rng.gen::<f32>(), rng.gen::<f32>());
        protons.push(ray_origin);
        proton_count += 1;
    }
    protons
        .par_iter()
        .for_each(|proton| dd_for_all_pos(*proton, m1, m2, pos))
}

pub fn dd_for_all_pos(ray_origin: Point3, m1: Point3, m2: Point3, pos: &Vec<NVLocation>) {
    for nv in pos.iter() {
        let mut nvinteract = nv.interaction.lock().unwrap();
        *nvinteract += dipole_dipole(ray_origin - nv.loc, m1, m2)
    }

    //for nv in pos {
    //    nv.interaction += dipole_dipole(ray_origin - nv.loc, m1, m2);
    //}
}

pub fn make_nv_locations(nv_depth: f32) -> Vec<NVLocation> {
    let mut nv_locations = std::vec::Vec::new();
    for x in -100..100 {
        for y in -100..100 {
            let new_location = NVLocation {
                loc: Point3::new(x as f32 / 50.0, y as f32 / 50.0, nv_depth / 1000.0),
                interaction: Mutex::from(0.0),
            };
            nv_locations.push(new_location);
        }
    }
    return nv_locations;
}

// The Units are completely wrong in the following!!!
// mu0 is in SI base whereas r, m1 and m2 are in mm.
pub fn dipole_dipole(r: Point3, m1: Point3, m2: Point3) -> f32 {
    let mu0 = 0.00000125663706212; // N⋅A−2;
    let k = mu0 / (4.0 * std::f32::consts::PI);
    let r_norm = r.length();
    let r_unit = r / r_norm;

    let interaction = -k / r_norm.powi(3) * (3.0 * m1.dot(r_unit) * m2.dot(r_unit) + m1.dot(m2));
    return interaction;
}

#[derive(Debug)]
pub struct NVLocation {
    loc: Point3,
    interaction: Mutex<f32>,
}
