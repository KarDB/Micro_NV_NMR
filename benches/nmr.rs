use bvh::Point3;
use rayon::prelude::*;

pub fn dd_for_all_pos(ray_origin: Point3, m1: Point3, m2: Point3, pos: &mut Vec<NVLocation>) {
    pos.par_iter_mut()
        .for_each(|nv| nv.interaction += dipole_dipole(ray_origin - nv.loc, m1, m2));

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
                interaction: 0.0,
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
    interaction: f32,
}
