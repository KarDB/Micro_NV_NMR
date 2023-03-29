use bvh::Point3;
use pyo3::prelude::*;
mod nmr_and_mesh;

/// A Python module implemented in Rust.
#[pymodule]
fn nmr_stl_python(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(calc_nmr, m)?)?;
    Ok(())
}

#[pyfunction]
fn calc_nmr(
    a: Vec<f32>,
    b: Vec<f32>,
    nv_depth: f32,
    n_prot: u32,
    filepath: String,
    stlfile: String,
) -> PyResult<()> {
    //let m1 = Point3::new(-1.0, 0.0, 0.0);
    //let m1 = Point3::new(0.0, -0.57728, 0.81654);
    //let m2 = Point3::new(0.0, 0.81654, 0.57728);
    let m1 = Point3::new(a[0], a[1], a[2]);
    let m2 = Point3::new(b[0], b[1], b[2]);
    nmr_and_mesh::start_sim(m1, m2, nv_depth, n_prot, filepath, stlfile);
    Ok(())
}
