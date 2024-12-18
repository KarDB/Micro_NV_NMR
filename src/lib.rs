// use bvh::Point3;
use nalgebra::Vector3;
use pyo3::prelude::*;
mod linear_algebra;
mod nmr_and_mesh;
mod types;

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
    n_prot: usize,
    filepath: String,
    stlfile: String,
    resolution_x: u32,
    resolution_y: u32,
    diffusion_coefficient: f32,
    frequency: f32,
    number_time_steps: usize,
    timestep: f32,
    zero: bool,
    // parallelization_level: usize,
) -> PyResult<f32> {
    let m1 = Vector3::new(a[0], a[1], a[2]);
    let m2 = Vector3::new(b[0], b[1], b[2]);
    let volume = nmr_and_mesh::start_sim(
        m1,
        m2,
        nv_depth,
        n_prot,
        filepath,
        stlfile,
        resolution_x,
        resolution_y,
        diffusion_coefficient,
        frequency,
        number_time_steps,
        timestep,
        zero,
        // parallelization_level,
    );
    Ok(volume)
}
