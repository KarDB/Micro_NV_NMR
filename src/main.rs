mod linear_algebra;
mod nmr_and_mesh;
mod types;
// use bvh::Point3;
use nalgebra::Vector3;
use serde::Deserialize;
use std::env;

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: {} <config_file>", args[0]);
        std::process::exit(1);
    }

    let config_file = &args[1];
    let config = load_config(config_file);
    let m1 = Vector3::new(config.m1x, config.m1y, config.m1z);
    let m2 = Vector3::new(config.m2x, config.m2y, config.m2z);
    nmr_and_mesh::start_sim(
        m1,
        m2,
        config.nv_depth,
        config.proton_count,
        config.output_file,
        config.stl_file,
        config.resolution_x,
        config.resolution_y,
        config.diffusion_coefficient,
        config.frequency,
        config.number_time_steps,
        config.timestep,
        config.zero,
        // config.parallelization_level,
    );
}

#[derive(Debug, Deserialize)]
struct Config {
    m1x: f32,
    m1y: f32,
    m1z: f32,
    m2x: f32,
    m2y: f32,
    m2z: f32,
    nv_depth: f32,
    proton_count: usize,
    output_file: String,
    stl_file: String,
    resolution_x: u32,
    resolution_y: u32,
    diffusion_coefficient: f32,
    frequency: f32,
    number_time_steps: usize,
    timestep: f32,
    zero: bool,
    // #[serde(default = "default_parallelization")]
    // parallelization_level: usize,
}

#[allow(dead_code)]
fn default_parallelization() -> usize {
    1
}

fn load_config(config_file: &str) -> Config {
    let f = std::fs::File::open(config_file).expect("Could not open file.");
    let scrape_config: Config = serde_yaml::from_reader(f).expect("Could not read values.");
    return scrape_config;
}
