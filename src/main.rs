mod linear_algebra;
mod nmr_and_mesh;
mod types;
// use bvh::Point3;
use nalgebra::Vector3;
use serde::Deserialize;

fn main() {
    let config = load_config();
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
        config.angular_frequency,
        config.diffusion_number_steps,
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
    angular_frequency: f32,
    diffusion_number_steps: usize,
}

fn load_config() -> Config {
    let f = std::fs::File::open("config.yaml").expect("Could not open file.");
    let scrape_config: Config = serde_yaml::from_reader(f).expect("Could not read values.");
    return scrape_config;
}
