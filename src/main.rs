mod nmr_and_mesh;
use bvh::Point3;
use serde::Deserialize;

fn main() {
    let config = load_config();
    let m1 = Point3::new(config.m1x, config.m1y, config.m1z);
    let m2 = Point3::new(config.m2x, config.m2y, config.m2z);
    nmr_and_mesh::start_sim(
        m1,
        m2,
        config.nv_depth,
        config.proton_count,
        config.output_file,
        config.stl_file,
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
    proton_count: u32,
    output_file: String,
    stl_file: String,
}

fn load_config() -> Config {
    let f = std::fs::File::open("config.yaml").expect("Could not open file.");
    let scrape_config: Config = serde_yaml::from_reader(f).expect("Could not read values.");
    return scrape_config;
}