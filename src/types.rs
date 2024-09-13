use hdf5::types::VarLenUnicode;
use hdf5::H5Type;
use nalgebra::Point3;

#[derive(Debug, Clone, Copy)]
pub struct ChipDimensions {
    pub min_x: f32,
    pub max_x: f32,
    pub min_y: f32,
    pub max_y: f32,
    pub min_z: f32,
    pub max_z: f32,
}
#[derive(H5Type, Clone, PartialEq, Debug)]
#[repr(C)]
pub struct Metadata {
    pub stl_file: VarLenUnicode, // Using VarLenUnicode to handle string data
    pub timestep: f32,
    pub m1x: f32,
    pub m1y: f32,
    pub m1z: f32,
    pub m2x: f32,
    pub m2y: f32,
    pub m2z: f32,
    pub nv_depth: f32,
    pub proton_count: usize,
    pub resolution_x: u32,
    pub resolution_y: u32,
    pub diffusion_coefficient: f32,
    pub frequency: f32,
    pub number_time_steps: usize,
}
#[derive(Debug, Clone, Copy)]
pub struct Triangle {
    pub a: Point3<f32>,
    pub b: Point3<f32>,
    pub c: Point3<f32>,
}

#[derive(Debug)]
pub struct NVLocation {
    pub loc: Point3<f32>,
    pub interaction: f32,
}
