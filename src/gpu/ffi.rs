use bytemuck::{Pod, Zeroable};

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct GPUParticle {
    // not using glam to make sure WGSL compatibility
    pub pos: [f32; 2],
    pub vel: [f32; 2],
    pub acc: [f32; 2],
    pub rho: f32,
    pub p: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct GridParams {
    pub min_world: [f32; 2], // (min_ix, min_iy) * h
    pub cell_size: f32,
    pub _pad0: f32, // 16B alignment
    pub dims: [u32; 2],
    pub _pad1: [u32; 2], // 16B alignment
}
// 16B alignment for uniform buffers
