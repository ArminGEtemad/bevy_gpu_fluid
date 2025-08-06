use bytemuck::{Pod, Zeroable};

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct GPUParticle {
    // not using glam to make sure WGSL compatibility
    pub pos: [f32; 2], 
    pub vel: [f32; 2],
    pub rho: f32,
    pub p: f32,
}

