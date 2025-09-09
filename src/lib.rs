use bevy::prelude::*;

pub mod solid_color;

pub mod cpu {
    pub mod sph2d;
}

pub mod gpu {
    pub mod ffi;
    pub mod buffers;
    pub mod pipeline;
}

#[derive(Component)]
pub struct SceneControl {
    pub target: ControlTarget,
    pub speed: f32, 
}

#[derive(Component, Copy, Clone)]
pub struct Rotates {
    pub axis: Vec3,
    pub speed: f32, // radians per seconds
    pub mode: RotationMode,
}

#[derive(Resource, PartialEq, Debug, Copy, Clone)]
pub enum ControlTarget {
    Camera, 
    Light, 
}

#[derive(Debug, Copy, Clone)]
pub enum RotationMode {
    SpinInPlace,
    OrbitAround, // assumes center = Vec3::ZERO for now
}