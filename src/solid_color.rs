use bevy::prelude::*;
use bevy::reflect::TypePath;
use bevy::render::render_resource::{AsBindGroup, ShaderRef};

#[derive(Asset, TypePath, AsBindGroup, Debug, Clone)]
pub struct SolidColor {
    #[uniform(0)]
    pub color: LinearRgba, // Color didn't work for uniform
}

impl Material for SolidColor {
    fn fragment_shader() -> ShaderRef {
        "shaders/solid_color.wgsl".into()
    }
}