use bevy::prelude::*;
use bevy::render::render_resource::{
    BindGroupLayout, BindGroupLayoutEntry, BindingType, BufferBindingType, ShaderStages,
};
use bevy::render::renderer::RenderDevice;

// Bind group layout for the grid-build passes:
// binding(0): counts (storage, rw)
// binding(1): GridBuildParams (uniform)
#[derive(Resource, Clone)]
pub struct GridBuildBindGroupLayout(pub BindGroupLayout);

/// Create the layout in the Render world (runs once)
pub fn init_grid_build_bind_group_layout(mut commands: Commands, render_device: Res<RenderDevice>) {
    let layout = render_device.create_bind_group_layout(
        Some("grid_build_bind_group_layout"),
        &[
            // binding 0: counts (rw storage)
            BindGroupLayoutEntry {
                binding: 0,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            // binding 1: GridBuildParams (uniform)
            BindGroupLayoutEntry {
                binding: 1,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
    );

    commands.insert_resource(GridBuildBindGroupLayout(layout));
}
