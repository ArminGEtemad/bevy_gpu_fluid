use bevy::prelude::*;
use bevy::render::render_resource::{
    BindGroup, BindGroupEntry, BindGroupLayout, BindGroupLayoutEntry, BindingType, Buffer,
    BufferBindingType, BufferDescriptor, BufferInitDescriptor, BufferUsages, ShaderStages,
};
use bevy::render::renderer::{RenderDevice, RenderQueue};

// Bind group layout for the grid-build passes:
// binding(0): counts (storage, rw)
// binding(1): GridBuildParams (uniform)
#[derive(Resource, Clone)]
pub struct GridBuildBindGroupLayout(pub BindGroupLayout);

#[derive(Resource)]
pub struct GridCountsBuffer {
    pub buffer: Buffer,
    pub num_cells: u32,
}

#[derive(Resource)]
pub struct GridBuildParamsBuffer {
    pub buffer: Buffer,
    pub value: crate::gpu::ffi::GridBuildParams,
}

// binding 0 = counts, 1 = params
#[derive(Resource)]
pub struct GridBuildBindGroup(pub BindGroup);

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

pub fn init_grid_build_buffers(
    mut commands: Commands,
    render_device: Res<RenderDevice>,
    _queue: Res<RenderQueue>,
    layout: Option<Res<GridBuildBindGroupLayout>>,
    extracted_grid: Option<Res<crate::gpu::buffers::ExtractedGrid>>,
) {
    let (Some(layout), Some(grid)) = (layout, extracted_grid) else {
        return; // layout or grid not ready this frame
    };

    let num_cells_usize = grid.num_cells;
    let num_cells = num_cells_usize as u32;

    let counts_size_bytes = (num_cells_usize.max(1) * std::mem::size_of::<u32>()) as u64;

    let counts = render_device.create_buffer(&BufferDescriptor {
        label: Some("grid_counts"),
        size: counts_size_bytes,
        usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let gb_val = crate::gpu::ffi::GridBuildParams {
        num_cells,
        _pad: [0; 7],
    };
    let gb_buf = render_device.create_buffer_with_data(&BufferInitDescriptor {
        label: Some("grid_build_params"),
        contents: bytemuck::bytes_of(&gb_val),
        usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
    });

    let bind_group = render_device.create_bind_group(
        Some("grid_build_bind_group"),
        &layout.0,
        &[
            BindGroupEntry {
                binding: 0,
                resource: counts.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 1,
                resource: gb_buf.as_entire_binding(),
            },
        ],
    );

    commands.insert_resource(GridCountsBuffer {
        buffer: counts,
        num_cells,
    });
    commands.insert_resource(GridBuildParamsBuffer {
        buffer: gb_buf,
        value: gb_val,
    });
    commands.insert_resource(GridBuildBindGroup(bind_group));
}
