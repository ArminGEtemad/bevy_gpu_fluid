use crate::gpu::buffers::{ExtractedGrid, ExtractedParticleBuffer};
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

#[derive(Resource, Clone)]
pub struct GridHistogramBindGroupLayout(pub BindGroupLayout);

#[derive(Resource)]
pub struct GridHistogramBindGroup(pub BindGroup);

#[derive(Resource)]
pub struct GridStartsBuffer {
    pub buffer: Buffer,
    pub num_cells: u32,
}

#[derive(Resource, Clone)]
pub struct GridCountsToStartsBindGroupLayout(pub BindGroupLayout);

#[derive(Resource)]
pub struct GridCountsToStartsBindGroup(pub BindGroup);

#[derive(Resource)]
pub struct GridBlockSumsBuffer {
    pub buffer: Buffer,
    pub num_blocks: u32,
}

// BGL for block_scan: 0=counts(ro), 1=starts(rw), 2=block_sums(rw)
#[derive(Resource, Clone)]
pub struct GridBlockScanBindGroupLayout(pub BindGroupLayout);

#[derive(Resource)]
pub struct GridBlockScanBindGroup(pub BindGroup);
#[derive(Resource, Clone)]
pub struct BlockSumsScanBindGroupLayout(pub BindGroupLayout);

#[derive(Resource, Clone)]
pub struct AddBackBindGroupLayout(pub BindGroupLayout);

#[derive(Resource)]
pub struct AddBackBindGroup(pub BindGroup);

#[derive(Resource)]
pub struct BlockSumsScanBindGroup(pub BindGroup);

#[derive(Resource)]
pub struct GridCursorBuffer {
    pub buffer: Buffer,
    pub num_cells: u32,
}

#[derive(Resource)]
pub struct GridOverflowCounter {
    pub buffer: Buffer, // a debug struct
}

#[derive(Resource, Clone)]
pub struct GridScatterBindGroupLayout(pub BindGroupLayout);

#[derive(Resource)]
pub struct GridScatterBindGroup(pub BindGroup);

#[derive(Resource)]
pub struct ClearCursorBindGroup(pub bevy::render::render_resource::BindGroup);

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

pub fn init_grid_histogram_bind_group_layout(
    mut commands: Commands,
    render_device: Res<RenderDevice>,
) {
    // binding(0): particles SSBO
    // binding(1): counts SSBO
    // binding(2): GridParams UBO
    let layout = render_device.create_bind_group_layout(
        Some("grid_histogram_bgl"),
        &[
            BindGroupLayoutEntry {
                binding: 0,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            BindGroupLayoutEntry {
                binding: 1,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            BindGroupLayoutEntry {
                binding: 2,
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

    commands.insert_resource(GridHistogramBindGroupLayout(layout));
}
pub fn init_grid_histogram_bind_group(
    mut commands: Commands,
    render_device: Res<RenderDevice>,
    layout: Option<Res<GridHistogramBindGroupLayout>>,
    particles: Option<Res<ExtractedParticleBuffer>>,
    counts: Option<Res<GridCountsBuffer>>,
    grid: Option<Res<ExtractedGrid>>,
) {
    let (Some(layout), Some(particles), Some(counts), Some(grid)) =
        (layout, particles, counts, grid)
    else {
        return;
    };

    let bind_group = render_device.create_bind_group(
        Some("grid_histogram_bg"),
        &layout.0,
        &[
            BindGroupEntry {
                binding: 0,
                resource: particles.buffer.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 1,
                resource: counts.buffer.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 2,
                resource: grid.params_buf.as_entire_binding(),
            },
        ],
    );

    commands.insert_resource(GridHistogramBindGroup(bind_group));
}

pub fn init_counts_to_starts_bgl(mut commands: Commands, render_device: Res<RenderDevice>) {
    let layout = render_device.create_bind_group_layout(
        Some("grid_counts_to_starts_bgl"),
        &[
            // counts (read-only STORAGE); we will read via atomicLoad in WGSL
            BindGroupLayoutEntry {
                binding: 0,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            // starts (rw STORAGE)
            BindGroupLayoutEntry {
                binding: 1,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
    );
    commands.insert_resource(GridCountsToStartsBindGroupLayout(layout));
}

pub fn init_starts_buffer_and_bg(
    mut commands: Commands,
    render_device: Res<RenderDevice>,
    extracted_grid: Option<Res<ExtractedGrid>>,
    counts: Option<Res<GridCountsBuffer>>,
    layout: Option<Res<GridCountsToStartsBindGroupLayout>>,
    existing: Option<Res<GridStartsBuffer>>,
) {
    let (Some(grid), Some(counts), Some(layout)) = (extracted_grid, counts, layout) else {
        return;
    };

    let num_cells = grid.num_cells as u32;
    if num_cells == 0 {
        return;
    }

    // no-op if already correct size
    if let Some(starts) = existing {
        if starts.num_cells == num_cells {
            return;
        }
    }

    let size_bytes = (grid.num_cells.max(1) * std::mem::size_of::<u32>()) as u64;
    let starts_buf = render_device.create_buffer(&BufferDescriptor {
        label: Some("grid_starts"),
        size: size_bytes,
        usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    // store the buffer resource
    let starts_res = GridStartsBuffer {
        buffer: starts_buf,
        num_cells,
    };
    // create a bind group for the future counts->starts pass
    let bg = render_device.create_bind_group(
        Some("grid_counts_to_starts_bg"),
        &layout.0,
        &[
            BindGroupEntry {
                binding: 0,
                resource: counts.buffer.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 1,
                resource: starts_res.buffer.as_entire_binding(),
            },
        ],
    );

    commands.insert_resource(starts_res);
    commands.insert_resource(GridCountsToStartsBindGroup(bg));
}

pub fn init_block_scan_bgl(mut commands: Commands, render_device: Res<RenderDevice>) {
    let layout = render_device.create_bind_group_layout(
        Some("grid_block_scan_bgl"),
        &[
            BindGroupLayoutEntry {
                binding: 0,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            BindGroupLayoutEntry {
                binding: 1,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            BindGroupLayoutEntry {
                binding: 2,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
    );
    commands.insert_resource(GridBlockScanBindGroupLayout(layout));
}

pub fn init_block_sums_and_bg(
    mut commands: Commands,
    render_device: Res<RenderDevice>,
    grid: Option<Res<ExtractedGrid>>,
    counts: Option<Res<GridCountsBuffer>>,
    starts: Option<Res<GridStartsBuffer>>,
    layout: Option<Res<GridBlockScanBindGroupLayout>>,
    existing: Option<Res<GridBlockSumsBuffer>>,
) {
    let (Some(grid), Some(counts), Some(starts), Some(layout)) = (grid, counts, starts, layout)
    else {
        return;
    };

    // one block per 256 cells (ceil)
    let num_cells = grid.num_cells as u32;
    if num_cells == 0 {
        return;
    }
    let num_blocks = ((num_cells + 255) / 256).max(1);

    if let Some(bs) = &existing {
        if bs.num_blocks == num_blocks {
            // still (re)create BG in case buffers changed
        }
    }

    let block_sums_size = (num_blocks as usize * std::mem::size_of::<u32>()) as u64;
    let block_sums_buf = render_device.create_buffer(&BufferDescriptor {
        label: Some("grid_block_sums"),
        size: block_sums_size.max(4),
        usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let block_sums_res = GridBlockSumsBuffer {
        buffer: block_sums_buf,
        num_blocks,
    };

    let bg = render_device.create_bind_group(
        Some("grid_block_scan_bg"),
        &layout.0,
        &[
            BindGroupEntry {
                binding: 0,
                resource: counts.buffer.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 1,
                resource: starts.buffer.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 2,
                resource: block_sums_res.buffer.as_entire_binding(),
            },
        ],
    );

    commands.insert_resource(block_sums_res);
    commands.insert_resource(GridBlockScanBindGroup(bg));
}

pub fn init_block_sums_scan_bgl(mut commands: Commands, rd: Res<RenderDevice>) {
    let layout = rd.create_bind_group_layout(
        Some("grid_block_sums_scan_bgl"),
        &[BindGroupLayoutEntry {
            binding: 0,
            visibility: ShaderStages::COMPUTE,
            ty: BindingType::Buffer {
                ty: BufferBindingType::Storage { read_only: false },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        }],
    );
    commands.insert_resource(BlockSumsScanBindGroupLayout(layout));
}

pub fn init_block_sums_scan_bg(
    mut commands: Commands,
    rd: Res<RenderDevice>,
    layout: Option<Res<BlockSumsScanBindGroupLayout>>,
    bs: Option<Res<GridBlockSumsBuffer>>,
) {
    let (Some(layout), Some(bs)) = (layout, bs) else {
        return;
    };
    let bg = rd.create_bind_group(
        Some("grid_block_sums_scan_bg"),
        &layout.0,
        &[BindGroupEntry {
            binding: 0,
            resource: bs.buffer.as_entire_binding(),
        }],
    );
    commands.insert_resource(BlockSumsScanBindGroup(bg));
}

pub fn init_add_back_bgl(mut commands: Commands, rd: Res<RenderDevice>) {
    let layout = rd.create_bind_group_layout(
        Some("grid_add_back_bgl"),
        &[
            BindGroupLayoutEntry {
                binding: 1,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            BindGroupLayoutEntry {
                binding: 2,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
    );
    commands.insert_resource(AddBackBindGroupLayout(layout));
}

pub fn init_add_back_bg(
    mut commands: Commands,
    rd: Res<RenderDevice>,
    layout: Option<Res<AddBackBindGroupLayout>>,
    starts: Option<Res<GridStartsBuffer>>,
    blocks: Option<Res<GridBlockSumsBuffer>>,
) {
    let (Some(layout), Some(starts), Some(blocks)) = (layout, starts, blocks) else {
        return;
    };

    let bg = rd.create_bind_group(
        Some("grid_add_back_bg"),
        &layout.0,
        &[
            BindGroupEntry {
                binding: 1,
                resource: starts.buffer.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 2,
                resource: blocks.buffer.as_entire_binding(),
            },
        ],
    );
    commands.insert_resource(AddBackBindGroup(bg));
}

pub fn init_scatter_bgl(mut commands: Commands, rd: Res<RenderDevice>) {
    let layout = rd.create_bind_group_layout(
        Some("grid_scatter_bgl"),
        &[
            // 0: particles
            BindGroupLayoutEntry {
                binding: 0,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            // 1: starts
            BindGroupLayoutEntry {
                binding: 1,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            // 2: GridParams
            BindGroupLayoutEntry {
                binding: 2,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            // 3: entries
            BindGroupLayoutEntry {
                binding: 3,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            // 4: cursor
            BindGroupLayoutEntry {
                binding: 4,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
    );
    commands.insert_resource(GridScatterBindGroupLayout(layout));
}

pub fn init_scatter_resources_and_bg(
    mut commands: Commands,
    rd: Res<RenderDevice>,
    layout: Option<Res<GridScatterBindGroupLayout>>,
    grid: Option<Res<ExtractedGrid>>,
    particles: Option<Res<ExtractedParticleBuffer>>,
    starts: Option<Res<GridStartsBuffer>>,
    entries_grid: Option<Res<ExtractedGrid>>,
    existing_cursor: Option<Res<GridCursorBuffer>>,
    existing_overflow: Option<Res<GridOverflowCounter>>,
) {
    let (Some(layout), Some(grid_res), Some(particles), Some(starts), Some(entries_grid)) =
        (layout, grid, particles, starts, entries_grid)
    else {
        return;
    };

    let num_cells = grid_res.num_cells as u32;
    if num_cells == 0 {
        return;
    }

    let mut new_cursor_buf: Option<Buffer> = None;
    let need_cursor = existing_cursor
        .as_ref()
        .map(|c| c.num_cells != num_cells)
        .unwrap_or(true);

    if need_cursor {
        let size_bytes = ((grid_res.num_cells + 1).max(1) * std::mem::size_of::<u32>()) as u64;
        let buf = rd.create_buffer(&BufferDescriptor {
            label: Some("grid_cursor"),
            size: size_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        new_cursor_buf = Some(buf);
    }
    let cursor_buf_ref: &Buffer = if let Some(ref buf) = new_cursor_buf {
        buf
    } else {
        &existing_cursor.as_ref().unwrap().buffer
    };

    // overflow counter (not used yet in BG)
    if existing_overflow.is_none() {
        let overflow_buf = rd.create_buffer(&BufferDescriptor {
            label: Some("grid_overflow_counter"),
            size: 4,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        commands.insert_resource(GridOverflowCounter {
            buffer: overflow_buf,
        });
    }

    let bg = rd.create_bind_group(
        Some("grid_scatter_bg"),
        &layout.0,
        &[
            BindGroupEntry {
                binding: 0,
                resource: particles.buffer.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 1,
                resource: starts.buffer.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 2,
                resource: grid_res.params_buf.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 3,
                resource: entries_grid.entries_buf.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 4,
                resource: cursor_buf_ref.as_entire_binding(),
            },
        ],
    );
    commands.insert_resource(GridScatterBindGroup(bg));

    if let Some(buf) = new_cursor_buf {
        commands.insert_resource(GridCursorBuffer {
            buffer: buf,
            num_cells,
        });
    }
}

pub fn init_clear_cursor_bg(
    mut commands: Commands,
    rd: Res<RenderDevice>,
    layout: Option<Res<GridBuildBindGroupLayout>>,
    cursor: Option<Res<GridCursorBuffer>>,
    gb: Option<Res<GridBuildParamsBuffer>>,
) {
    let (Some(layout), Some(cursor), Some(gb)) = (layout, cursor, gb) else {
        return;
    };

    let bg = rd.create_bind_group(
        Some("grid_clear_cursor_bg"),
        &layout.0,
        &[
            // binding(0) -> cursor buffer (rw storage)
            BindGroupEntry {
                binding: 0,
                resource: cursor.buffer.as_entire_binding(),
            },
            // binding(1) -> GridBuildParams uniform
            BindGroupEntry {
                binding: 1,
                resource: gb.buffer.as_entire_binding(),
            },
        ],
    );
    commands.insert_resource(ClearCursorBindGroup(bg));
}
