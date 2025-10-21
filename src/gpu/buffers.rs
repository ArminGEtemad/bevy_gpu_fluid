use std::sync::Arc;
use std::sync::atomic::{AtomicU8, Ordering};

use bevy::prelude::*;
use bevy::render::extract_resource::ExtractResource;
use bevy::render::render_resource::{
    BindGroup, BindGroupEntry, BindGroupLayout, BindGroupLayoutEntry, BindingType, Buffer,
    BufferBindingType, BufferDescriptor, BufferInitDescriptor, BufferUsages, Maintain, MapMode,
    ShaderStages,
};
use bevy::render::renderer::{RenderDevice, RenderQueue};
use bevy::render::{Extract, ExtractSchedule, Render, RenderApp, RenderSet};

use crate::cpu::sph2d::SPHState;
use crate::gpu::ffi::{GPUParticle, GridParams, IntegrateParams};
use crate::gpu::grid_build::{
    init_add_back_bg, init_add_back_bgl, init_block_scan_bgl, init_block_sums_and_bg,
    init_block_sums_scan_bg, init_block_sums_scan_bgl, init_counts_to_starts_bgl,
    init_cursor_buffer_and_clear_bg, init_gpu_entries_buffer, init_grid_build_bind_group_layout,
    init_grid_build_buffers, init_grid_histogram_bind_group, init_grid_histogram_bind_group_layout,
    init_scatter_bg, init_scatter_bgl, init_starts_buffer_and_bg,
};
use crate::gpu::pipeline::{
    add_add_back_node_to_graph, add_block_scan_node_to_graph, add_block_sums_scan_node_to_graph,
    add_clear_counts_node_to_graph, add_clear_cursor_node_to_graph, add_density_node_to_graph,
    add_histogram_node_to_graph, add_scatter_node_to_graph, add_write_sentinel_node_to_graph,
    prepare_add_back_pipeline, prepare_block_scan_pipeline, prepare_block_sums_scan_pipeline,
    prepare_clear_counts_pipeline, prepare_density_pipeline, prepare_forces_pipeline,
    prepare_histogram_pipeline, prepare_integrate_pipeline, prepare_pressure_pipeline,
    prepare_scatter_pipeline, prepare_write_sentinel_pipeline,
};
use glam::{IVec2, Vec2};

// ==================== resources ======================================

/* interface of resources for a shader -> actual resource binding via BindGroup
and is created via RenderDevice::create_bind_group_layout. */
#[derive(Resource, Clone)]
pub struct ParticleBindGroupLayout(pub BindGroupLayout);

// responsible for render resources --> accessible in the pipeline
#[derive(Resource, Clone, ExtractResource)]
pub struct ParticleBindGroup(pub BindGroup);

#[derive(Resource)]
pub struct ParticleBuffers {
    pub particle_buffer: Buffer,
    pub num_particles: u32,
}

// Rendering world Copy
#[derive(Resource, Clone, ExtractResource)]
pub struct ExtractedParticleBuffer {
    pub buffer: Buffer,
    pub num_particles: u32,
}

// need information back on CPU
#[derive(Resource)]
pub struct ReadbackBuffer {
    pub buffer: Buffer,
    pub size_bytes: u64,
}

#[derive(Resource, Clone, ExtractResource)]
pub struct ExtractedReadbackBuffer {
    pub buffer: Buffer,
    pub size_bytes: u64,
}

#[derive(Resource, Clone, Copy, Default)]
pub struct AllowCopy(pub bool);

#[derive(Resource, Clone, ExtractResource, Default)]
pub struct ExtractedAllowCopy(pub bool);

#[derive(Resource)]
pub struct GridBuffers {
    pub params_buf: Buffer,  // UNIFORM
    pub starts_buf: Buffer,  // STORAGE
    pub entries_buf: Buffer, // STORAGE
    pub num_cells: usize,
    pub num_particles: usize,
}

#[derive(Resource, Clone)]
pub struct ExtractedGrid {
    pub params_buf: Buffer,
    pub starts_buf: Buffer,
    pub entries_buf: Buffer,
    pub num_cells: usize,
}

#[derive(Resource)]
pub struct IntegrateParamsBuffer {
    pub buffer: Buffer,
}

#[derive(Resource, Clone, ExtractResource)]
pub struct ExtractedIntegrateParamsBuffer {
    pub buffer: Buffer,
}
#[derive(Resource, Default, Clone, Copy)]
pub struct UseGpuIntegration(pub bool);

#[derive(Resource, Default)]
pub struct SimStep(pub u64);

#[derive(Resource, Clone, Copy, Debug)]
pub struct IntegrateConfig {
    pub dt: f32,
    pub x_min: f32,
    pub x_max: f32,
    pub bounce: f32,
}

impl Default for IntegrateConfig {
    fn default() -> Self {
        Self {
            dt: 0.0005,
            x_min: -5.0,
            x_max: 3.0,
            bounce: -3.0,
        }
    }
}

// =====================================================================

// ========================== systems ==================================

// Startup systems that have to run only once

fn init_gpu_buffers(mut commands: Commands, render_device: Res<RenderDevice>, sph: Res<SPHState>) {
    let particle_buffers = ParticleBuffers::new(&render_device, &sph);
    commands.insert_resource(particle_buffers);
}

fn init_particle_bind_group_layout(mut commands: Commands, render_device: Res<RenderDevice>) {
    let layout = render_device.create_bind_group_layout(
        Some("particle_bind_group_layout"),
        &[
            // binding 0: particles (read_write)
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
            // binding 1: cell_starts (read-only)
            BindGroupLayoutEntry {
                binding: 1,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            // binding 2: cell_entries (read-only)
            BindGroupLayoutEntry {
                binding: 2,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            // binding 3: grid params (uniform)
            BindGroupLayoutEntry {
                binding: 3,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            // binding 4: integrate params (uniform)
            BindGroupLayoutEntry {
                binding: 4,
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
    commands.insert_resource(ParticleBindGroupLayout(layout));
}

fn init_readback_buffer(
    mut commands: Commands,
    render_device: Res<RenderDevice>,
    particle_buffers: Option<Res<ParticleBuffers>>, // solving the panic problem
) {
    let Some(particle_buffers) = particle_buffers else {
        return;
    };
    let size_bytes =
        (particle_buffers.num_particles as u64) * (std::mem::size_of::<GPUParticle>() as u64);
    let buffer = render_device.create_buffer(&BufferDescriptor {
        label: Some("readback_buffer"),
        size: size_bytes,
        usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    commands.insert_resource(ReadbackBuffer { buffer, size_bytes });
}

fn init_allow_copy(mut commands: Commands) {
    commands.insert_resource(AllowCopy(true));
}

pub fn init_grid_buffers(
    mut commands: Commands,
    render_device: Res<RenderDevice>,
    sph: Res<SPHState>,
) {
    commands.insert_resource(GridBuffers::new(&render_device, &sph));
}

fn init_integrate_params_buffer(
    mut commands: Commands,
    render_device: Res<RenderDevice>,
    config: Res<IntegrateConfig>,
) {
    let params = IntegrateParams {
        dt: config.dt,
        x_min: config.x_min,
        x_max: config.x_max,
        bounce: config.bounce,
    };
    let buffer = render_device.create_buffer_with_data(&BufferInitDescriptor {
        label: Some("integrate_params_uniform"),
        contents: bytemuck::bytes_of(&params),
        usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
    });
    commands.insert_resource(IntegrateParamsBuffer { buffer });
}

fn init_use_gpu_integration(mut commands: Commands) {
    commands.insert_resource(UseGpuIntegration(true)); // had to become true for gpu demo to work
}

// Update systems that have to run per frame

fn queue_particle_buffer(
    sph: Res<SPHState>,
    particle_buffers: Option<Res<ParticleBuffers>>, // so that cpu example still works
    render_queue: Res<RenderQueue>,
    use_gpu_integration: Res<UseGpuIntegration>,
) {
    let Some(particle_buffers) = particle_buffers else {
        return;
    };
    if use_gpu_integration.0 {
        return;
    }
    let mut gpu_particles = Vec::with_capacity(sph.particles.len());
    for particle in &sph.particles {
        gpu_particles.push(GPUParticle {
            pos: [particle.pos.x, particle.pos.y],
            vel: [particle.vel.x, particle.vel.y],
            acc: [particle.acc.x, particle.acc.y],
            rho: particle.rho,
            p: particle.p,
        });
    }

    // writing the slice into the whole buffer
    render_queue.write_buffer(
        &particle_buffers.particle_buffer,
        0,
        bytemuck::cast_slice(&gpu_particles),
    );
}

pub fn update_grid_buffers(
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
    sph: Res<SPHState>,
    mut grid: ResMut<GridBuffers>,
) {
    grid.update(&render_device, &render_queue, &sph);
}

fn update_integrate_params_buffer(
    render_queue: Res<RenderQueue>,
    ub: Res<IntegrateParamsBuffer>,
    config: Res<IntegrateConfig>,
) {
    let params = IntegrateParams {
        dt: config.dt,
        x_min: config.x_min,
        x_max: config.x_max,
        bounce: config.bounce,
    };
    render_queue.write_buffer(&ub.buffer, 0, bytemuck::bytes_of(&params));
}

// Extract systems that send from App to Render

fn extract_particle_buffer(
    mut commands: Commands,
    particle_buffers: Extract<Res<ParticleBuffers>>,
) {
    commands.insert_resource(ExtractedParticleBuffer {
        buffer: particle_buffers.particle_buffer.clone(),
        num_particles: particle_buffers.num_particles,
    });
}

fn extract_bind_group_layout(
    mut commands: Commands,
    layout: Extract<Res<ParticleBindGroupLayout>>,
) {
    commands.insert_resource(ParticleBindGroupLayout(layout.0.clone()));
}

// Extract systems that in Render

fn prepare_particle_bind_group(
    mut commands: Commands,
    render_device: Res<RenderDevice>,
    layout: Res<ParticleBindGroupLayout>,
    extracted: Res<ExtractedParticleBuffer>,
    grid: Res<ExtractedGrid>,
    integ: Res<ExtractedIntegrateParamsBuffer>,
) {
    let bind_group = render_device.create_bind_group(
        Some("particle_bind_group"),
        &layout.0,
        &[
            BindGroupEntry {
                binding: 0,
                resource: extracted.buffer.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 1,
                resource: grid.starts_buf.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 2,
                resource: grid.entries_buf.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 3,
                resource: grid.params_buf.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 4,
                resource: integ.buffer.as_entire_binding(),
            },
        ],
    );
    commands.insert_resource(ParticleBindGroup(bind_group));
    info!("particle_bind_group is READY");
}

fn extract_readback_buffer(mut commands: Commands, readback: Extract<Res<ReadbackBuffer>>) {
    commands.insert_resource(ExtractedReadbackBuffer {
        buffer: readback.buffer.clone(),
        size_bytes: readback.size_bytes,
    });
}

fn extract_allow_copy(mut commands: Commands, allow: Extract<Res<AllowCopy>>) {
    commands.insert_resource(ExtractedAllowCopy(allow.0));
}

fn cell_ix(pos: Vec2, h: f32) -> IVec2 {
    (pos / h).floor().as_ivec2()
}

fn build_compressed_grid(sph: &SPHState) -> (GridParams, Vec<u32>, Vec<u32>) {
    let h = sph.h;

    let mut min_c = IVec2::new(i32::MAX, i32::MAX);
    let mut max_c = IVec2::new(i32::MIN, i32::MIN);
    for p in &sph.particles {
        let c = cell_ix(p.pos, h);
        min_c = IVec2::new(min_c.x.min(c.x), min_c.y.min(c.y));
        max_c = IVec2::new(max_c.x.max(c.x), max_c.y.max(c.y));
    }
    let dims = IVec2::new(max_c.x - min_c.x + 1, max_c.y - min_c.y + 1);
    let nx = dims.x.max(1) as usize;
    let ny = dims.y.max(1) as usize;
    let num_cells = nx * ny;
    let n = sph.particles.len();

    let mut counts = vec![0u32; num_cells];
    for (_i, p) in sph.particles.iter().enumerate() {
        let c = cell_ix(p.pos, h);
        let ix = (c.x - min_c.x) as usize;
        let iy = (c.y - min_c.y) as usize;
        let id = ix + iy * nx;
        debug_assert!(id < num_cells);
        counts[id] += 1;
    }

    let mut starts = vec![0u32; num_cells + 1];
    for i in 0..num_cells {
        starts[i + 1] = starts[i] + counts[i];
    }

    let mut offsets = starts.clone();
    let mut entries = vec![0u32; n];
    for (pi, p) in sph.particles.iter().enumerate() {
        let c = cell_ix(p.pos, h);
        let ix = (c.x - min_c.x) as usize;
        let iy = (c.y - min_c.y) as usize;
        let id = ix + iy * nx;
        let dst = &mut offsets[id];
        let idx = *dst as usize;
        entries[idx] = pi as u32;
        *dst += 1;
    }

    let params = GridParams {
        min_world: [min_c.x as f32 * h, min_c.y as f32 * h],
        cell_size: h,
        _pad0: 0.0,
        dims: [nx as u32, ny as u32],
        _pad1: [0, 0],
    };

    (params, starts, entries)
}

impl GridBuffers {
    pub fn new(render_device: &RenderDevice, sph: &SPHState) -> Self {
        let (params, starts, entries) = build_compressed_grid(sph);

        let params_buf = render_device.create_buffer_with_data(&BufferInitDescriptor {
            label: Some("Grid Params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
        });

        let starts_buf = render_device.create_buffer_with_data(&BufferInitDescriptor {
            label: Some("Grid Starts"),
            contents: bytemuck::cast_slice(&starts),
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
        });

        let entries_buf = render_device.create_buffer_with_data(&BufferInitDescriptor {
            label: Some("Grid entries"),
            contents: bytemuck::cast_slice(&entries),
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
        });

        info!(
            "Grid Init: cells={} ({}x{}), starts.len={}, entries.len={}",
            (params.dims[0] as usize) * (params.dims[1] as usize),
            params.dims[0],
            params.dims[1],
            starts.len(),
            entries.len()
        );

        Self {
            params_buf,
            starts_buf,
            entries_buf,
            num_cells: starts.len() - 1,
            num_particles: entries.len(),
        }
    }

    pub fn update(&mut self, render_device: &RenderDevice, queue: &RenderQueue, sph: &SPHState) {
        let (params, starts, entries) = build_compressed_grid(sph);

        let new_num_cells = starts.len() - 1;
        let new_num_particles = entries.len();

        if new_num_cells != self.num_cells {
            self.starts_buf = render_device.create_buffer_with_data(&BufferInitDescriptor {
                label: Some("Grid Starts"),
                contents: bytemuck::cast_slice(&starts),
                usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
            });
            self.num_cells = new_num_cells;
        } else {
            queue.write_buffer(&self.starts_buf, 0, bytemuck::cast_slice(&starts));
        }

        if new_num_particles != self.num_particles {
            self.entries_buf = render_device.create_buffer_with_data(&BufferInitDescriptor {
                label: Some("Grid Entries"),
                contents: bytemuck::cast_slice(&entries),
                usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
            });
            self.num_particles = new_num_particles;
        } else {
            queue.write_buffer(&self.entries_buf, 0, bytemuck::cast_slice(&entries));
        }

        queue.write_buffer(&self.params_buf, 0, bytemuck::bytes_of(&params));

        let nx = params.dims[0];
        let ny = params.dims[1];

        info!(
            "grid update: dims=({}x{}), cells={}, entries={}, starts[0..5]={:?}",
            nx,
            ny,
            self.num_cells,
            self.num_particles,
            &starts[0..starts.len().min(5)]
        );
    }
}

pub fn extract_grid_buffers(mut commands: Commands, grid: Extract<Res<GridBuffers>>) {
    commands.insert_resource(ExtractedGrid {
        params_buf: grid.params_buf.clone(),
        starts_buf: grid.starts_buf.clone(),
        entries_buf: grid.entries_buf.clone(),
        num_cells: grid.num_cells,
    });
}

// extract to render-world
fn extract_integrate_params_buffer(
    mut commands: Commands,
    ub: Extract<Res<IntegrateParamsBuffer>>,
) {
    commands.insert_resource(ExtractedIntegrateParamsBuffer {
        buffer: ub.buffer.clone(),
    });
}

// comparison between GPU results and CPU
pub fn readback_and_compare(
    render_device: Res<RenderDevice>,
    readback: Res<ReadbackBuffer>,
    sph: Res<SPHState>,
    mut allow_copy: ResMut<AllowCopy>,
    mut done: Local<bool>,
    mut frames_seen: Local<u32>,
    mut state: Local<u8>,
    step: Res<SimStep>,
) {
    const EPS: f32 = 1e-6;
    const MAX_REL: f32 = 0.01; // 1 % for rho, p, a
    const MAX_ABS_ACC: f32 = 0.50;
    const FRAMES_BEFORE_RD: u32 = 60; // give the sim time to warm up

    #[inline(always)]
    fn rel_err(a: f32, b: f32) -> f32 {
        ((b - a) / a.abs().max(EPS)).abs()
    }

    if *done {
        return;
    }

    *frames_seen += 1;
    info!("frame {}, sim step {}", *frames_seen, step.0);

    if *frames_seen < FRAMES_BEFORE_RD {
        return;
    }

    match *state {
        0 => {
            allow_copy.0 = false; // skip copy next render frame
            *state = 1;
            return;
        }

        1 => {
            let slice = readback.buffer.slice(..);

            // async map
            let status = Arc::new(AtomicU8::new(0)); // 0=pending 1=ok 2=err
            let cb = status.clone();
            slice.map_async(MapMode::Read, move |r| {
                cb.store(if r.is_ok() { 1 } else { 2 }, Ordering::SeqCst);
            });

            // spin-wait: RenderSchedule runs on the main thread anyway
            loop {
                render_device.poll(Maintain::Poll);
                match status.load(Ordering::SeqCst) {
                    0 => std::thread::yield_now(),
                    1 => break,
                    2 => {
                        error!("GPU buffer map failed");
                        readback.buffer.unmap();
                        *done = true;
                        *state = 2;
                        return;
                    }
                    _ => unreachable!(),
                }
            }

            // comparison in one pass
            let data = slice.get_mapped_range();
            let gpu: &[GPUParticle] = bytemuck::cast_slice(&data);

            let mut max_rel_rho: f32 = 0.0;
            let mut max_rel_p: f32 = 0.0;
            let mut max_rel_a: f32 = 0.0;
            let mut max_abs_a: f32 = 0.0;

            for (cpu, g) in sph.particles.iter().zip(gpu) {
                max_rel_rho = max_rel_rho.max(rel_err(cpu.rho, g.rho));
                max_rel_p = max_rel_p.max(rel_err(cpu.p, g.p));

                let cpu_a = glam::Vec2::new(cpu.acc.x, cpu.acc.y);
                let gpu_a = glam::Vec2::new(g.acc[0], g.acc[1]);
                let diff = (gpu_a - cpu_a).length();
                max_abs_a = max_abs_a.max(diff);
                max_rel_a = max_rel_a.max(diff / cpu_a.length().max(EPS));
            }

            // helper macro so we don’t repeat boilerplate
            macro_rules! check {
                ($label:literal, $err:expr, $lim:expr) => {
                    if $err > $lim {
                        error!(
                            "FAIL: {} error {:.3} % > {:.1} %",
                            $label,
                            $err * 100.0,
                            $lim * 100.0
                        );
                        return Err(());
                    } else {
                        info!(
                            "PASS: {} within {:.1} % (max {:.3} %)",
                            $label,
                            $lim * 100.0,
                            $err * 100.0
                        );
                    }
                };
            }

            let res: Result<(), ()> = (|| {
                check!("density", max_rel_rho, MAX_REL);
                check!("pressure", max_rel_p, MAX_REL);
                if max_rel_a > MAX_REL || max_abs_a > MAX_ABS_ACC {
                    error!(
                        "FAIL: accel rel {:.3} %, abs {:.3} (limits {:.1} %, {:.2})",
                        max_rel_a * 100.0,
                        max_abs_a,
                        MAX_REL * 100.0,
                        MAX_ABS_ACC
                    );
                    return Err(());
                } else {
                    info!(
                        "PASS: accel within limits (rel {:.3} %, abs {:.3})",
                        max_rel_a * 100.0,
                        max_abs_a
                    );
                }
                Ok(())
            })();

            drop(data);
            readback.buffer.unmap();
            *done = true;
            *state = 2;

            if res.is_err() {
                panic!("GPU <-> CPU validation failed; see log above");
            }
        }

        _ => {}
    }
}

// Implementations

impl ParticleBuffers {
    pub fn new(render_device: &RenderDevice, sph: &SPHState) -> Self {
        // converting the cpu particle to gpu
        let mut gpu_particles = Vec::with_capacity(sph.particles.len());
        for particle in &sph.particles {
            gpu_particles.push(GPUParticle {
                pos: [particle.pos.x, particle.pos.y],
                vel: [particle.vel.x, particle.vel.y],
                acc: [particle.acc.x, particle.acc.y],
                rho: particle.rho,
                p: particle.p,
            });
        }

        // storage buffer with the init data
        let particle_buffer = render_device.create_buffer_with_data(&BufferInitDescriptor {
            label: Some("Particle Buffer"),
            contents: bytemuck::cast_slice(&gpu_particles),
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
        });

        Self {
            particle_buffer,
            num_particles: gpu_particles.len() as u32,
        }
    }
}

// =====================================================================

// Plugin

pub struct GPUSPHPlugin;

impl Plugin for GPUSPHPlugin {
    fn build(&self, app: &mut App) {
        // ================== App world ==================
        app.init_resource::<IntegrateConfig>();
        app.add_systems(
            Startup,
            (
                init_gpu_buffers,
                init_readback_buffer,
                init_particle_bind_group_layout,
                init_allow_copy,
                init_grid_buffers,
                init_integrate_params_buffer,
                init_use_gpu_integration,
            )
                .chain(),
        )
        .add_systems(
            Update,
            (
                queue_particle_buffer,
                update_grid_buffers,
                update_integrate_params_buffer,
            ),
        );

        // ================== Render world ==================
        let render_app = app.sub_app_mut(RenderApp);

        // ---- Extract (App -> Render) ----
        render_app.add_systems(
            ExtractSchedule,
            (
                extract_particle_buffer,
                extract_bind_group_layout,
                extract_readback_buffer,
                extract_allow_copy,
                extract_grid_buffers,
                extract_integrate_params_buffer,
            ),
        );

        // ---- Prepare (pipelines, bind groups) ----
        render_app.add_systems(
            Render,
            (
                // SPH compute
                prepare_particle_bind_group,
                prepare_density_pipeline,
                prepare_pressure_pipeline,
                prepare_forces_pipeline,
                prepare_integrate_pipeline,
                // Grid build: counts & params
                init_grid_build_bind_group_layout,
                init_grid_build_buffers.after(init_grid_build_bind_group_layout),
                prepare_clear_counts_pipeline.after(init_grid_build_bind_group_layout),
                // Histogram
                init_grid_histogram_bind_group_layout,
                init_grid_histogram_bind_group
                    .after(init_grid_histogram_bind_group_layout)
                    .after(init_grid_build_buffers)
                    .after(prepare_particle_bind_group),
                prepare_histogram_pipeline.after(init_grid_histogram_bind_group_layout),
            )
                .in_set(RenderSet::Prepare),
        );

        // Render — block B (starts + block scan)
        render_app.add_systems(
            Render,
            (
                init_counts_to_starts_bgl,
                init_starts_buffer_and_bg
                    .after(init_counts_to_starts_bgl)
                    .after(init_grid_build_buffers),
                // prepare_prefix_sum_naive_pipeline ... (kept disabled)
                init_block_scan_bgl,
                init_block_sums_and_bg
                    .after(init_block_scan_bgl)
                    .after(init_starts_buffer_and_bg),
                prepare_block_scan_pipeline.after(init_block_scan_bgl),
            )
                .in_set(RenderSet::Prepare),
        );

        // Render — block C (block_sums scan + add-back + sentinel)
        render_app.add_systems(
            Render,
            (
                init_block_sums_scan_bgl,
                init_block_sums_scan_bg
                    .after(init_block_sums_scan_bgl)
                    .after(init_block_sums_and_bg),
                prepare_block_sums_scan_pipeline.after(init_block_sums_scan_bgl),
                init_add_back_bgl,
                init_add_back_bg
                    .after(init_add_back_bgl)
                    .after(init_block_sums_and_bg)
                    .after(init_starts_buffer_and_bg),
                prepare_add_back_pipeline.after(init_add_back_bgl),
                // Sentinel pipeline (after add_back)
                prepare_write_sentinel_pipeline.after(prepare_add_back_pipeline),
            )
                .in_set(RenderSet::Prepare),
        );

        // Render — block D (cursor + scatter)
        render_app.add_systems(
            Render,
            (
                init_cursor_buffer_and_clear_bg.after(prepare_add_back_pipeline),
                init_gpu_entries_buffer.after(init_grid_build_buffers),
                init_scatter_bgl,
                init_scatter_bg
                    .after(init_scatter_bgl)
                    .after(init_cursor_buffer_and_clear_bg)
                    .after(init_starts_buffer_and_bg)
                    .after(init_gpu_entries_buffer),
                prepare_scatter_pipeline.after(init_scatter_bgl),
            )
                .in_set(RenderSet::Prepare),
        );

        // ---- Render Graph nodes (order via edges) ----
        add_density_node_to_graph(render_app);
        add_clear_counts_node_to_graph(render_app);
        add_histogram_node_to_graph(render_app);
        // add_prefix_sum_naive_node_to_graph(render_app);
        add_block_scan_node_to_graph(render_app);
        add_block_sums_scan_node_to_graph(render_app);
        add_add_back_node_to_graph(render_app);
        add_write_sentinel_node_to_graph(render_app);
        add_clear_cursor_node_to_graph(render_app);
        add_scatter_node_to_graph(render_app);
    }
}
