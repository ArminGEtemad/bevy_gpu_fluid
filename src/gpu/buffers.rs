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
use crate::gpu::ffi::GPUParticle;
use crate::gpu::pipeline::{
    add_density_node_to_graph, prepare_density_pipeline, prepare_forces_pipeline,
    prepare_integrate_pipeline, prepare_pressure_pipeline,
};

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
    commands.insert_resource(ParticleBindGroupLayout(layout));
}

// Update systems that have to run per frame

fn queue_particle_buffer(
    sph: Res<SPHState>,
    particle_buffers: Option<Res<ParticleBuffers>>, // so that cpu example still works
    render_queue: Res<RenderQueue>,
) {
    let Some(particle_buffers) = particle_buffers else {
        return;
    };
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
) {
    let bind_group = render_device.create_bind_group(
        Some("particle_bind_group"),
        &layout.0,
        &[BindGroupEntry {
            binding: 0,
            resource: extracted.buffer.as_entire_binding(),
        }],
    );
    commands.insert_resource(ParticleBindGroup(bind_group));

    info!("particle_bind_group is READY")
}

fn extract_readback_buffer(mut commands: Commands, readback: Extract<Res<ReadbackBuffer>>) {
    commands.insert_resource(ExtractedReadbackBuffer {
        buffer: readback.buffer.clone(),
        size_bytes: readback.size_bytes,
    });
}

fn init_allow_copy(mut commands: Commands) {
    commands.insert_resource(AllowCopy(true));
}

fn extract_allow_copy(mut commands: Commands, allow: Extract<Res<AllowCopy>>) {
    commands.insert_resource(ExtractedAllowCopy(allow.0));
}

// comparison between GPU results and CPU
pub fn readback_and_compare(
    render_device: Res<RenderDevice>,
    readback: Res<ReadbackBuffer>,
    sph: Res<SPHState>,
    mut allow_copy: ResMut<AllowCopy>,
    mut done: Local<bool>,
    mut frames_seen: Local<u32>,
) {
    const FRAMES_BEFORE_READBACK: u32 = 60; // just to get a fast response

    if *done {
        return;
    }

    *frames_seen += 1; // increment
    info!("from readback: frames_seen= {}", *frames_seen);

    if *frames_seen < FRAMES_BEFORE_READBACK {
        return;
    }

    allow_copy.0 = false;

    let slice = readback.buffer.slice(..);
    let status = Arc::new(AtomicU8::new(0)); // on gpu (another thread) -> atomicity is crucial
    let status_cb = Arc::clone(&status);

    // status flag to know when the operation is done
    slice.map_async(MapMode::Read, move |res| {
        status_cb.store(if res.is_ok() { 1 } else { 2 }, Ordering::SeqCst);
    });

    render_device.poll(Maintain::Wait);
    match status.load(Ordering::SeqCst) {
        1 => {}      // mapped OK
        2 => return, // failed
        _ => return, // not ready
    }

    let data = slice.get_mapped_range();
    let gpu_particles: &[GPUParticle] = bytemuck::cast_slice(&data);

    // ----------------------------- debug only ---------------------

    info!(
        "GPU rho head: [{:.0}, {:.0}, {:.0}, {:.0}, {:.0}]",
        gpu_particles[0].rho,
        gpu_particles[1].rho,
        gpu_particles[2].rho,
        gpu_particles[3].rho,
        gpu_particles[4].rho,
    );

    let mut max_rel: f32 = 0.0;
    for (i, cpu_p) in sph.particles.iter().enumerate() {
        let a = cpu_p.rho;
        let b = gpu_particles[i].rho;
        let denom = a.abs().max(1e-6);
        let rel = ((b - a) / denom).abs();
        if rel > max_rel {
            max_rel = rel;
        }
    }
    info!("GPU density max relatice error: {:.3}%", max_rel * 100.0);

    info!(
        "GPU p head: [{:.0}, {:.0}, {:.0}, {:.0}, {:.0}]",
        gpu_particles[0].p,
        gpu_particles[1].p,
        gpu_particles[2].p,
        gpu_particles[3].p,
        gpu_particles[4].p,
    );

    let mut max_rel_p: f32 = 0.0;
    for (i, cpu_p) in sph.particles.iter().enumerate() {
        let a = cpu_p.p;
        let b = gpu_particles[i].p;
        let denom = a.abs().max(1e-6);
        let rel = ((b - a) / denom).abs();
        if rel > max_rel_p {
            max_rel_p = rel;
        }
    }
    info!("GPU pressure max relatice error: {:.3}%", max_rel_p * 100.0);

    // Print both CPU and GPU acc for a quick eyeball check
    info!(
        "CPU acc head: [{:.3}, {:.3}] [{:.3}, {:.3}] [{:.3}, {:.3}] [{:.3}, {:.3}] [{:.3}, {:.3}]",
        sph.particles[0].acc.x,
        sph.particles[0].acc.y,
        sph.particles[1].acc.x,
        sph.particles[1].acc.y,
        sph.particles[2].acc.x,
        sph.particles[2].acc.y,
        sph.particles[3].acc.x,
        sph.particles[3].acc.y,
        sph.particles[4].acc.x,
        sph.particles[4].acc.y,
    );

    info!(
        "GPU acc head: [{:.3}, {:.3}] [{:.3}, {:.3}] [{:.3}, {:.3}] [{:.3}, {:.3}] [{:.3}, {:.3}]",
        gpu_particles[0].acc[0],
        gpu_particles[0].acc[1],
        gpu_particles[1].acc[0],
        gpu_particles[1].acc[1],
        gpu_particles[2].acc[0],
        gpu_particles[2].acc[1],
        gpu_particles[3].acc[0],
        gpu_particles[3].acc[1],
        gpu_particles[4].acc[0],
        gpu_particles[4].acc[1],
    );

    // Vector relative error with an epsilon to avoid divide-by-near-zero
    let mut max_rel_acc = 0.0f32;
    let mut max_abs_acc = 0.0f32;
    for (i, cpu_p) in sph.particles.iter().enumerate() {
        let cpu = glam::Vec2::new(cpu_p.acc.x, cpu_p.acc.y);
        let gpu = glam::Vec2::new(gpu_particles[i].acc[0], gpu_particles[i].acc[1]);

        let diff = (gpu - cpu).length();
        let denom = cpu.length().max(1e-3); // epsilon to avoid huge % when cpu≈0
        let rel = diff / denom;

        max_rel_acc = max_rel_acc.max(rel);
        max_abs_acc = max_abs_acc.max(diff);
    }
    info!(
        "GPU acceleration max relative error: {:.3}%",
        max_rel_acc * 100.0
    );
    info!("GPU acceleration max absolute error: {:.3}", max_abs_acc);

    info!(
        "GPU pos head: [{:.3},{:.3}] [{:.3},{:.3}] [{:.3},{:.3}] [{:.3},{:.3}] [{:.3},{:.3}]",
        gpu_particles[0].pos[0],
        gpu_particles[0].pos[1],
        gpu_particles[1].pos[0],
        gpu_particles[1].pos[1],
        gpu_particles[2].pos[0],
        gpu_particles[2].pos[1],
        gpu_particles[3].pos[0],
        gpu_particles[3].pos[1],
        gpu_particles[4].pos[0],
        gpu_particles[4].pos[1],
    );
    info!(
        "GPU vel head: [{:.3},{:.3}] [{:.3},{:.3}] [{:.3},{:.3}] [{:.3},{:.3}] [{:.3},{:.3}]",
        gpu_particles[0].vel[0],
        gpu_particles[0].vel[1],
        gpu_particles[1].vel[0],
        gpu_particles[1].vel[1],
        gpu_particles[2].vel[0],
        gpu_particles[2].vel[1],
        gpu_particles[3].vel[0],
        gpu_particles[3].vel[1],
        gpu_particles[4].vel[0],
        gpu_particles[4].vel[1],
    );

    // vector relative errors with epsilon
    let mut max_rel_pos = 0.0f32;
    let mut max_rel_vel = 0.0f32;

    for (i, cpu_p) in sph.particles.iter().enumerate() {
        let cpu_pos = glam::Vec2::new(cpu_p.pos.x, cpu_p.pos.y);
        let cpu_vel = glam::Vec2::new(cpu_p.vel.x, cpu_p.vel.y);
        let gpu_pos = glam::Vec2::new(gpu_particles[i].pos[0], gpu_particles[i].pos[1]);
        let gpu_vel = glam::Vec2::new(gpu_particles[i].vel[0], gpu_particles[i].vel[1]);

        let pos_diff = (gpu_pos - cpu_pos).length();
        let vel_diff = (gpu_vel - cpu_vel).length();
        let pos_rel = pos_diff / cpu_pos.length().max(1e-6);
        let vel_rel = vel_diff / cpu_vel.length().max(1e-6);

        max_rel_pos = max_rel_pos.max(pos_rel);
        max_rel_vel = max_rel_vel.max(vel_rel);
    }

    info!(
        "GPU position max relative error: {:.3}%",
        max_rel_pos * 100.0
    );
    info!(
        "GPU velocity max relative error: {:.3}%",
        max_rel_vel * 100.0
    );

    // ----------------------------------------------------------------

    // marking done
    drop(data);
    readback.buffer.unmap();
    *done = true;
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
        // App
        app.add_systems(
            Startup,
            (
                init_gpu_buffers,
                init_readback_buffer,
                init_particle_bind_group_layout,
                init_allow_copy,
            )
                .chain(),
        )
        //.add_systems(Startup, init_particle_bind_group_layout)
        //.add_systems(Startup, init_allow_copy)
        .add_systems(Update, queue_particle_buffer);

        // Render
        let render_app = app.sub_app_mut(RenderApp);
        render_app
            .add_systems(
                ExtractSchedule,
                (
                    extract_particle_buffer,
                    extract_bind_group_layout,
                    extract_readback_buffer,
                    extract_allow_copy,
                ),
            )
            .add_systems(
                Render,
                (
                    prepare_particle_bind_group.in_set(RenderSet::Prepare),
                    prepare_density_pipeline.in_set(RenderSet::Prepare),
                    prepare_pressure_pipeline.in_set(RenderSet::Prepare),
                    prepare_forces_pipeline.in_set(RenderSet::Prepare),
                    prepare_integrate_pipeline.in_set(RenderSet::Prepare),
                ),
            );

        add_density_node_to_graph(render_app);
    }
}

/* used https://cocalc.com/github/bevyengine/bevy/blob/main/examples/shader/compute_shader_game_of_life.rs
as my example here */
