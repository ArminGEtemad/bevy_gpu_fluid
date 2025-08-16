use bevy::prelude::*;
use bevy::render::extract_resource::ExtractResource;
use bevy::render::render_resource::{
    BindGroup, BindGroupEntry, BindGroupLayout, BindGroupLayoutEntry, BindingType, Buffer,
    BufferBindingType, BufferInitDescriptor, BufferUsages, ShaderStages,
};
use bevy::render::renderer::{RenderDevice, RenderQueue};
use bevy::render::{Extract, ExtractSchedule, Render, RenderApp, RenderSet};

use crate::cpu::sph2d::SPHState;
use crate::gpu::ffi::GPUParticle;
use crate::gpu::pipeline::{add_density_node_to_graph, prepare_density_pipeline};

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
        app.add_systems(Startup, (init_gpu_buffers, init_particle_bind_group_layout))
            .add_systems(Update, queue_particle_buffer);

        // Render
        let render_app = app.sub_app_mut(RenderApp);
        render_app
            .add_systems(
                ExtractSchedule,
                (extract_particle_buffer, extract_bind_group_layout),
            )
            .add_systems(
                Render,
                (
                    prepare_particle_bind_group.in_set(RenderSet::Prepare),
                    prepare_density_pipeline.in_set(RenderSet::Prepare),
                ),
            );

        add_density_node_to_graph(render_app);
    }
}

/* used https://cocalc.com/github/bevyengine/bevy/blob/main/examples/shader/compute_shader_game_of_life.rs
as my example here */
