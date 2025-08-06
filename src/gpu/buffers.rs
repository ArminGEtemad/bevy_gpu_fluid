use bevy::prelude::*;
use bevy::render::render_resource::{BufferUsages, Buffer, BufferInitDescriptor};
use bevy::render::renderer::RenderDevice;
// https://docs.rs/bevy/latest/bevy/render/renderer/struct.RenderDevice.html

use crate::gpu::ffi::GPUParticle;
use crate::cpu::sph2d::SPHState;


#[derive(Resource)]
pub struct ParticleBuffers {
    pub particle_buffer: Buffer,
    pub num_particles: u32,
}

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
        let particle_buffer = render_device.create_buffer_with_data(
            &BufferInitDescriptor{
                label: Some("Particle Buffer"),
                contents: bytemuck::cast_slice(&gpu_particles),
                usage: BufferUsages::STORAGE
                    | BufferUsages::COPY_DST
                    | BufferUsages::COPY_SRC,
            },
        );

        Self { particle_buffer, num_particles: gpu_particles.len() as u32 }
    }
}


fn init_gpu_buffers(
    mut commands: Commands,
    render_device: Res<RenderDevice>,
    sph: Res<SPHState>,
) {
    let particle_buffers = ParticleBuffers::new(&render_device, &sph);
    commands.insert_resource(particle_buffers);
}

pub struct GPUSPHPlugin;

impl Plugin for GPUSPHPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Startup, init_gpu_buffers);
    }
}


