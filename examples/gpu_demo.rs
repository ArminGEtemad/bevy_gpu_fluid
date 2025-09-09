use bevy::diagnostic::{DiagnosticsStore, FrameTimeDiagnosticsPlugin};
use bevy::prelude::*;
use bevy::render::render_resource::{Maintain, MapMode};
use bevy_gpu_fluid::cpu::sph2d::SPHState;
use bevy_gpu_fluid::gpu::buffers::update_grid_buffers;
use bevy_gpu_fluid::gpu::buffers::{AllowCopy, GPUSPHPlugin, ReadbackBuffer, UseGpuIntegration};
use bevy_gpu_fluid::gpu::ffi::GPUParticle;

const RENDER_SCALE: f32 = 100.0;
const PARTICLE_SIZE: f32 = 15.0;
const CYAN: Color = Color::srgb(0.0, 1.0, 1.0);

#[derive(Component)]
struct ParticleVisual(usize);

fn main() {
    App::new()
        .add_plugins((DefaultPlugins, FrameTimeDiagnosticsPlugin::default()))
        .insert_resource(ClearColor(Color::Srgba(
            bevy::color::palettes::css::DARK_SLATE_GRAY,
        )))
        // 5k particles config
        .insert_resource(SPHState::demo_block_5k())
        // RUN the GPU integration
        .insert_resource(UseGpuIntegration(true))
        .add_plugins(GPUSPHPlugin)
        .add_systems(Startup, setup)
        .add_systems(Update, sync_sprites_from_gpu.before(update_grid_buffers))
        .add_systems(Update, log_fps)
        .run();
}

fn setup(mut commands: Commands, sph: Res<SPHState>) {
    commands.spawn(Camera2d::default());

    // one sprite per particle
    for (i, p) in sph.particles.iter().enumerate() {
        commands.spawn((
            Sprite {
                color: CYAN,
                custom_size: Some(Vec2::splat(PARTICLE_SIZE)),
                ..Default::default()
            },
            Transform::from_translation(Vec3::new(
                p.pos.x * RENDER_SCALE,
                p.pos.y * RENDER_SCALE,
                0.0,
            )),
            GlobalTransform::default(),
            ParticleVisual(i),
        ));
    }
}

// Read GPU buffer every other frame:
//   even frames:  allow copy GPU→readback
//   odd frames:   map+read CPU, update sprite transforms, unmap
fn sync_sprites_from_gpu(
    mut allow_copy: ResMut<AllowCopy>,
    readback: Option<Res<ReadbackBuffer>>,
    mut q: Query<(&ParticleVisual, &mut Transform)>,
    render_device: Res<bevy::render::renderer::RenderDevice>,
    mut sph: ResMut<SPHState>,
    mut fsm: Local<u8>, // 0 copy, 1 disable, 2 wait, 3 map, 4 cool-down
) {
    let Some(readback) = readback else { return };

    match *fsm {
        0 => {
            allow_copy.0 = true;
            *fsm = 1;
            return;
        }
        1 => {
            allow_copy.0 = false;
            *fsm = 2;
            return;
        }
        2 => {
            *fsm = 3;
            return;
        }

        3 => {
            let slice = readback.buffer.slice(..);
            render_device.poll(Maintain::Wait);

            let status = std::sync::Arc::new(std::sync::atomic::AtomicU8::new(0));
            let cb = status.clone();
            slice.map_async(MapMode::Read, move |r| {
                cb.store(
                    if r.is_ok() { 1 } else { 2 },
                    std::sync::atomic::Ordering::SeqCst,
                )
            });

            loop {
                render_device.poll(Maintain::Poll);
                match status.load(std::sync::atomic::Ordering::SeqCst) {
                    0 => std::thread::yield_now(),
                    1 => break,
                    2 => {
                        readback.buffer.unmap();
                        *fsm = 0;
                        return;
                    }
                    _ => unreachable!(),
                }
            }

            {
                let data = slice.get_mapped_range();
                let gpu: &[GPUParticle] = bytemuck::cast_slice(&data);

                // (2) Mirror GPU -> CPU state AND update sprites
                for (i, p_gpu) in gpu.iter().enumerate() {
                    // update CPU state so grid rebuild uses current positions
                    let p_cpu = &mut sph.particles[i];
                    p_cpu.pos.x = p_gpu.pos[0];
                    p_cpu.pos.y = p_gpu.pos[1];
                    p_cpu.vel.x = p_gpu.vel[0];
                    p_cpu.vel.y = p_gpu.vel[1];
                    p_cpu.acc.x = p_gpu.acc[0];
                    p_cpu.acc.y = p_gpu.acc[1];
                    p_cpu.rho = p_gpu.rho;
                    p_cpu.p = p_gpu.p;
                }

                // update transforms (separate loop to avoid borrow clash)
                for (vis, mut tf) in q.iter_mut() {
                    let p = &gpu[vis.0];
                    tf.translation.x = p.pos[0] * RENDER_SCALE;
                    tf.translation.y = p.pos[1] * RENDER_SCALE;
                }
            }
            readback.buffer.unmap();

            *fsm = 4;
            return;
        }

        4 => {
            allow_copy.0 = false;
            *fsm = 0;
            return;
        }
        _ => *fsm = 0,
    }
}

fn log_fps(diagnostics: Res<DiagnosticsStore>, mut counter: Local<u32>) {
    *counter += 1;
    if *counter >= 120 {
        *counter = 0;

        if let Some(fps_diag) = diagnostics.get(&FrameTimeDiagnosticsPlugin::FPS) {
            if let Some(avg) = fps_diag.average() {
                info!("==== Average FPS over last ~2 s: {:.1} ====", avg); // grabbing the FPS 
            }
        }
    }
}
