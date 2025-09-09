use bevy::app::AppExit;
use bevy::prelude::*;
use bevy::render::render_resource::{Maintain, MapMode};
use bevy_gpu_fluid::gpu::buffers::update_grid_buffers;
use bevy_gpu_fluid::{
    cpu::sph2d::SPHState,
    gpu::buffers::{AllowCopy, GPUSPHPlugin, ReadbackBuffer, UseGpuIntegration},
    gpu::ffi::GPUParticle,
};

const DT: f32 = 0.0005;
const X_MIN: f32 = -5.0;
const X_MAX: f32 = 3.0;
const BOUNCE: f32 = -3.0;

const STEPS: u32 = 10; // <— compare after this many steps

#[inline(always)]
fn rel_norm_sym(a: glam::Vec2, b: glam::Vec2) -> f32 {
    let diff = (b - a).length();
    let scale = a.length().max(b.length()).max(1e-6);
    diff / scale
}

fn main() {
    App::new()
        .add_plugins(DefaultPlugins)
        .insert_resource(SPHState::demo_block_5k())
        .insert_resource(UseGpuIntegration(false))
        .insert_resource(AllowCopy(false))
        .add_plugins(GPUSPHPlugin)
        .add_systems(Startup, |mut commands: Commands| {
            commands.spawn(Camera2d::default());
        })
        .add_systems(Update, orchestrate_100.before(update_grid_buffers))
        .run();
}

fn orchestrate_100(
    mut allow_copy: ResMut<AllowCopy>,
    mut use_gpu: ResMut<UseGpuIntegration>,
    mut sph: ResMut<SPHState>,
    readback: Option<Res<ReadbackBuffer>>,
    render_device: Res<bevy::render::renderer::RenderDevice>,
    mut exit: EventWriter<AppExit>,
    mut frame: Local<u32>,
    mut state: Local<u8>,
    mut cpu_steps: Local<u32>,
) {
    let Some(readback) = readback else { return };
    *frame += 1;

    match *state {
        0 => {
            if *frame == 1 {
                info!("frame 1: seeding GPU from CPU (streaming)");
            }
            if *frame == 2 {
                use_gpu.0 = true; // GPU advance itself from now on
                info!("frame 2: GPU integration ENABLED");
            }

            // CPU advances exactly once per frame until STEPS reached
            if *cpu_steps < STEPS {
                sph.step(DT, X_MAX, X_MIN, BOUNCE);
                *cpu_steps += 1;
                if *cpu_steps == STEPS {
                    info!("Reached {} CPU steps; preparing readback.", STEPS);
                    *state = 1;
                }
            }
        }

        // copy this frame
        1 => {
            allow_copy.0 = true;
            *state = 2;
        }

        // avoid mapping race
        2 => {
            allow_copy.0 = false;
            *state = 3;
        }

        3 => {
            render_device.poll(Maintain::Wait);
            let slice = readback.buffer.slice(..);

            // async map
            let status = std::sync::Arc::new(std::sync::atomic::AtomicU8::new(0));
            let cb = status.clone();
            slice.map_async(MapMode::Read, move |r| {
                cb.store(
                    if r.is_ok() { 1 } else { 2 },
                    std::sync::atomic::Ordering::SeqCst,
                )
            });

            // wait for map completion
            loop {
                render_device.poll(Maintain::Poll);
                match status.load(std::sync::atomic::Ordering::SeqCst) {
                    0 => std::thread::yield_now(),
                    1 => break,
                    2 => {
                        error!("map_async failed; unmapping and exiting");
                        readback.buffer.unmap();
                        exit.write(AppExit::Success);
                        return;
                    }
                    _ => unreachable!(),
                }
            }

            // compute diffs
            {
                let data = slice.get_mapped_range();
                let gpu: &[GPUParticle] = bytemuck::cast_slice(&data);
                assert_eq!(
                    gpu.len(),
                    sph.particles.len(),
                    "GPU/CPU particle counts differ"
                );

                let mut max_rel_x = 0.0f32;
                let mut max_rel_v = 0.0f32;
                let mut max_abs_x = 0.0f32;
                let mut max_abs_v = 0.0f32;

                const TOP_N: usize = 3;
                let mut top: Vec<(usize, f32)> = Vec::with_capacity(TOP_N);

                for (i, cpu_p) in sph.particles.iter().enumerate() {
                    let cx = glam::Vec2::new(cpu_p.pos.x, cpu_p.pos.y);
                    let cv = glam::Vec2::new(cpu_p.vel.x, cpu_p.vel.y);
                    let gx = glam::Vec2::new(gpu[i].pos[0], gpu[i].pos[1]);
                    let gv = glam::Vec2::new(gpu[i].vel[0], gpu[i].vel[1]);

                    let abs_x = (gx - cx).length();
                    let abs_v = (gv - cv).length();
                    let rel_x = rel_norm_sym(cx, gx);
                    let rel_v = rel_norm_sym(cv, gv);

                    max_abs_x = max_abs_x.max(abs_x);
                    max_abs_v = max_abs_v.max(abs_v);
                    max_rel_x = max_rel_x.max(rel_x);
                    max_rel_v = max_rel_v.max(rel_v);

                    if top.len() < TOP_N {
                        top.push((i, abs_x));
                        top.sort_by(|a, b| b.1.total_cmp(&a.1));
                    } else if abs_x > top[TOP_N - 1].1 {
                        top[TOP_N - 1] = (i, abs_x);
                        top.sort_by(|a, b| b.1.total_cmp(&a.1));
                    }
                }

                info!(
                    "10-step parity:  max_rel |x| = {:.3}% |v| = {:.3}%   max_abs |x| = {:.6} |v| = {:.6}",
                    max_rel_x * 100.0,
                    max_rel_v * 100.0,
                    max_abs_x,
                    max_abs_v
                );

                if !top.is_empty() {
                    info!("Top {} particles by |x| abs diff:", top.len());
                    for (rank, (i, abs_x)) in top.iter().enumerate() {
                        let i = *i;
                        let cx = glam::Vec2::new(sph.particles[i].pos.x, sph.particles[i].pos.y);
                        let gx = glam::Vec2::new(gpu[i].pos[0], gpu[i].pos[1]);
                        let cv = glam::Vec2::new(sph.particles[i].vel.x, sph.particles[i].vel.y);
                        let gv = glam::Vec2::new(gpu[i].vel[0], gpu[i].vel[1]);
                        info!(
                            "#{:>2} idx {}: |Delta_x|={:.6}  CPU x=({:.6},{:.6})  GPU x=({:.6},{:.6})  |Delta_v|={:.6}",
                            rank + 1,
                            i,
                            abs_x,
                            cx.x,
                            cx.y,
                            gx.x,
                            gx.y,
                            (gv - cv).length()
                        );
                    }
                }
            }

            readback.buffer.unmap();
            info!("Done. Exiting.");
            exit.write(AppExit::Success);
        }

        _ => {}
    }
}
