//TODO make it into a test

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

const MAX_REL_RHO: f32 = 0.01;
const MAX_ABS_P: f32 = 30.0;

#[inline(always)]
fn rel_err(a: f32, b: f32) -> f32 {
    const EPS: f32 = 1e-6;
    ((b - a) / a.abs().max(EPS)).abs()
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
        .add_systems(Update, readback.before(update_grid_buffers))
        .run();
}

fn readback(
    mut allow_copy: ResMut<AllowCopy>,
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
            if *cpu_steps < 10 {
                sph.step(DT, X_MAX, X_MIN, BOUNCE);
                *cpu_steps += 1;
                if *cpu_steps == 10 {
                    *state = 1;
                }
            }
        }
        1 => {
            if *frame >= 11 {
                allow_copy.0 = true;
                *state = 2;
            }
        }
        2 => {
            if *frame >= 12 {
                allow_copy.0 = false;
                *state = 3;
            }
        }
        3 => {
            if *frame >= 13 {
                render_device.poll(Maintain::Wait);
                let slice = readback.buffer.slice(..);

                let status = std::sync::Arc::new(std::sync::atomic::AtomicU8::new(0));
                let cb = status.clone();
                slice.map_async(MapMode::Read, move |r| {
                    cb.store(
                        if r.is_ok() { 1 } else { 2 },
                        std::sync::atomic::Ordering::SeqCst,
                    )
                });

                // wait for map
                loop {
                    render_device.poll(Maintain::Poll);
                    match status.load(std::sync::atomic::Ordering::SeqCst) {
                        0 => std::thread::yield_now(),
                        1 => break,
                        2 => {
                            readback.buffer.unmap();
                            exit.write(AppExit::Success);
                            return;
                        }
                        _ => unreachable!(),
                    }
                }

                // compare
                {
                    let data = slice.get_mapped_range();
                    let gpu: &[GPUParticle] = bytemuck::cast_slice(&data);
                    assert_eq!(
                        gpu.len(),
                        sph.particles.len(),
                        "GPU/CPU particle counts differ"
                    );

                    let mut max_rel_rho: f32 = 0.0;
                    let mut max_abs_p: f32 = 0.0;

                    let mut max_rel_p_all: f32 = 0.0;
                    let mut max_rel_p_filtered: f32 = 0.0;
                    let mut n_filtered: u32 = 0;

                    const P_FLOOR: f32 = 30.0;

                    for (i, cpu_p) in sph.particles.iter().enumerate() {
                        let g = &gpu[i];

                        // density (relative)
                        max_rel_rho = max_rel_rho.max(rel_err(cpu_p.rho, g.rho));

                        // pressure (absolute)
                        let dp = (g.p - cpu_p.p).abs();
                        max_abs_p = max_abs_p.max(dp);

                        let relp = rel_err(cpu_p.p, g.p);
                        max_rel_p_all = max_rel_p_all.max(relp);
                        if cpu_p.p.abs() > P_FLOOR {
                            max_rel_p_filtered = max_rel_p_filtered.max(relp);
                            n_filtered += 1;
                        }
                    }

                    info!(
                        "10-step parity (GPU vs CPU):  rho max_rel = {:.3}%  |  p max_abs = {:.3}  |  p max_rel_all = {:.3}%  |  p max_rel(|p|>{}) = {:.3}% (n={})",
                        max_rel_rho * 100.0,
                        max_abs_p,
                        max_rel_p_all * 100.0,
                        P_FLOOR,
                        max_rel_p_filtered * 100.0,
                        n_filtered
                    );

                    assert!(
                        max_rel_rho <= MAX_REL_RHO,
                        "FAIL: density max_rel {:.4} > {:.4}",
                        max_rel_rho,
                        MAX_REL_RHO
                    );
                    assert!(
                        max_abs_p <= MAX_ABS_P,
                        "FAIL: pressure max_abs {:.3} > {:.3}",
                        max_abs_p,
                        MAX_ABS_P
                    );
                }

                readback.buffer.unmap();
                exit.write(AppExit::Success);
            }
        }
        _ => {}
    }
}
