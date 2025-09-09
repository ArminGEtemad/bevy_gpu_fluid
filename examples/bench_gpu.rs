use bevy::diagnostic::{DiagnosticsStore, FrameTimeDiagnosticsPlugin};
use bevy::prelude::*;
use bevy_gpu_fluid::cpu::sph2d::SPHState;
use bevy_gpu_fluid::gpu::buffers::{GPUSPHPlugin, UseGpuIntegration};

const DURATION_SEC: f32 = 3.0;

#[derive(Resource)]
struct BenchPlan(Vec<usize>); // particle counts to run

#[derive(Resource)]
struct BenchTimer {
    t: f32,
    accum_fps: f64,
    frames: u32,
}

#[derive(Resource, Default)]
struct SeedRequest(bool); // one-frame CPU->GPU seed after switching cases

fn make_state(count: usize) -> SPHState {
    let mut s = SPHState::new(0.045, 1000.0, 3.0, 0.2, 1.6);
    let n = (count as f32).sqrt() as usize;
    s.init_grid(n, n, 0.04);
    s
}

// plain helper (NOT a system)
fn setup_case_now(commands: &mut Commands, n: usize) {
    info!("--- Starting bench: {} particles ---", n);
    commands.insert_resource(make_state(n));
    commands.spawn(Camera2d::default());
}

fn main() {
    let plan_vec = vec![10_000, 5_041, 1_024];
    let first_n = plan_vec[0];

    App::new()
        .add_plugins(DefaultPlugins)
        .add_plugins(FrameTimeDiagnosticsPlugin::default())
        .insert_resource(BenchPlan(plan_vec))
        .insert_resource(BenchTimer {
            t: 0.0,
            accum_fps: 0.0,
            frames: 0,
        })
        .insert_resource(UseGpuIntegration(false))
        .insert_resource(SeedRequest::default())
        .insert_resource(make_state(first_n))
        .add_plugins(GPUSPHPlugin)
        .add_systems(Startup, setup_first_case)
        .add_systems(Update, (gpu_seed_fsm, accum_fps, step_timer, rotate_cases))
        .run();
}

fn setup_first_case(mut commands: Commands, plan: Res<BenchPlan>) {
    if plan.0.first().is_some() {
        commands.spawn(Camera2d::default());
    }
}

fn gpu_seed_fsm(
    mut use_gpu: ResMut<UseGpuIntegration>,
    mut req: ResMut<SeedRequest>,
    mut phase: Local<u8>,
) {
    if !req.0 {
        return;
    }
    match *phase {
        0 => {
            use_gpu.0 = false;
            *phase = 1;
        } // this frame: upload
        1 => {
            use_gpu.0 = true;
            *phase = 0;
            req.0 = false;
        } // next frame: resume GPU
        _ => {
            *phase = 0;
            req.0 = false;
        }
    }
}

fn accum_fps(diags: Res<DiagnosticsStore>, mut bench: ResMut<BenchTimer>) {
    if let Some(fps) = diags.get(&FrameTimeDiagnosticsPlugin::FPS) {
        if let Some(s) = fps.smoothed() {
            bench.accum_fps += s as f64;
            bench.frames += 1;
        }
    }
}

fn step_timer(time: Res<Time>, mut bench: ResMut<BenchTimer>) {
    bench.t += time.delta_secs();
}

fn rotate_cases(
    mut commands: Commands,
    mut plan: ResMut<BenchPlan>,
    mut bench: ResMut<BenchTimer>,
    mut req: ResMut<SeedRequest>,
    q_cam: Query<Entity, With<Camera>>,
) {
    if bench.t < DURATION_SEC {
        return;
    }

    let avg_fps = if bench.frames > 0 {
        bench.accum_fps / bench.frames as f64
    } else {
        0.0
    };
    info!(
        "Result: avg FPS over ~{:.1}s = {:.1}",
        DURATION_SEC, avg_fps
    );

    // reset timers
    bench.t = 0.0;
    bench.accum_fps = 0.0;
    bench.frames = 0;

    // advance plan
    if !plan.0.is_empty() {
        plan.0.remove(0);
    }

    // clear previous camera
    for e in q_cam.iter() {
        commands.entity(e).despawn();
    }

    if let Some(&n) = plan.0.first() {
        // replace SPHState & request one-frame seed
        commands.insert_resource(make_state(n));
        req.0 = true;
        setup_case_now(&mut commands, n);
    } else {
        info!("Bench complete.");
        std::process::exit(0);
    }
}
