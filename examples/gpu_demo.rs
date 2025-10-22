use bevy::diagnostic::{DiagnosticsStore, FrameTimeDiagnosticsPlugin};
use bevy::prelude::*;
use bevy::render::RenderPlugin;
use bevy::render::settings::{RenderCreation, WgpuFeatures, WgpuSettings};
use bevy_gpu_fluid::cpu::sph2d::SPHState;
use bevy_gpu_fluid::gpu::buffers::{GPUSPHPlugin, UseGpuIntegration};

fn main() {
    App::new()
        .add_plugins(DefaultPlugins.set(RenderPlugin {
            render_creation: RenderCreation::Automatic(WgpuSettings {
                features: WgpuFeatures::VERTEX_WRITABLE_STORAGE
                    | WgpuFeatures::TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES,
                ..Default::default()
            }),
            synchronous_pipeline_compilation: true,
            ..Default::default()
        }))
        .add_plugins(FrameTimeDiagnosticsPlugin::default()) // <- not DefaultPlugins again
        .insert_resource(ClearColor(Color::Srgba(
            bevy::color::palettes::css::DARK_SLATE_GRAY,
        )))
        .insert_resource(SPHState::demo_block_5k())
        .insert_resource(UseGpuIntegration(true))
        .add_plugins(GPUSPHPlugin)
        .add_systems(Startup, setup)
        .add_systems(Update, log_fps)
        .run();
}

fn setup(mut commands: Commands, _sph: Res<SPHState>) {
    commands.spawn(Camera2d::default());
}

fn log_fps(diagnostics: Res<DiagnosticsStore>, mut counter: Local<u32>) {
    *counter += 1;
    if *counter >= 120 {
        *counter = 0;
        if let Some(fps) = diagnostics
            .get(&FrameTimeDiagnosticsPlugin::FPS)
            .and_then(|d| d.average())
        {
            info!("==== Average FPS over last ~2 s: {:.1} ====", fps);
        }
    }
}
