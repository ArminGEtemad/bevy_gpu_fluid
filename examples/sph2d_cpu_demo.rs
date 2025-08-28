use bevy::input::ButtonInput;
use bevy::prelude::*;
use bevy::sprite::Sprite;
use bevy::window::PrimaryWindow;
use glam::Vec2 as GVec2;

use bevy_gpu_fluid::cpu::sph2d::SPHState;
use bevy_gpu_fluid::gpu::buffers::{SimStep, readback_and_compare};

const RENDER_SCALE: f32 = 100.0;
const PARTICLE_SIZE: f32 = 15.0;
const DT: f32 = 0.0005;
const X_MAX: f32 = 3.0;
const X_MIN: f32 = -5.0;
const BOUNCINESS: f32 = -3.0;
const INTERACTION_AREA: f32 = 0.04; // when using mouse to interact
const IMPULSE: f32 = 10.0; // when using mouse to interact
const CYAN: Color = Color::srgb(0.0, 1.0, 1.0);

#[derive(Component)]
struct ParticleVisual(usize);

#[derive(Resource, Default)]
struct DragInput {
    screen_pos: Vec2,
    delta: Vec2,        // movement from a frame ago
    pressed_down: bool, // left mouse button must be held
}

#[derive(Resource)]
enum ViewMode {
    ConstColor,
    DensityColor,
}

fn main() {
    App::new()
        .add_plugins(DefaultPlugins)
        .add_plugins(bevy_gpu_fluid::gpu::buffers::GPUSPHPlugin)
        .insert_resource(SPHState::demo_block_5k())
        .insert_resource(DragInput::default())
        .insert_resource(ViewMode::DensityColor)
        .insert_resource(SimStep::default())
        .add_systems(Startup, setup)
        .add_systems(
            Update,
            (
                drag_input,
                sph_step,
                apply_drag,
                toggle_view,
                sync_particles,
                // readback_and_compare,
            ),
        )
        .run();
}

// toggle between the view modes
fn toggle_view(keys: Res<ButtonInput<KeyCode>>, mut view: ResMut<ViewMode>) {
    if keys.just_pressed(KeyCode::Space) {
        *view = match *view {
            ViewMode::ConstColor => ViewMode::DensityColor,
            ViewMode::DensityColor => ViewMode::ConstColor,
        }
    }
}

// from blue to red based on the density
fn density_color(t: f32) -> Color {
    let t = t.clamp(0.0, 1.0);
    if t < 0.5 {
        let u = t * 2.0;
        Color::srgb(0.0, u, 1.0)
    } else if t < 0.75 {
        let u = (t - 0.5) / 0.25;
        Color::srgb(u, 1.0, 1.0 - u)
    } else {
        let u = (t - 0.75) / 0.25;
        Color::srgb(1.0, 1.0 - u, 0.0)
    }
}

// by pressing left mouse button, fluid is "touched"
fn drag_input(
    mut drag: ResMut<DragInput>,
    buttons: Res<ButtonInput<MouseButton>>,
    windows: Query<&Window, With<PrimaryWindow>>,
) {
    let window = match windows.single() {
        Ok(w) => w,
        Err(_) => return,
    };

    if let Some(pos) = window.cursor_position() {
        let pos = Vec2::new(pos.x, pos.y);
        if buttons.pressed(MouseButton::Left) {
            drag.delta = if drag.pressed_down {
                pos - drag.screen_pos
            } else {
                Vec2::ZERO
            };
            drag.screen_pos = pos;
            drag.pressed_down = true;
        } else {
            drag.pressed_down = false;
            drag.delta = Vec2::ZERO;
        }
    }
}

fn apply_drag(
    mut sph: ResMut<SPHState>,
    drag: Res<DragInput>,
    windows: Query<&Window, With<PrimaryWindow>>,
) {
    if !drag.pressed_down || drag.delta.length_squared() == 0.0 {
        return;
    }

    let window = match windows.single() {
        Ok(w) => w,
        Err(_) => return,
    };

    let win_w = window.resolution.width();
    let win_h = window.resolution.height();

    let cursor_world = GVec2::new(
        (drag.screen_pos.x - win_w * 0.5) / RENDER_SCALE,
        (-drag.screen_pos.y + win_h * 0.5) / RENDER_SCALE,
    );
    let force_dir = GVec2::new(drag.delta.x / RENDER_SCALE, -drag.delta.y / RENDER_SCALE);

    for p in &mut sph.particles {
        let to_particle = p.pos - cursor_world;
        if to_particle.length_squared() < INTERACTION_AREA {
            p.vel += IMPULSE * force_dir;
        }
    }
}

// all the mathematic happens here!
fn sph_step(mut sph: ResMut<SPHState>, time: Res<Time>, mut step: ResMut<SimStep>) {
    let dt = time.delta_secs().min(DT);
    sph.step(dt, X_MAX, X_MIN, BOUNCINESS); // integral
    step.0 += 1;
}

fn sync_particles(
    sph: Res<SPHState>,
    view: Res<ViewMode>,
    mut query: Query<(&ParticleVisual, &mut Transform, &mut Sprite)>,
) {
    let (mut min_rho, mut max_rho) = (f32::MAX, f32::MIN);
    for p in &sph.particles {
        // find min and max density
        min_rho = min_rho.min(p.rho);
        max_rho = max_rho.max(p.rho);
    }
    let inv_range = if max_rho > min_rho {
        1.0 / (max_rho - min_rho)
    } else {
        0.0
    };

    for (visual, mut transform, mut sprite) in query.iter_mut() {
        let particle = &sph.particles[visual.0];

        // position must be matched with the Bevy world
        transform.translation.x = particle.pos.x * RENDER_SCALE;
        transform.translation.y = particle.pos.y * RENDER_SCALE;
        match *view {
            ViewMode::ConstColor => {
                sprite.color = CYAN;
            }
            ViewMode::DensityColor => {
                let t = ((particle.rho - min_rho) * inv_range).clamp(0.0, 1.0);
                sprite.color = density_color(t);
            }
        }
    }
}

// spawn a camera and particles
fn setup(mut commands: Commands, sph: Res<SPHState>) {
    commands.spawn(Camera2d::default());

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
