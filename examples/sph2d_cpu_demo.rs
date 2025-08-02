use bevy::prelude::*;
use bevy::sprite::Sprite;
use bevy_gpu_fluid::cpu::sph2d::SPHState;

#[derive(Component)]
struct ParticleVisual(usize);

const RENDER_SCALE: f32 = 100.0;
const PARTICLE_SIZE: f32 = 15.0;
const DT: f32 = 0.0005;

fn main() {
    App::new()
        .add_plugins(DefaultPlugins)
        .insert_resource(SPHState::demo_block_5k())
        .add_systems(Startup, setup)
        .add_systems(Update, (sph_step, sync_particles))
        .run();
}

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


fn sph_step(mut sph: ResMut<SPHState>, time: Res<Time>) {
    let dt = time.delta_secs().min(DT);
    sph.step(dt); // integral
}

fn sync_particles(
    sph: Res<SPHState>,
    mut query: Query<(&ParticleVisual, &mut Transform, &mut Sprite)>,
) {
    let (mut min_rho, mut max_rho) = (f32::MAX, f32::MIN);
    for p in &sph.particles {
        min_rho = min_rho.min(p.rho);
        max_rho = max_rho.max(p.rho);
    }
    let inv_range = if max_rho > min_rho { 1.0 / (max_rho - min_rho) } else { 0.0 };

    for (visual, mut transform, mut sprite) in query.iter_mut() {
        let particle = &sph.particles[visual.0];
        transform.translation.x = particle.pos.x * RENDER_SCALE;
        transform.translation.y = particle.pos.y * RENDER_SCALE;

        let t = ((particle.rho - min_rho) * inv_range).clamp(0.0, 1.0);
        sprite.color = density_color(t);
    }
}

fn setup(
    mut commands: Commands,
    sph: Res<SPHState>,
) {
    commands.spawn(Camera2d::default());

    for (i, p) in sph.particles.iter().enumerate() {
        commands.spawn((
            Sprite {
                color: Color::srgb(0.0, 1.0, 1.0),
                custom_size: Some(Vec2::splat(PARTICLE_SIZE)),
                ..Default::default()
            },
            Transform::from_translation(Vec3::new(p.pos.x * RENDER_SCALE,
                                                  p.pos.y * RENDER_SCALE,
                                                  0.0)),
            GlobalTransform::default(),
            ParticleVisual(i),
        ));
    }
}