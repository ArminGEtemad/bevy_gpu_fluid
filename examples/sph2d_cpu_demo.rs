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


fn sph_step(mut sph: ResMut<SPHState>, time: Res<Time>) {
    let dt = time.delta_secs().min(DT);
    sph.step(dt); // integral
}

fn sync_particles(
    sph: Res<SPHState>,
    mut query: Query<(&ParticleVisual, &mut Transform)>,
) {
    for (visual, mut transform) in query.iter_mut() {
        let pos = sph.particles[visual.0].pos;
        transform.translation.x = pos.x * RENDER_SCALE;
        transform.translation.y = pos.y * RENDER_SCALE;
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