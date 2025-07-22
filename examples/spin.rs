use bevy::prelude::*;

#[derive(Component)]
struct Rotates {
    axis: Vec3,
    speed: f32, // radians per seconds
    mode: RotationMode,
}
#[derive(Debug, Copy, Clone)]
enum RotationMode {
    SpinInPlace,
    OrbitAround, // assumes center = Vec3::ZERO for now
}

fn main() {
    App::new()
        .add_plugins(DefaultPlugins)
        .add_systems(Startup, setup)
        .add_systems(Update, spin)
        .run();

}

// setup a sample 3d Scene
fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    // circular base
    commands.spawn((
        Mesh3d(meshes.add(Circle::new(4.0))),
        MeshMaterial3d(materials.add(Color::WHITE)),
        Transform::from_rotation(Quat::from_rotation_x(-std::f32::consts::FRAC_PI_2)),
    ));

    // cube
    commands.spawn((
        Mesh3d(meshes.add(Cuboid::new(1.0, 1.0, 1.0))),
        MeshMaterial3d(materials.add(Color::srgb_u8(124, 144, 255))),
        Transform::from_xyz(0.0, 0.5, 0.0),
        Rotates {
            axis: Vec3::Y,
            speed: 1.0,
            mode: RotationMode::SpinInPlace,
        },
    ));

    // light
    commands.spawn((
        PointLight {
            shadows_enabled: true,
            ..default()
        },
        Transform::from_xyz(1.0, 3.0, 1.0),
        Rotates {
            axis: Vec3::X,
            speed: 2.0,
            mode: RotationMode::OrbitAround,
        },
    ));

    // camera
    commands.spawn((
        Camera3d::default(),
        Transform::from_xyz(-2.5, 4.5, 9.0).looking_at(Vec3::ZERO, Vec3::Y)
    ));
}

fn spin(mut query: Query<(&mut Transform, &Rotates)>, time: Res<Time>) {
    let dt = time.delta_secs();
    for (mut transform, rotate) in &mut query {
        match rotate.mode {
            RotationMode::SpinInPlace => {
                transform.rotate(Quat::from_axis_angle(rotate.axis, rotate.speed * dt));
            }

            RotationMode::OrbitAround => {
                let pos = transform.translation;
                let rotation = Quat::from_axis_angle(rotate.axis, rotate.speed *dt);
                transform.translation = rotation * pos;
                transform.look_at(Vec3::ZERO, Vec3::X);
            }
        }
    }
}