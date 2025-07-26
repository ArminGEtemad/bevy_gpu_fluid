use bevy::prelude::*;
use bevy::input::mouse::MouseMotion;
use bevy::input::mouse::MouseWheel;

use bevy_gpu_fluid::{SceneControl, Rotates, ControlTarget, RotationMode};
use bevy_gpu_fluid::solid_color::SolidColor;



fn main() {
    App::new()
        .add_plugins(DefaultPlugins.set(AssetPlugin {
            watch_for_changes_override: Some(true),
            ..default()
        }))
        .add_plugins(MaterialPlugin::<SolidColor>::default())
        .add_systems(Startup, setup)
        .add_systems(Update, (spin, scene_control))
        .run();

}

// setup a sample 3d Scene
fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut solid_mats: ResMut<Assets<SolidColor>>,
) {
    commands.insert_resource(ControlTarget::Camera);

    // circular base
    commands.spawn((
        Mesh3d(meshes.add(Circle::new(4.0))),
        MeshMaterial3d(materials.add(Color::WHITE)),
        Transform::from_rotation(Quat::from_rotation_x(-std::f32::consts::FRAC_PI_2)),
    ));

    // cube
    let mat = solid_mats.add(SolidColor {
        color: LinearRgba { red: 0.2, green: 0.8, blue: 0.9, alpha: 1.0 }
    });
    commands.spawn((
        Mesh3d(meshes.add(Cuboid::new(1.0, 1.0, 1.0))),
        MeshMaterial3d(mat),
        //MeshMaterial3d(materials.add(Color::srgb_u8(124, 144, 255))),
        Transform::from_xyz(0.0, 0.5, 0.0),
        Rotates {
            axis: Vec3::Y,
            speed: 0.0,
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
        SceneControl { target: ControlTarget::Light, speed: 2.0 },
        Rotates {
            axis: Vec3::X,
            speed: 0.0,
            mode: RotationMode::OrbitAround,
        },
    ));

    // camera
    commands.spawn((
        Camera3d::default(),
        Transform::from_xyz(-2.5, 4.5, 9.0).looking_at(Vec3::ZERO, Vec3::Y),
        SceneControl { target: ControlTarget::Camera, speed: 3.0 },
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
                transform.look_at(Vec3::ZERO, Vec3::Y);
            }
        }
    }
}

fn scene_control(
    time: Res<Time>,
    keys: Res<ButtonInput<KeyCode>>,
    mouse_button: Res<ButtonInput<MouseButton>>,
    mut evr_motion: EventReader<MouseMotion>,
    mut evr_scroll: EventReader<MouseWheel>,
    mut control_target: ResMut<ControlTarget>,
    mut query: Query<(&mut Transform, &SceneControl)>,
) {
    let dt = time.delta_secs();
    // toggle between camera and light
    if keys.just_pressed(KeyCode::Tab) {
        *control_target = match *control_target {
            ControlTarget::Camera => ControlTarget::Light,
            ControlTarget::Light => ControlTarget::Camera,
        };
        println!("Switched target to {:?}", control_target);
    }
    
    for (mut transform, control) in &mut query {
        
        if control.target != *control_target {
            continue;
        }

        let mut direction = Vec3::ZERO;
        let center = Vec3::ZERO;
        let forward = transform.forward();
        let right = transform.right();
        let speed_multiplier = if keys.pressed(KeyCode::ShiftLeft) || keys.pressed(KeyCode::ShiftRight) {
            2.0
        } else {
            1.0
        };
        
        // WASD movement
        if keys.pressed(KeyCode::KeyW) { direction += *forward; }
        if keys.pressed(KeyCode::KeyS) { direction -= *forward; }
        if keys.pressed(KeyCode::KeyA) { direction -= *right; }
        if keys.pressed(KeyCode::KeyD) { direction += *right; }

        if direction != Vec3::ZERO {
            if *control_target == ControlTarget::Camera {
                let displacement = direction.normalize() * control.speed * speed_multiplier * dt;
                transform.translation += displacement;
            } else if *control_target == ControlTarget::Light {
                let light_offset = transform.translation - center;

                let yaw = Quat::from_axis_angle(Vec3::Y, -direction.x * control.speed * dt);
                let pitch = Quat::from_axis_angle(*right, -direction.y * control.speed * dt);

                let light_rotated_offset = yaw * pitch * light_offset;
                transform.translation = center + light_rotated_offset;

                transform.look_at(center, Vec3::Y);
            }
        }

        // mouse movement
        if mouse_button.pressed(MouseButton::Middle) && control.target == ControlTarget::Camera{
            for ev in evr_motion.read() {
                let delta = ev.delta;
                let mouse_sensitivity: f32 = 0.005;
                
                let yaw = Quat::from_axis_angle(Vec3::Y, -delta.x * mouse_sensitivity);
                let pitch = Quat::from_axis_angle(*right, -delta.y * mouse_sensitivity);
                
                let camera_offset = transform.translation - center;
                let camera_rotated_offset = yaw * pitch * camera_offset;
                
                transform.translation = center + camera_rotated_offset;
                transform.look_at(center, Vec3::Y);

            }
        }
        // zoom function
        for ev in evr_scroll.read() {
            let scroll_amount = ev.y;
            let zoom_speed: f32 = 10.0;

            let camera_offset = transform.translation - center;
            let zoom_delta = camera_offset.normalize() * scroll_amount * zoom_speed * dt;

            transform.translation -= zoom_delta;

            transform.look_at(center, Vec3::Y);
        }
    }
}
