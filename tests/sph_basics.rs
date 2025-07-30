use bevy_gpu_fluid::cpu::sph2d::SPHState;

#[test]
fn init_grid_n() {
    let mut sph = SPHState::new(0.1);
    sph.init_grid(10, 5, 0.12);
    assert_eq!(sph.particles.len(), 50); // 10 * 5
    assert_eq!(sph.particles[0].pos, glam::Vec2::new(0.0, 0.0)); // 0 * 0.12, 0 * 0.12
    assert_eq!(sph.particles[1].pos, glam::Vec2::new(0.12, 0.0)); // 1 * 0.12, 0 * 0.12
    assert_eq!(sph.particles[10].pos, glam::Vec2::new(0.0, 0.12)); // 0 * 0.12, 1 * 0.12
}
