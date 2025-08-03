use bevy_gpu_fluid::cpu::sph2d::SPHState;

#[test]
fn init_grid_n() {
    let h = 0.045;
    let spacing = 0.12; // spacing < h for overlap
    let rho_0 = 1000.0;
    let k = 3.0;
    let mu = 0.1;
    let m = rho_0  * spacing * spacing;

    let mut sph = SPHState::new(h, rho_0, k, mu, m);
    sph.init_grid(10, 5, spacing);
    assert_eq!(sph.particles.len(), 50); // 10 * 5
    assert_eq!(sph.particles[0].pos, glam::Vec2::new(0.0, 0.0)); // 0 * 0.12, 0 * 0.12
    assert_eq!(sph.particles[1].pos, glam::Vec2::new(0.12, 0.0)); // 1 * 0.12, 0 * 0.12
    assert_eq!(sph.particles[10].pos, glam::Vec2::new(0.0, 0.12)); // 0 * 0.12, 1 * 0.12
}

#[test]
fn grid_contains_all_particles() {
    let h = 0.045;
    let spacing = 0.08; // spacing < h for overlap
    let rho_0 = 1000.0;
    let k = 3.0;
    let mu = 0.1;
    let m = rho_0  * spacing * spacing;

    let mut sph = SPHState::new(h, rho_0, k, mu, m);
    sph.init_grid(4, 3, spacing); // 4 * 3 = 12 particles
    let grid = sph.build_grid();

    let amount: usize = grid.values().map(Vec::len).sum();
    assert_eq!(amount, sph.particles.len()); // 12 for it to work
}

#[test]
fn uniform_density_compare_to_rho_0() {
    let h = 0.045;
    let spacing = 0.04; // spacing < h for overlap
    let rho_0 = 1000.0;
    let k = 3.0;
    let mu = 0.1;
    let m = rho_0 * spacing * spacing;

    let mut sph = SPHState::new(h, rho_0, k, mu, m);
    sph.init_grid(6, 6, spacing); 
    sph.density_pressure_calc();
    let max_rel_err = sph.particles
                        .iter()
                        .map(|p| ((p.rho - sph.rho_0) / sph.rho_0).abs())
                        .fold(0.0, f32::max); // maximum relative error
    assert!(max_rel_err < 0.05, "relative error for density > 5%");
}

#[test]
fn integral_no_nan() {
    let h = 0.045;
    let spacing = 0.04; // spacing < h for overlap
    let rho_0 = 1000.0;
    let k = 3.0;
    let mu = 0.1;
    let m = rho_0  * spacing * spacing;
    let x_max = 3.0;
    let x_min = -3.0;
    let bounce = 3.0;

    let mut sph = SPHState::new(h, rho_0, k, mu, m);
    sph.init_grid(10, 10, spacing);
    for _ in 0..50 { sph.step(0.001, x_max, x_min, bounce); }
    assert!(sph.particles.iter().all(|p| p.pos.is_finite()));

}