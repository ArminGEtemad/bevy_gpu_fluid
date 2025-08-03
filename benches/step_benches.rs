use bevy_gpu_fluid::cpu::sph2d::*;
use criterion::{criterion_group, criterion_main, Criterion};

fn bench_step(c: &mut Criterion) {
    let h = 0.045;
    let spacing = 0.08; // spacing < h for overlap
    let rho_0 = 1000.0;
    let k = 3.0;
    let mu = 0.1;
    let m = rho_0  * spacing * spacing;

    let mut sph = SPHState::new(h, rho_0, k, mu, m);

    sph.init_grid(70, 70, spacing);

    c.bench_function("step_4.9k", |b| b.iter(|| sph.step(0.001)));
}

criterion_group!(benches, bench_step);
criterion_main!(benches);