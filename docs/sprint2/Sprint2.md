# Goals

The goals in sprint 2 was to have a working 2D fluid simulation that runs
smoothly on CPU. The math behind the simulation is SPH or smoothed-particle 
hydrodynamics. 

The following steps were taken

- AoS particle store
- Density calculation
- Pressure, vescosity and gravity acceleration
- Integration using Euler methid
- defining boundary conditions
- No-NaN tests
- Benchmarking for about 5k particles
- Bevy 2D visual demos
- adding density color mapping
- interctive manipulation of the fluid using mouse left button

# AoS particle store
To arrange a sequence of records in the memory, I used AoS (Array of Structures)
instead of SoA (Structure of Arrays). The reason was that it was easier and faster for
prototyping. The idea was that to change to SoA if the programm wouldn't pass the benchmarking.

# Kernels
My source of information of this part all came from:
[Here](https://courses.grainger.illinois.edu/CS418/sp2023/text/sph.html)

[Here](https://physics.stackexchange.com/questions/138700/kernel-normalization-in-smoothed-particle-hydrodynamcs)

[Here](https://www.cs.cmu.edu/~scoros/cs15467-s16/lectures/11-fluids2.pdf)

[Here](https://cs418.cs.illinois.edu/website/text/sph.html)


## Density Calculation
To calculate the Density, I had to define **Poly6 Kernel**.

```rust
#[inline]
fn w_poly6(r2: f32, h: f32) -> f32 {
    let k: f32 = 4.0 / (PI * h.powi(8));
    if r2 >= 0.0 && r2 <= h * h {
        k * (h * h - r2).powi(3)
    } else { 0.0 }
}
```

## Pressure Calculation
Here, I used **Spiky Kernel**. In the calculation the gradiant was needed.

```rust
#[inline]
fn grad_spiky_kernel(r: Vec2, h: f32) -> Vec2 {
    let r_len = r.length();
    let k = -10.0 / (PI * h.powi(5));
    if r_len == 0.0 || r_len >= h {
        Vec2::ZERO
    } else {
        k * (h - r_len).powi(2) * r.normalize()
    }
}
```

## Viscosity Calculation
The laplacian of the **Viscosity Kernel** was needed:

```rust
#[inline]
fn laplacian_visc(r: f32, h: f32) -> f32 {
    let k: f32 = 40.0 / (PI * h.powi(5));
    if r == 0.0 || r >= h {
        0.0
    } else {
        k * (h - r)
    } 
}
```
## Summary

| Name & Range                       | $W(r, h)$  (2D)       | $\nabla W$  (2D)         |  $\Delta W$  (2D)         |
|--------------------------------------|-----------------------|--------------------------|--------------------------|
| **Poly6** $0\le r\le h$              | $\frac{4}{\pi h^8} (h^2 - r^2)^3$ | $\times$ | $\times$ |
| **Spiky** $0\le r\le h$              | $\times$ | $\frac{-10}{\pih^5} (h - r)^2 \hat{r}$ | $\times$ |
| **Viscosity (Müller)** $0\le r\le h$ | $\times$ | $\times$ | $\frac{40}{\pi h^5} (h - r)$ |

The difference between the 2D and the 3D version seems to only be the normalization factors. So the code would have still worked
if I would have used 3D normalization factors (just scaled differently).


# Integration

## Euler method
for the integration I used the simplest Euler method 

```rust
pub fn integrate(&mut self, dt: f32) {
        for p in &mut self.particles {
            p.vel += p.acc * dt;
            p.pos += p.vel * dt;
        }
    }

    pub fn apply_boundaries(&mut self, x_max: f32, x_min: f32, bounce: f32) {
        // bounciness must be a negative number
        for p in &mut self.particles {
            // floor
            if p.pos.y < 0.0 {
                p.pos.y = 0.0;
                p.vel.y *= bounce;
            }

            // right wall
            if p.pos.x > x_max {
                p.pos.x = x_max;
                p.vel.x *= bounce;
            }

            // left wall
            if p.pos.x < x_min {
                p.pos.x = x_min;
                p.vel.x *= bounce;
            }
        }
    }

    pub fn step(&mut self, dt: f32, x_max: f32, x_min: f32, bounce: f32) {
        self.density_pressure_calc();
        self.accel_field_calc();
        self.integrate(dt);
        self.apply_boundaries(x_max, x_min, bounce)
    }
```
I could have used other kind of boundaries as well. But the walls and the floor I made, satisfied me enough. New kind of boundaries 
could be 
1. oscillating walls
2. floor with a slope
3. circular container 
4. etc.

## No NaN Test
To make sure that the integral does not diverges I used a test:
```rust 
#[test]
fn integral_no_nan() {
    let h = 0.045;
    let spacing = 0.04;
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
```
## Benchmarking
To make sure that the calculation works smoothly and fast enough, I did a benchmarking using Criterion for 
about 5k particles.

```rust
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
```

I have an Interl core i7 (14th Generation). On my computer the result was that for about 5k particles every step needed 1.87ms.

# Visualization
I used Bevy (of course!) for the visualization. Added two visualization modes. And it is possible to interact with the fluid with left mouse button. The only thing that took me long to figure out was the sprite import `use bevy::sprite::Sprite;`. It took me along time to figure that out. 

and with that Sprint is finished. 