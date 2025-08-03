// smoothed particle hydrodynamics in 2D (CPU prototype)
use std::{collections::HashMap, f32::consts::PI};

use glam::{Vec2, IVec2};
use bevy::prelude::Resource;

type Cell = IVec2;

const GRAVITY: Vec2 = Vec2::new(0.0, -9.81);

#[inline]
fn cell(pos: Vec2, h: f32) -> IVec2 {
    (pos / h).floor().as_ivec2()
}

// define 2D Kernels

#[inline]
fn w_poly6(r2: f32, h: f32) -> f32 {
    let k: f32 = 4.0 / (PI * h.powi(8));
    if r2 >= 0.0 && r2 <= h * h {
        k * (h * h - r2).powi(3)
    } else { 0.0 }
}

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

#[inline]
fn laplacian_visc(r: f32, h: f32) -> f32 {
    let k: f32 = 40.0 / (PI * h.powi(5));
    if r == 0.0 || r >= h {
        0.0
    } else {
        k * (h - r)
    } 
}

#[derive(Clone, Debug)]
pub struct Particle {
    pub pos: Vec2, // position 
    pub vel: Vec2, // velocity
    pub acc: Vec2, // acceleration
    pub rho: f32, // density
    pub p: f32, // pressure
}

impl Particle {
    pub fn new(pos: Vec2) -> Self {
        Self { pos, vel: Vec2::ZERO, acc: Vec2::ZERO, rho: 0.0, p: 0.0 }
    }
}

#[derive(Resource)]
pub struct SPHState {
    pub h: f32, // smoothing length
    pub rho_0: f32, 
    pub k: f32, // stiffness
    pub mu: f32, // viscosity
    pub m: f32, // mass
    pub particles: Vec<Particle>,
}

impl SPHState {
    pub fn new(h: f32, rho_0: f32, k: f32, mu: f32, m: f32) -> Self {
        Self { h, rho_0, k, mu, m, particles: Vec::new()}
    }

    // initializing particles
    pub fn init_grid(&mut self, n_x: usize, n_y: usize, spacing: f32) {
        for iy in 0..n_y {
            for ix in 0..n_x {
                let x = ix as f32 * spacing;
                let y = iy as f32 * spacing;
                self.particles.push(Particle::new(Vec2::new(x, y)));
            }
        }
    }

    pub fn build_grid(&self) -> HashMap<Cell, Vec<usize>> {
        let mut grid: HashMap<Cell, Vec<usize>> = HashMap::with_capacity(self.particles.len());

        for (i, p) in self.particles.iter().enumerate() {
            let key = cell(p.pos, self.h);
            grid.entry(key).or_default().push(i);
        }
        grid
    }

    pub fn density_pressure_calc(&mut self) {
        let mut rho_vec = vec![0.0; self.particles.len()];
        let grid = self.build_grid();
        let h2 = self.h * self.h;

        for i in 0..self.particles.len() {
            let particle_i_po = self.particles[i].pos;
            let c = cell(particle_i_po, self.h);
            let mut rho = 0.0;

            // covering a 3 x 3 surrounding cells
            for ox in -1..=1 {
                for oy in -1..=1 {
                    if let Some(list) = grid.get(&(c + IVec2::new(ox, oy))) {
                        for &j in list {
                            let r2 = (particle_i_po - self.particles[j].pos).length_squared();
                            if r2 < h2 {
                                rho += self.m * w_poly6(r2, self.h);
                            }
                        }
                    }
                }
            }
            rho_vec[i] = rho;
        }
        for i in 0..self.particles.len() {
            self.particles[i].rho = rho_vec[i];
            self.particles[i].p = self.k * (rho_vec[i] - self.rho_0).max(0.0);
        }
    }

    fn accel_field_calc(&mut self) {
        let grid = self.build_grid();

        let mut acc_vec = vec![Vec2::ZERO; self.particles.len()];

        for i in 0..self.particles.len() {
            let particle_i = &self.particles[i];
            let pos_i = particle_i.pos;
            let p_i = particle_i.p;
            let vel_i = particle_i.vel;
            let cell_i = cell(pos_i, self.h);

            for ox in -1..=1 {
                for oy in -1..=1 {
                    if let Some(list) = grid.get(&(cell_i + IVec2::new(ox, oy))) {
                        for &j in list {
                            if i == j { continue; }
                            let particle_j = &self.particles[j];
                            let r = pos_i - particle_j.pos;
                            let r2 = r.length_squared();
                            
                            // acceleration due to pressure
                            let grad_spiky = grad_spiky_kernel(r, self.h);
                            // not text book but cheap to claculate for now
                            let a_p = -self.m * (p_i + particle_j.p) / (2.0 * particle_j.rho) * grad_spiky; 

                            // acceleration because of viscosity (fraction)
                            let r_mag = r2.sqrt(); // not len so not confused with len()
                            let laplacian = laplacian_visc(r_mag, self.h);
                            let a_v = self.mu * self.m * (particle_j.vel - vel_i) / particle_j.rho * laplacian;

                            acc_vec[i] += a_p + a_v;
                        }
                    }
                }
            }

            acc_vec[i] += GRAVITY;
        }

        for i in 0..self.particles.len() {
            self.particles[i].acc = acc_vec[i];
        }
    }

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


    // demo function ----------------------------------------------
    pub fn demo_block_5k() -> Self {
        let mut demo_sim_sph = Self::new(
        0.045,
        1000.0,
        3.0,
        0.2,
        1.6,
        );

        demo_sim_sph.init_grid(71, 71, 0.04);
        demo_sim_sph
    }
    // ------------------------------------------------------------
}
