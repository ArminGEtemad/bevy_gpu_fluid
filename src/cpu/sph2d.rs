// smoothed particle hydrodynamics in 2D (CPU prototype)

use glam::{Vec2, IVec2}; // glam is a linear algebra library
// https://docs.rs/glam/latest/glam
use std::collections::HashMap;
use std::f32::consts::PI;

#[inline] // avoid function call overhead
fn cell(pos: Vec2, h: f32) -> IVec2 {
    // maps a particle's position to a grid cell coordinates
    (pos / h).floor().as_ivec2()
}

// https://www.cs.cornell.edu/courses/cs5643/2015sp/stuff/BridsonFluidsCourseNotes_SPH_pp83-86.pdf
// eq 15.5
// https://physics.stackexchange.com/questions/138700/kernel-normalization-in-smoothed-particle-hydrodynamcs
// for 2 D
#[inline]
fn w_poly6(r2: f32, h: f32) -> f32 {
    // let k: f32 = 315.0 / (64.0 * PI * h.powi(9)); 3D
    let k: f32 = 4.0 / (PI * h.powi(8));
    if r2 >= 0.0 && r2 <= h * h {
        k * (h * h - r2).powi(3)
    } else { 0.0 }
}

// https://www.cs.cmu.edu/~scoros/cs15467-s16/lectures/11-fluids2.pdf
// https://cs418.cs.illinois.edu/website/text/sph.html
// TODO: TeX my derivation analog to 3D
// https://courses.grainger.illinois.edu/CS418/sp2023/text/sph.html
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

/* to describe equation of motion of a fluid we need 
   the position of the elements,
   momentum (or velocity),
   density and
   pressure.*/ 

/* let's go with AoS meaning each struct will save
   all the properties. Reason is that it is easier 
   for now and more intuitive for a prototype */ 

#[derive(Clone, Debug)]
pub struct Particle {
    pub pos: Vec2,
    pub vel: Vec2,
    pub acc: Vec2,
    pub rho: f32,
    pub p: f32,
    // TODO: viscosity has to be added at some point?
}

impl Particle {
    pub fn new(pos: Vec2) -> Self {
        Self {
            pos, 
            vel: Vec2::ZERO,
            acc: Vec2::ZERO,
            rho: 0.0,
            p: 0.0,
        }
    }
}

pub struct SPHState {
    pub h: f32, // https://en.wikipedia.org/wiki/Smoothed-particle_hydrodynamics
    pub rho_0: f32, // (not inital density but the rest density)
    pub k: f32,
    pub mu: f32, 
    pub m: f32,
    pub particles: Vec<Particle>,
}

type Cell = IVec2;

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
        // hash grid, with key -> cell index, returns -> list of particle indices
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
            self.particles[i].p = self.k * (rho_vec[i] - self.rho_0).max(0.0); // eq 15.15 Bridson Fluids
                                                                               // making sure pressure is not negative
        }
    }

    fn accel_field_calc(&mut self) {
        let grid = self.build_grid();

        let mut acc_vec = vec![Vec2::ZERO; self.particles.len()];

        for i in 0..self.particles.len() {
            let particle_i = &self.particles[i];
            let pos_i = particle_i.pos;
            let p_i = particle_i.p;
            let rho_i = particle_i.rho;
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

            // gravity
            acc_vec[i] += Vec2::new(0.0, -9.81);
        }

        for i in 0..self.particles.len() {
            self.particles[i].acc = acc_vec[i];
        }
    }

    pub fn integrate(&mut self, dt: f32) {
        for p in &mut self.particles {
            p.vel += p.acc * dt;
            p.pos += p.vel * dt;

            // bounce at the boundary
            if p.pos.y < 0.0 {
                p.pos.y = 0.0;
                p.vel.y *= -3.0;
            }
        }
    }

    pub fn step(&mut self, dt: f32) {
        self.density_pressure_calc();
        self.accel_field_calc();
        self.integrate(dt);
    }
}
