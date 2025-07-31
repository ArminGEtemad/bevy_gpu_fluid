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
    pub rho: f32,
    pub p: f32,
    // TODO: viscosity has to be added at some point?
}

impl Particle {
    pub fn new(pos: Vec2) -> Self {
        Self {
            pos, 
            vel: Vec2::ZERO,
            rho: 0.0,
            p: 0.0,
        }
    }
}

pub struct SPHState {
    pub h: f32, // https://en.wikipedia.org/wiki/Smoothed-particle_hydrodynamics
    pub rho_0: f32, // (not inital density but the rest density)
    pub k: f32, // 
    pub m: f32,
    pub particles: Vec<Particle>,
}

type Cell = IVec2;

impl SPHState {
    pub fn new(h: f32, rho_0: f32, k: f32, m: f32) -> Self {
        Self { h, rho_0, k, m, particles: Vec::new()}
    }

    // initializing particles
    pub fn init_grid(&mut self, n_x: usize, n_y: usize, space: f32) {
        for iy in 0..n_y {
            for ix in 0..n_x {
                let x = ix as f32 * space;
                let y = iy as f32 * space;
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
            let p_i_po = self.particles[i].pos;
            let c = cell(p_i_po, self.h);
            let mut rho = 0.0;

            // covering a 3 x 3 surrounding cells
            for ox in -1..=1 {
                for oy in -1..=1 {
                    if let Some(list) = grid.get(&(c + IVec2::new(ox, oy))) {
                        for &j in list {
                            let r2 = (p_i_po - self.particles[j].pos).length_squared();
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
}