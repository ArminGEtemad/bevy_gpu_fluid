// smoothed particle hydrodynamics in 2D (CPU prototype)

use glam::Vec2; // glam is a linear algebra library
// https://docs.rs/glam/latest/glam

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
    pub particles: Vec<Particle>,
}

impl SPHState {
    pub fn new(h: f32) -> Self {
        Self { h, particles: Vec::new()}
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

    // solve a differential equation?
}