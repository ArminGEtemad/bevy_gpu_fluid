struct Particle {
    pos: vec2<f32>,
    vel: vec2<f32>,
    rho: f32,
    p: f32,
};

struct ParticleBuffer {
    data: array<Particle>,
};

@group(0) @binding(0)
var<storage, read_write> particles : ParticleBuffer;

// Constants
const PI : f32 = 3.14159265358979;
const H : f32 = 0.045;
const MASS : f32 = 1.6;

// POLY6 Kernel
fn poly6(r2: f32, h: f32) -> f32 {
    let h2: f32 = h * h;
    let h4: f32 = h2 * h2;
    let h8: f32 = h4 * h4;
    let k: f32 = 4.0 / (PI * h8);
    if r2 >= 0.0 && r2 <= h2 {
        return k * (h2 - r2) * (h2 - r2) * (h2 - r2);
    } else {
        return 0.0;
    }
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    let n = arrayLength(&particles.data);
    if i >= n { return; }

    let xi = particles.data[i].pos;

    var rho_i: f32 = 0.0;
    var j: u32 = 0u;

    loop {
        if j >= n { break; }
        let xj = particles.data[j].pos;
        let r = xi - xj;
        let r2 = dot(r, r);
        rho_i = rho_i + MASS * poly6(r2, H);
        j = j + 1u;
    }

    particles.data[i].rho = particles.data[i].rho + 0.0;
}