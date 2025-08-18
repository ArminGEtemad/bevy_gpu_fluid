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

const PI : f32 = 3.141592653589793;
const H : f32 = 0.045; 
const MASS : f32 = 1.6;
const RHO0 : f32 = 1000.0;
const K : f32 = 3.0;

const H2 : f32 = H * H;
const H4 : f32 = H2 * H2;
const H8 : f32 = H4 * H4;

const POLY6_2D : f32 = 4.0 / (PI * H8);

fn poly6_contrib(r2: f32) -> f32 {
    if r2 >= H2 {
        return 0.0;
    }
    let k = H2 - r2;
    return POLY6_2D * k * k * k;
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i: u32 = gid.x;
    let n: u32 = arrayLength(&particles.data);
    if i >= n { return; }

    let xi = particles.data[i].pos;

    var rho_i: f32 = 0.0;
    var j: u32 = 0u;
    loop {
        if j >= n { break; }
        let xj = particles.data[j].pos;
        let r = xi - xj;
        let r2 = dot(r, r);
        rho_i = rho_i + MASS * poly6_contrib(r2);
        j = j + 1u;
    }

    particles.data[i].rho = rho_i;
}

// pressure from density
@compute @workgroup_size(256)
fn pressure_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i: u32 = gid.x;
    let n: u32 = arrayLength(&particles.data);
    if i >= n { return; }

    let rho_i = particles.data[i].rho;

    let p_i = K * (rho_i - RHO0);

    particles.data[i].p = p_i;
}