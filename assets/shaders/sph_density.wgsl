struct Particle {
    pos: vec2<f32>,
    vel: vec2<f32>,
    rho: f32,
    p: f32,
};

@group(0) @binding(0)
var<storage, read_write> particles : array<Particle>;

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
    let N = arrayLength(&particles);
    if i >= N { return; }

    // Must give me [0, 1, 2, 3, 4]
    particles[i].rho = f32(i);
}