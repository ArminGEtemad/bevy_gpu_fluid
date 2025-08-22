struct Particle {
    pos: vec2<f32>,
    vel: vec2<f32>,
    acc: vec2<f32>,
    rho: f32,
    p: f32,
};

struct ParticleBuffer {
    data: array<Particle> // runtime-sized array must be last
};

@group(0) @binding(0)
var<storage, read_write> particles : ParticleBuffer;

const PI : f32 = 3.141592653589793;
const H : f32 = 0.045;
const MASS : f32 = 1.6;
const RHO0 : f32 = 1000.0;
const K : f32 = 3.0;
const MU : f32 = 0.2;
const G : vec2<f32> = vec2<f32>(0.0, -9.81);

const H2 : f32 = H * H;
const H4 : f32 = H2 * H2;
const H8 : f32 = H4 * H4;

const POLY6_2D      : f32 =  4.0 / (PI * H8);
const SPIKY_GRAD_2D : f32 = -10.0 / (PI * pow(H, 5.0));
const VISC_LAP_2D   : f32 =  40.0 / (PI * pow(H, 5.0));

// ---------------- kernels --------------------

fn w_poly6(r2: f32) -> f32 {
    if r2 >= 0.0 && r2 <= H2 {
        let k = H2 - r2;
        return POLY6_2D * k * k * k;
    }
    return 0.0;
}

fn grad_spiky_kernel(r: vec2<f32>) -> vec2<f32> {
    let r_len = length(r);
    if r_len == 0.0 || r_len >= H {
        return vec2<f32>(0.0, 0.0);
    }
    let factor = SPIKY_GRAD_2D * (H - r_len) * (H - r_len);
    return factor * (r / r_len);
}

fn laplacian_visc(r_len: f32) -> f32 {
    if r_len == 0.0 || r_len >= H {
        return 0.0;
    }
    return VISC_LAP_2D * (H - r_len);
}

// density 

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    let n = arrayLength(&particles.data);
    if i >= n { return; }

    let xi = particles.data[i].pos;
    var rho: f32 = 0.0;

    var j: u32 = 0u;
    loop {
        if j >= n { break; }
        let rvec = xi - particles.data[j].pos;
        let r2 = dot(rvec, rvec);
        if r2 < H2 {
            rho += MASS * w_poly6(r2);
        }
        j = j + 1u;
    }

    particles.data[i].rho = rho;
}

@compute @workgroup_size(256)
fn pressure_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    let n = arrayLength(&particles.data);
    if i >= n { return; }

    let rho_i = particles.data[i].rho;
    // CPU clamps to non-negative
    let p_i = max(0.0, K * (rho_i - RHO0));
    particles.data[i].p = p_i;
}

@compute @workgroup_size(256)
fn forces_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    let n = arrayLength(&particles.data);
    if i >= n { return; }

    let xi = particles.data[i].pos;
    let vi = particles.data[i].vel;
    let p_i = particles.data[i].p;

    var acc_i: vec2<f32> = vec2<f32>(0.0, 0.0);

    var j: u32 = 0u;
    loop {
        if j >= n { break; }
        if j != i {
            let xj = particles.data[j].pos;
            let vj = particles.data[j].vel;
            let rhoj = particles.data[j].rho;
            let pj = particles.data[j].p;

            let rvec = xi - xj;
            let r2 = dot(rvec, rvec);
            if r2 < H2 {
                let r_len = sqrt(max(r2, 1e-12));

                let grad = grad_spiky_kernel(rvec);
                let a_p = -MASS * (p_i + pj) / (2.0 * rhoj) * grad;

                let lap = laplacian_visc(r_len);
                let a_v = MU * MASS * (vj - vi) / rhoj * lap;

                acc_i += a_p + a_v;
            }
        }
        j = j + 1u;
    }

    // gravity
    acc_i += G;

    particles.data[i].acc = acc_i;
}

const DT     : f32 = 0.0005;
const X_MIN  : f32 = -5.0;
const X_MAX  : f32 = 3.0;
const BOUNCE : f32 = -3.0;

@compute @workgroup_size(256)
fn integrate_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    let n = arrayLength(&particles.data);
    if i >= n { return; }

    var x = particles.data[i].pos;
    var v = particles.data[i].vel;
    let a = particles.data[i].acc;

    // Euler
    v = v + a * DT;
    x = x + v * DT;

    // Boundaries (identical to CPU)
    // floor
    if x.y < 0.0 {
        x.y = 0.0;
        v.y = v.y * BOUNCE;
    }
    // right wall
    if x.x > X_MAX {
        x.x = X_MAX;
        v.x = v.x * BOUNCE;
    }
    // left wall
    if x.x < X_MIN {
        x.x = X_MIN;
        v.x = v.x * BOUNCE;
    }

    particles.data[i].pos = x;
    particles.data[i].vel = v;
}