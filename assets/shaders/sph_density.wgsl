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

struct GridParams {
    min_world: vec2<f32>,
    cell_size: f32,
    _pad0: f32,          // keep alignment in sync with Rust
    dims: vec2<u32>,
    _pad1: vec2<u32>,
};

@group(0) @binding(1)
var<storage, read> cell_starts : array<u32>;

@group(0) @binding(2)
var<storage, read> cell_entries : array<u32>;

@group(0) @binding(3)
var<uniform> grid : GridParams;

struct IntegrateParams {
    dt: f32,
    x_min: f32,
    x_max: f32,
    bounce: f32,
};

@group(0) @binding(4)
var<uniform> integ : IntegrateParams;

const PI : f32 = 3.141592653589793;
const MASS : f32 = 1.6;
const RHO0 : f32 = 1000.0;
const K    : f32 = 3.0;
const MU   : f32 = 0.2;
const G    : vec2<f32> = vec2<f32>(0.0, -9.81);

// ---------------- kernels --------------------

fn w_poly6(r2: f32) -> f32 {
    let h = grid.cell_size;
    let h2 = h * h;
    if r2 >= 0.0 && r2 <= h2 {

        let h4 = h2 * h2;
        let h8 = h4 * h4;
        let coeff = 4.0 / (PI * h8);
        let k = h2 - r2;
        return coeff * k * k * k;
    }
    return 0.0;
}

fn grad_spiky_kernel(r: vec2<f32>) -> vec2<f32> {
    let h = grid.cell_size;
    let r_len = length(r);
    if r_len == 0.0 || r_len >= h {
        return vec2<f32>(0.0, 0.0);
    }

    let h2 = h * h;
    let h5 = h2 * h2 * h;
    let coeff = -10.0 / (PI * h5);
    let factor = coeff * (h - r_len) * (h - r_len);
    return factor * (r / r_len);
}

fn laplacian_visc(r_len: f32) -> f32 {
    let h = grid.cell_size;
    if r_len == 0.0 || r_len >= h {
        return 0.0;
    }

    let h2 = h * h;
    let h5 = h2 * h2 * h;
    let coeff = 40.0 / (PI * h5);
    return coeff * (h - r_len);
}

// density 

fn cell_of_pos(pos: vec2<f32>) -> vec2<i32> {
    // cell index for the position (like CPU's floor(pos / h))
    let c = vec2<i32>(floor(pos / grid.cell_size));

    // recover the integer min cell from the uniform using round to avoid -ε issues
    let origin = vec2<i32>(
        i32(round(grid.min_world.x / grid.cell_size)),
        i32(round(grid.min_world.y / grid.cell_size))
    );

    return c - origin;
}

fn cell_id(ix: i32, iy: i32) -> u32 {
    return u32(ix) + u32(iy) * grid.dims.x;
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    let h = grid.cell_size;
    let h2 = h * h;
    let n = arrayLength(&particles.data);
    if i >= n { return; }

    let xi = particles.data[i].pos;
    var rho: f32 = 0.0;

    let c0 = cell_of_pos(xi);
    let nx = i32(grid.dims.x);
    let ny = i32(grid.dims.y);

    var oy: i32 = -1;
    loop {
        if oy > 1 { break; }
        var ox: i32 = -1;
        loop {
            if ox > 1 { break; }

            let cx_i = c0.x + ox;
            let cy_i = c0.y + oy;

            // skip cells outside the grid (no clamping -> no duplicates)
            if cx_i >= 0 && cx_i < nx && cy_i >= 0 && cy_i < ny {
                let cid = cell_id(cx_i, cy_i);

                let start = cell_starts[cid];
                let end = cell_starts[cid + 1u];

                var k = start;
                loop {
                    if k >= end { break; }
                    let j = cell_entries[k];
                    let rvec = xi - particles.data[j].pos;
                    let r2 = dot(rvec, rvec);
                    if r2 < h2 {
                        rho += MASS * w_poly6(r2);
                    }
                    k = k + 1u;
                }
            }

            ox = ox + 1;
        }
        oy = oy + 1;
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
    let h = grid.cell_size;
    let h2 = h * h;
    let n = arrayLength(&particles.data);
    if i >= n { return; }

    let xi = particles.data[i].pos;
    let vi = particles.data[i].vel;
    let pi = particles.data[i].p;

    var acc_i: vec2<f32> = vec2<f32>(0.0, 0.0);

    // --- 3×3 neighbor cells (same as density) ---
    let c0 = cell_of_pos(xi);
    let nx = i32(grid.dims.x);
    let ny = i32(grid.dims.y);

    var oy: i32 = -1;
    loop {
        if oy > 1 { break; }
        var ox: i32 = -1;
        loop {
            if ox > 1 { break; }

            let cx_i = c0.x + ox;
            let cy_i = c0.y + oy;

            if cx_i >= 0 && cx_i < nx && cy_i >= 0 && cy_i < ny {
                let cid = cell_id(cx_i, cy_i);
                let start = cell_starts[cid];
                let end = cell_starts[cid + 1u];

                var k = start;
                loop {
                    if k >= end { break; }
                    let j = cell_entries[k];

                    if j != i {
                        let xj = particles.data[j].pos;
                        let vj = particles.data[j].vel;
                        let rhoj = particles.data[j].rho;
                        let pj = particles.data[j].p;

                        let rvec = xi - xj;
                        let r2 = dot(rvec, rvec);
                        if r2 < h2 {
                            let r_len = sqrt(max(r2, 1e-12));

                            let grad = grad_spiky_kernel(rvec);
                            let a_p = -MASS * (pi + pj) / (2.0 * rhoj) * grad;

                            let lap = laplacian_visc(r_len);
                            let a_v = MU * MASS * (vj - vi) / rhoj * lap;

                            acc_i += a_p + a_v;
                        }
                    }

                    k = k + 1u;
                }
            }

            ox = ox + 1;
        }
        oy = oy + 1;
    }

    // gravity
    acc_i += G;

    particles.data[i].acc = acc_i;
}

@compute @workgroup_size(256)
fn integrate_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    let n = arrayLength(&particles.data);
    if i >= n { return; }

    var p = particles.data[i];

    p.vel += p.acc * integ.dt;
    p.pos += p.vel * integ.dt;

    // boundaries (match CPU)
    if p.pos.y < 0.0 {
        p.pos.y = 0.0;
        p.vel.y *= integ.bounce;
    }
    if p.pos.x > integ.x_max {
        p.pos.x = integ.x_max;
        p.vel.x *= integ.bounce;
    }
    if p.pos.x < integ.x_min {
        p.pos.x = integ.x_min;
        p.vel.x *= integ.bounce;
    }

    particles.data[i] = p;
}