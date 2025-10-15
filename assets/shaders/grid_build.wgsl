struct U32AtomicBuf {
    data: array<atomic<u32>>,
}
struct U32Buf {
    data: array<u32>,
};

struct GridBuildParams {
    num_cells: u32,
    _pad: vec3<u32>,
};

struct Particle {
    pos: vec2<f32>,
};

struct ParticleBuf {
    data: array<Particle>,
};

struct GridParams {
    min_world: vec2<f32>,
    cell_size: f32,
    _pad0: f32,
    dims: vec2<u32>,
    _pad1: vec2<u32>,
};

@group(0) @binding(0) var<storage, read_write> counts_rw: U32AtomicBuf;
@group(0) @binding(1) var<uniform> gb: GridBuildParams;

@compute @workgroup_size(256)
fn clear_counts(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if i < gb.num_cells {
        atomicStore(&counts_rw.data[i], 0u);
    }
}

@group(0) @binding(0) var<storage, read> particles: ParticleBuf;
@group(0) @binding(1) var<storage, read_write> counts_hist: U32AtomicBuf;
@group(0) @binding(2) var<uniform> grid: GridParams;

fn cell_index(p: vec2<f32>) -> u32 {
    let rel = (p - grid.min_world) / grid.cell_size;
    
    // floor to cell coords
    let cx = clamp(i32(floor(rel.x)), 0, i32(grid.dims.x) - 1);
    let cy = clamp(i32(floor(rel.y)), 0, i32(grid.dims.y) - 1);
    return u32(cy) * grid.dims.x + u32(cx);
}

@compute @workgroup_size(256)
fn histogram(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if i >= arrayLength(&particles.data) { return; }
    let idx = cell_index(particles.data[i].pos);
    // atomic increment
    _ = atomicAdd(&counts_hist.data[idx], 1u);
}

@group(0) @binding(0) var<storage, read> counts_ro : U32AtomicBuf;
@group(0) @binding(1) var<storage, read_write> starts_rw : U32Buf;

@compute @workgroup_size(256)
fn prefix_sum_naive(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    let n = arrayLength(&counts_ro.data);
    if i >= n { return; }

    var sum: u32 = 0u;
    var j: u32 = 0u;
    loop {
        if j >= i { break; }
        sum += atomicLoad(&counts_ro.data[j]);
        j += 1u;
    }
    starts_rw.data[i] = sum;
}
