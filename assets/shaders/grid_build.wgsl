// ---------- Types ----------
struct U32AtomicBuf {
    data: array<atomic<u32>>,
};
struct U32Buf {
    data: array<u32>,
};

struct GridBuildParams {
    num_cells: u32,
    _pad: vec3<u32>,
};

struct Particle {
    pos: vec2<f32>,
    vel: vec2<f32>,
    acc: vec2<f32>,
    rho: f32,
    p: f32,
};
struct ParticleBuf {
    data: array<Particle>, // runtime-sized array must be last
}

struct GridParams {
    min_world: vec2<f32>,
    cell_size: f32,
    _pad0: f32,
    dims: vec2<u32>,
    _pad1: vec2<u32>,
};

struct BlockSumsBuf {
    data: array<u32>,
};

// ---------- Module-scope shared memory for block_scan ----------
var<workgroup> wg_s: array<u32, 256u>;

// =================== ClearCounts (group 0) ===================
@group(0) @binding(0) var<storage, read_write> counts_rw: U32AtomicBuf;
@group(0) @binding(1) var<uniform> gb: GridBuildParams;

@compute @workgroup_size(256)
fn clear_counts(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if i < gb.num_cells {
        atomicStore(&counts_rw.data[i], 0u);
    }
}

// =================== Histogram (group 0) ===================
@group(0) @binding(0) var<storage, read> particles:  ParticleBuf;
@group(0) @binding(1) var<storage, read_write> counts_hist: U32AtomicBuf;
@group(0) @binding(2) var<uniform> grid: GridParams;

fn cell_index(p: vec2<f32>) -> u32 {
    let h = grid.cell_size;
    // CPU-compatible: floor(pos / h) - round(min_world / h)
    let c = vec2<i32>(floor(p / h));
    let origin = vec2<i32>(round(grid.min_world / h));

    let ix = clamp(c.x - origin.x, 0, i32(grid.dims.x) - 1);
    let iy = clamp(c.y - origin.y, 0, i32(grid.dims.y) - 1);
    return u32(iy) * grid.dims.x + u32(ix);
}

@compute @workgroup_size(256)
fn histogram(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if i >= arrayLength(&particles.data) { return; }
    let idx = cell_index(particles.data[i].pos);
    _ = atomicAdd(&counts_hist.data[idx], 1u);
}

// =================== Prefix / Block Scan family (group 0) ===================
// Shared bindings for counts→starts passes:
@group(0) @binding(0) var<storage, read> counts_ro : U32AtomicBuf; // read via atomicLoad
@group(0) @binding(1) var<storage, read_write> starts_rw : U32Buf; // where we write prefixes
@group(0) @binding(2) var<storage, read_write> block_sums: BlockSumsBuf; // used by block_scan

// Naive O(n^2) exclusive prefix sum (used during bring-up)
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

// Per-block (256) exclusive scan + block totals
@compute @workgroup_size(256)
fn block_scan(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
) {
    let i = gid.x;
    let li = lid.x; // 0..255
    let n = arrayLength(&counts_ro.data);
    let base = wid.x * 256u;

    // load to shared mem (0 if OOB)
    var v: u32 = 0u;
    if i < n {
        v = atomicLoad(&counts_ro.data[i]);
    }
    wg_s[li] = v;

    workgroupBarrier();

    // Hillis–Steele inclusive scan
    var offset: u32 = 1u;
    loop {
        if offset >= 256u { break; }

        var t: u32 = 0u;
        if li >= offset {
            t = wg_s[li - offset];
        }

        workgroupBarrier();
        wg_s[li] = wg_s[li] + t;
        workgroupBarrier();

        offset = offset << 1u;
    }

    var excl: u32 = 0u;
    if li != 0u {
        excl = wg_s[li - 1u];
    }

    if i < n {
        starts_rw.data[i] = excl;
    }

    // last thread writes this block's total
    if li == 255u {
        // how many elements this block really covers
        let remain = select(0u, n - base, n > base);

        var last: u32 = 255u;
        if remain == 0u {
            last = 0u;
        } else if remain < 256u {
            last = remain - 1u;
        }

        let sum = wg_s[last];
        block_sums.data[wid.x] = sum;
    }
}

@group(0) @binding(0) var<storage, read_write> block_rw : BlockSumsBuf;

@compute @workgroup_size(256)
fn block_sums_scan(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    let n = arrayLength(&block_rw.data);
    if i >= n { return; }

    var sum: u32 = 0u;
    var j: u32 = 0u;
    loop {
        if j >= i { break; }
        sum = sum + block_rw.data[j];
        j = j + 1u;
    }
    block_rw.data[i] = sum;
}

@compute @workgroup_size(256)
fn add_back_block_offsets(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    let n = arrayLength(&starts_rw.data);
    if i >= n { return; }

    let block = i / 256u;
    let offs = block_sums.data[block];
    starts_rw.data[i] = starts_rw.data[i] + offs;
}
@compute @workgroup_size(1)
fn write_sentinel(@builtin(global_invocation_id) gid: vec3<u32>) {
    if gid.x > 0u { return; }

    // number of real cells == length of counts
    let n_cells = arrayLength(&counts_ro.data);
    if n_cells == 0u { return; }

    let last_prefix = starts_rw.data[n_cells - 1u];
    let last_count = atomicLoad(&counts_ro.data[n_cells - 1u]);

    // exclusive scan sentinel: starts[num_cells] = sum(counts)
    starts_rw.data[n_cells] = last_prefix + last_count;
}

// =================== Scatter (group 0) ===================
// Particles → Entries using Starts + per-cell Cursor
@group(0) @binding(0) var<storage, read> particles_scatter : ParticleBuf;
@group(0) @binding(1) var<storage, read> starts_scatter : U32Buf;
@group(0) @binding(2) var<storage, read_write> cursor_rw : U32AtomicBuf;
@group(0) @binding(3) var<storage, read_write> entries_rw : U32Buf;
@group(0) @binding(4) var<uniform> grid_scatter : GridParams;

fn cell_index_scatter(p: vec2<f32>) -> u32 {
    let h = grid_scatter.cell_size;
    let c = vec2<i32>(floor(p / h));
    let origin = vec2<i32>(round(grid_scatter.min_world / h));

    let ix = clamp(c.x - origin.x, 0, i32(grid_scatter.dims.x) - 1);
    let iy = clamp(c.y - origin.y, 0, i32(grid_scatter.dims.y) - 1);
    return u32(iy) * grid_scatter.dims.x + u32(ix);
}

@compute @workgroup_size(256)
fn scatter(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    let n = arrayLength(&particles_scatter.data);
    if i >= n { return; }

    let pos = particles_scatter.data[i].pos;
    let cell = cell_index_scatter(pos);

    // reserve a slot in this cell
    let base = starts_scatter.data[cell];
    let offs = atomicAdd(&cursor_rw.data[cell], 1u);
    let idx = base + offs;

    // bounds guard (paranoia)
    let entries_len = arrayLength(&entries_rw.data);
    if idx < entries_len {
        entries_rw.data[idx] = i;
    }
}