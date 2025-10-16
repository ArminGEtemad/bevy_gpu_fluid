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
    let rel = (p - grid.min_world) / grid.cell_size;
    let cx = clamp(i32(floor(rel.x)), 0, i32(grid.dims.x) - 1);
    let cy = clamp(i32(floor(rel.y)), 0, i32(grid.dims.y) - 1);
    return u32(cy) * grid.dims.x + u32(cx);
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
