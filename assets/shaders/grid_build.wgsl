struct U32Buf {
    data: array<u32>,
};

struct GridBuildParams {
    num_cells: u32,
    _pad: vec3<u32>,
};

@group(0) @binding(0) var<storage, read_write> counts: U32Buf;
@group(0) @binding(1) var<uniform> gb: GridBuildParams;

// not used yet
@compute @workgroup_size(256)
fn clear_counts(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if i < gb.num_cells {
        counts.data[i] = 0u;
    }
}
