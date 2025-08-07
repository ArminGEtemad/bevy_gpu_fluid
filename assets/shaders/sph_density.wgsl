struct Particle {
    pos : vec2f,
    vel : vec2f,
    rho : f32,
    p : f32,
};

@group(0) @binding(0)
var<storage, read_write> particles : array<Particle>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid : vec3u) {
    if (gid.x >= arrayLength(&particles)) {
        return;
    }
    
}
