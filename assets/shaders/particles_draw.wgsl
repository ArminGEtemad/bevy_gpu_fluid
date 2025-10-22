struct Particle {
    pos: vec2<f32>,
    vel: vec2<f32>,
    acc: vec2<f32>,
    rho: f32,
    p: f32,
};

struct ParticleBuf {
    data: array<Particle>,
};

struct DrawParams {
    view_proj: mat4x4<f32>,
    particle_size: f32,
    scale: f32,
    _pad: vec2<f32>,
    color: vec4<f32>,
};

@group(0) @binding(0) var<storage, read> particles: ParticleBuf;
@group(0) @binding(1) var<uniform> draw: DrawParams;

struct VsIn {
    @location(0) quad_pos: vec2<f32>,
};

struct VsOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) v_color: vec4<f32>,
};

@vertex
fn vs_main(in: VsIn, @builtin(instance_index) iid: u32) -> VsOut {
    let p = particles.data[iid].pos * draw.scale;
    let world = vec2<f32>(
        p.x + in.quad_pos.x * draw.particle_size,
        p.y + in.quad_pos.y * draw.particle_size
    );

    var out: VsOut;
    out.clip = draw.view_proj * vec4<f32>(world, 0.0, 1.0);
    out.v_color = draw.color;
    return out;
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    return in.v_color;
}
