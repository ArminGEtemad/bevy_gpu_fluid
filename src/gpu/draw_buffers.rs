use bevy::prelude::*;
use bevy::render::render_resource::*;
use bevy::render::renderer::{RenderDevice, RenderQueue};

use crate::gpu::buffers::ExtractedParticleBuffer;
use bevy::render::Extract;
use bevy::render::extract_resource::ExtractResource;

// ---------------- Types ----------------

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct DrawParams {
    pub view_proj: [[f32; 4]; 4],
    pub particle_size: f32,
    pub scale: f32,
    pub _pad: [f32; 2],
    pub color: [f32; 4],
}

#[derive(Resource)]
pub struct DrawParamsBuffer {
    pub buffer: Buffer,
}

#[derive(Resource, Clone)]
pub struct DrawBindGroupLayout(pub BindGroupLayout);

#[derive(Resource)]
pub struct DrawBindGroup(pub BindGroup);

#[derive(Resource)]
pub struct QuadVertexBuffer {
    pub buffer: Buffer,
}

#[derive(Resource, Clone, ExtractResource)]
pub struct ExtractedDrawParamsBuffer {
    pub buffer: Buffer,
}

const QUAD_VERTS: &[[f32; 2]] = &[
    [-0.5, -0.5],
    [0.5, -0.5],
    [0.5, 0.5],
    [-0.5, -0.5],
    [0.5, 0.5],
    [-0.5, 0.5],
];

pub fn extract_draw_params_buffer(mut commands: Commands, dp: Extract<Res<DrawParamsBuffer>>) {
    commands.insert_resource(ExtractedDrawParamsBuffer {
        buffer: dp.buffer.clone(),
    });
}

// Create a default DrawParams UBO
pub fn init_draw_params(mut commands: Commands, rd: Res<RenderDevice>) {
    let dp = DrawParams {
        view_proj: glam::Mat4::IDENTITY.to_cols_array_2d(),
        particle_size: 0.15, // world units; tweak later
        scale: 1.0,
        _pad: [0.0; 2],
        color: [0.0, 1.0, 1.0, 1.0],
    };
    let buffer = rd.create_buffer_with_data(&BufferInitDescriptor {
        label: Some("draw_params_uniform"),
        contents: bytemuck::bytes_of(&dp),
        usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
    });
    commands.insert_resource(DrawParamsBuffer { buffer });
}

// Update the UBO each frame (cheap). Keep identity until we wire a camera.
pub fn update_draw_params(rq: Res<RenderQueue>, dp: Res<DrawParamsBuffer>) {
    // Use your real sim bounds here; these are safe defaults for x∈[0,10], y∈[0,6]
    let min_x = 0.0;
    let max_x = 10.0;
    let min_y = 0.0;
    let max_y = 6.0;
    let view_proj = glam::Mat4::orthographic_rh(min_x, max_x, min_y, max_y, -1.0, 1.0);

    let dp_cpu = DrawParams {
        view_proj: view_proj.to_cols_array_2d(),
        particle_size: 0.1, // world units; bump if too small
        scale: 1.0,
        _pad: [0.0; 2],
        color: [0.0, 1.0, 1.0, 1.0],
    };
    rq.write_buffer(&dp.buffer, 0, bytemuck::bytes_of(&dp_cpu));
}

// ---------------- Systems (Render world) ----------------

// Make a small quad VB for instancing later
pub fn init_quad_vb(mut commands: Commands, rd: Res<RenderDevice>) {
    let vb = rd.create_buffer_with_data(&BufferInitDescriptor {
        label: Some("instanced_quad_vb"),
        contents: bytemuck::cast_slice(QUAD_VERTS),
        usage: BufferUsages::VERTEX,
    });
    commands.insert_resource(QuadVertexBuffer { buffer: vb });
}

// Layout: 0 = particles SSBO (VERTEX visibility later), 1 = draw params UBO
pub fn init_draw_bgl(mut commands: Commands, rd: Res<RenderDevice>) {
    let bgl = rd.create_bind_group_layout(
        Some("draw_bgl"),
        &[
            BindGroupLayoutEntry {
                binding: 0,
                visibility: ShaderStages::VERTEX, // we’ll fetch in vertex
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            BindGroupLayoutEntry {
                binding: 1,
                visibility: ShaderStages::VERTEX | ShaderStages::FRAGMENT,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
    );
    commands.insert_resource(DrawBindGroupLayout(bgl));
    info!("draw_bgl is READY");
}

// Create the BG: particles SSBO + draw params UBO
pub fn prepare_draw_bg(
    mut commands: Commands,
    rd: Res<RenderDevice>,
    layout: Option<Res<DrawBindGroupLayout>>,
    particles: Option<Res<ExtractedParticleBuffer>>,
    dp: Option<Res<ExtractedDrawParamsBuffer>>,
) {
    if let (Some(layout), Some(particles), Some(dp)) =
        (layout.as_ref(), particles.as_ref(), dp.as_ref())
    {
        let bg = rd.create_bind_group(
            Some("draw_bg"),
            &layout.0,
            &[
                BindGroupEntry {
                    binding: 0,
                    resource: particles.buffer.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: dp.buffer.as_entire_binding(),
                },
            ],
        );
        commands.insert_resource(DrawBindGroup(bg));
        info!("draw_bg is READY");
    } else {
        if layout.is_none() {
            info!("prepare_draw_bg: no DrawBindGroupLayout yet");
        }
        if particles.is_none() {
            info!("prepare_draw_bg: no ExtractedParticleBuffer yet");
        }
        if dp.is_none() {
            info!("prepare_draw_bg: no ExtractedDrawParamsBuffer yet");
        }
    }
}
