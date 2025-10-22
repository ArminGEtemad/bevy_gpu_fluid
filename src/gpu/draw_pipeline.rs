use bevy::asset::AssetServer;
use bevy::prelude::*;
use bevy::render::render_resource::TextureFormat;
use bevy::render::render_resource::{
    CachedPipelineState, CachedRenderPipelineId, ColorTargetState, ColorWrites, FragmentState,
    MultisampleState, PipelineCache, PrimitiveState, RenderPipelineDescriptor, Shader,
    VertexAttribute, VertexBufferLayout, VertexFormat, VertexState,
};

use super::draw_buffers::DrawBindGroupLayout;
#[derive(Resource)]
pub struct DrawPipeline(pub CachedRenderPipelineId);

pub fn prepare_draw_pipeline(
    mut commands: Commands,
    cache: Res<PipelineCache>,
    bgl: Option<Res<DrawBindGroupLayout>>,
    assets: Res<AssetServer>,
    mut cached: Local<Option<CachedRenderPipelineId>>,
) {
    let Some(bgl) = bgl else {
        return;
    };

    let shader: Handle<Shader> = assets.load("shaders/particles_draw.wgsl");

    if cached.is_none() {
        let vbuf_layout = VertexBufferLayout {
            array_stride: std::mem::size_of::<[f32; 2]>() as u64,
            step_mode: bevy::render::render_resource::VertexStepMode::Vertex,
            attributes: vec![VertexAttribute {
                format: VertexFormat::Float32x2,
                offset: 0,
                shader_location: 0,
            }],
        };

        let desc = RenderPipelineDescriptor {
            label: Some("particles_draw_pipeline".into()),
            layout: vec![bgl.0.clone()],
            vertex: VertexState {
                shader: shader.clone(),
                entry_point: "vs_main".into(),
                shader_defs: vec![],
                buffers: vec![vbuf_layout],
            },
            fragment: Some(FragmentState {
                shader,
                entry_point: "fs_main".into(),
                shader_defs: vec![],
                targets: vec![Some(ColorTargetState {
                    format: TextureFormat::Rgba8UnormSrgb,
                    blend: Some(bevy::render::render_resource::BlendState::ALPHA_BLENDING),
                    write_mask: ColorWrites::ALL,
                })],
            }),
            primitive: PrimitiveState::default(),
            depth_stencil: None,
            multisample: MultisampleState {
                count: 4, // match the RenderPass saw in the logs
                ..Default::default()
            },
            push_constant_ranges: vec![],
            zero_initialize_workgroup_memory: false,
        };

        let id = cache.queue_render_pipeline(desc);
        *cached = Some(id);
        info!("draw_pipeline QUEUED");
        return;
    }

    if let Some(id) = *cached {
        match cache.get_render_pipeline_state(id) {
            &CachedPipelineState::Ok(_) => {
                info!("draw_pipeline READY");
                commands.insert_resource(DrawPipeline(id));
            }
            &CachedPipelineState::Err(ref err) => {
                error!("draw_pipeline ERROR: {err:?}");
            }
            &CachedPipelineState::Queued => {
                info!("draw_pipeline QUEUED (waiting for compilation)...");
            }
            &CachedPipelineState::Creating(_) => {
                info!("draw_pipeline CREATING (compiling now)...");
            }
        }
    }
}
