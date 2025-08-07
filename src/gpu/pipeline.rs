use std::borrow::Cow;

use bevy::prelude::*;
use bevy::render::render_resource::{
    CachedComputePipelineId, ComputePipeline, ComputePipelineDescriptor, PipelineCache,
    PushConstantRange, ShaderDefVal,
};

use crate::gpu::buffers::ParticleBindGroupLayout;


#[derive(Resource)]
pub struct DensityPipeline(pub ComputePipeline);


pub fn prepare_density_pipeline(
    mut commands: Commands,
    pipeline_cache: Res<PipelineCache>,
    layout: Res<ParticleBindGroupLayout>,
    mut pipeline_id: Local<Option<CachedComputePipelineId>>,
    assets: Res<AssetServer>,
) {
    if pipeline_id.is_none() {
        let shader: Handle<Shader> = assets.load("shaders/sph_density.wgsl");
        let desc = ComputePipelineDescriptor {
            label: Some("sph_density_pipeline".into()),
            layout: vec![layout.0.clone()],
            push_constant_ranges: Vec::<PushConstantRange>::new(),
            shader,
            shader_defs: Vec::<ShaderDefVal>::new(),
            entry_point: Cow::from("main"),
            zero_initialize_workgroup_memory: false,
        };
        *pipeline_id = Some(pipeline_cache.queue_compute_pipeline(desc));
        return; // waits for compilation
    }

    // where grabs the compiled GPU object.
    if let Some(id) = *pipeline_id {
        if let Some(pipeline) = pipeline_cache.get_compute_pipeline(id) {
            commands.insert_resource(DensityPipeline(pipeline.clone()));
        }
    }
}