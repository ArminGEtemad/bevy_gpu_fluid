/* used https://docs.rs/bevy/latest/bevy/render/render_resource/struct.ComputePass.html?utm_source=chatgpt.com
as my source for computepass */ 

use std::borrow::Cow;

use bevy::prelude::*;
use bevy::render::render_resource::{
    CachedComputePipelineId, ComputePipeline, ComputePipelineDescriptor, PipelineCache,
    PushConstantRange, ShaderDefVal, ComputePassDescriptor,
};
use bevy::render::graph::CameraDriverLabel;
use bevy::render::render_graph::{
    self as render_graph, Node, NodeRunError, RenderGraph, RenderGraphContext, RenderLabel,
};
use bevy::render::renderer::RenderContext;

use crate::gpu::buffers::{ParticleBindGroupLayout, ParticleBindGroup, ExtractedParticleBuffer};


#[derive(Resource)]
pub struct DensityPipeline(pub ComputePipeline);

#[derive(Debug, Hash, PartialEq, Eq, Clone, RenderLabel)]
pub struct DensityPassLabel;

#[derive(Default)]
struct DensityNode;

impl Node for DensityNode {
    fn run(
        &self,
        _graph: &mut RenderGraphContext,
        render_context: &mut RenderContext,
        world: &World,
    ) -> Result<(), NodeRunError> {
        // return because the calculations doesn't exist yet
        let Some(pipeline) = world.get_resource::<DensityPipeline>() else { return Ok(()); };
        let Some(bind_group) = world.get_resource::<ParticleBindGroup>() else { return Ok(()); };
        let Some(extracted) = world.get_resource::<ExtractedParticleBuffer>() else { return Ok(()); };

        // how many workgroups do we actually need?
        let n = extracted.num_particles.max(1);
        let workgroups = (n + 255) / 256;

        let mut pass = render_context
            .command_encoder()
            .begin_compute_pass(&ComputePassDescriptor::default());
        
        pass.set_pipeline(&pipeline.0);
        pass.set_bind_group(0, &bind_group.0, &[]);
        pass.dispatch_workgroups(workgroups, 1, 1);

        Ok(())
    }
}

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

pub fn add_density_node_to_graph(render_app: &mut bevy::app::SubApp) {
    let mut graph = render_app.world_mut().resource_mut::<RenderGraph>();
    graph.add_node(DensityPassLabel, DensityNode::default());
    graph.add_node_edge(DensityPassLabel, CameraDriverLabel);
}