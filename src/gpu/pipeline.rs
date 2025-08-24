/* used https://docs.rs/bevy/latest/bevy/render/render_resource/struct.ComputePass.html?utm_source=chatgpt.com
as my source for computepass */

use std::borrow::Cow;

use bevy::prelude::*;
use bevy::render::graph::CameraDriverLabel;
use bevy::render::render_graph::{
    Node, NodeRunError, RenderGraph, RenderGraphContext, RenderLabel,
};
use bevy::render::render_resource::{
    CachedComputePipelineId, ComputePassDescriptor, ComputePipeline, ComputePipelineDescriptor,
    PipelineCache, PushConstantRange, ShaderDefVal,
};
use bevy::render::renderer::RenderContext;

use crate::gpu::buffers::{
    ExtractedAllowCopy, ExtractedParticleBuffer, ExtractedReadbackBuffer, ParticleBindGroup,
    ParticleBindGroupLayout,
};

// ==================== resources ======================================
#[derive(Resource)]
pub struct DensityPipeline(pub ComputePipeline);

#[derive(Resource)]
pub struct PressurePipeline(pub ComputePipeline);

#[derive(Resource)]
pub struct ForcesPipeline(pub ComputePipeline);

#[derive(Debug, Hash, PartialEq, Eq, Clone, RenderLabel)]
pub struct DensityPassLabel;
#[derive(Default)]
struct DensityNode;

#[derive(Resource)]
pub struct IntegratePipeline(pub ComputePipeline);

// =====================================================================

// ========================== systems ==================================

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
            info!("density_pipe_line is READY");

            commands.insert_resource(DensityPipeline(pipeline.clone()));
        }
    }
}

pub fn prepare_pressure_pipeline(
    mut commands: Commands,
    pipeline_cache: Res<PipelineCache>,
    layout: Res<ParticleBindGroupLayout>,
    mut pipeline_id: Local<Option<CachedComputePipelineId>>,
    assets: Res<AssetServer>,
) {
    if pipeline_id.is_none() {
        let shader: Handle<Shader> = assets.load("shaders/sph_density.wgsl");
        let desc = ComputePipelineDescriptor {
            label: Some("sph_pressure_pipeline".into()),
            layout: vec![layout.0.clone()],
            push_constant_ranges: Vec::<PushConstantRange>::new(),
            shader,
            shader_defs: Vec::<ShaderDefVal>::new(),
            entry_point: Cow::from("pressure_main"),
            zero_initialize_workgroup_memory: false,
        };
        *pipeline_id = Some(pipeline_cache.queue_compute_pipeline(desc));
        return; // waits for compilation
    }

    // where grabs the compiled GPU object.
    if let Some(id) = *pipeline_id {
        if let Some(pipeline) = pipeline_cache.get_compute_pipeline(id) {
            commands.insert_resource(PressurePipeline(pipeline.clone()));
        }
    }
}

pub fn prepare_forces_pipeline(
    mut commands: Commands,
    pipeline_cache: Res<PipelineCache>,
    layout: Res<ParticleBindGroupLayout>,
    mut pipeline_id: Local<Option<CachedComputePipelineId>>,
    assets: Res<AssetServer>,
) {
    if pipeline_id.is_none() {
        let shader: Handle<Shader> = assets.load("shaders/sph_density.wgsl");
        let desc = ComputePipelineDescriptor {
            label: Some("sph_forces_pipeline".into()),
            layout: vec![layout.0.clone()],
            shader,
            entry_point: Cow::Borrowed("forces_main"),
            push_constant_ranges: vec![],
            shader_defs: vec![],
            zero_initialize_workgroup_memory: false,
        };
        *pipeline_id = Some(pipeline_cache.queue_compute_pipeline(desc));
        return;
    }
    if let Some(id) = *pipeline_id {
        if let Some(pipeline) = pipeline_cache.get_compute_pipeline(id) {
            commands.insert_resource(ForcesPipeline(pipeline.clone()));
        }
    }
}

pub fn prepare_integrate_pipeline(
    mut commands: Commands,
    pipeline_cache: Res<PipelineCache>,
    layout: Res<ParticleBindGroupLayout>,
    assets: Res<AssetServer>,
    mut cached: Local<Option<CachedComputePipelineId>>,
) {
    if cached.is_none() {
        let shader: Handle<Shader> = assets.load("shaders/sph_density.wgsl");
        let desc = ComputePipelineDescriptor {
            label: Some("sph_integrate_pipeline".into()),
            layout: vec![layout.0.clone()],
            push_constant_ranges: Vec::<PushConstantRange>::new(),
            shader,
            shader_defs: Vec::<ShaderDefVal>::new(),
            entry_point: Cow::from("integrate_main"),
            zero_initialize_workgroup_memory: false,
        };
        *cached = Some(pipeline_cache.queue_compute_pipeline(desc));
        return; // wait for compilation
    }

    if let Some(id) = *cached {
        if let Some(pipeline) = pipeline_cache.get_compute_pipeline(id) {
            commands.insert_resource(IntegratePipeline(pipeline.clone()));
        }
    }
}
// dispatch compute shader

impl Node for DensityNode {
    fn run(
        &self,
        _graph: &mut RenderGraphContext,
        render_context: &mut RenderContext,
        world: &World,
    ) -> Result<(), NodeRunError> {
        // return because the calculations doesn't exist yet
        let Some(pipeline) = world.get_resource::<DensityPipeline>() else {
            return Ok(());
        };
        let Some(bind_group) = world.get_resource::<ParticleBindGroup>() else {
            return Ok(());
        };
        let Some(extracted) = world.get_resource::<ExtractedParticleBuffer>() else {
            return Ok(());
        };

        // ==== debugging info ====
        if world.get_resource::<DensityPipeline>().is_none() {
            info!("Info Node: no pipeline");
            return Ok(());
        }
        if world.get_resource::<ParticleBindGroup>().is_none() {
            info!("Info Node: no particle bind group");
            return Ok(());
        }
        if world.get_resource::<ExtractedParticleBuffer>().is_none() {
            info!("Info Node: no particle buffer");
            return Ok(());
        }
        // ========================

        // how many workgroups do we actually need?
        let n = extracted.num_particles.max(1);
        let workgroups = (n + 255) / 256; // for every 256 -> 1 workgroup
        info!("Info Node: DISPATCH, N = {}, groups = {}", n, workgroups);

        let mut pass = render_context
            .command_encoder()
            .begin_compute_pass(&ComputePassDescriptor::default());

        pass.set_pipeline(&pipeline.0); // bind the compiled pipeline
        pass.set_bind_group(0, &bind_group.0, &[]); // inject the particle buffer
        pass.dispatch_workgroups(workgroups, 1, 1); // start the shader

        if let Some(pressure) = world.get_resource::<PressurePipeline>() {
            pass.set_pipeline(&pressure.0);
            pass.set_bind_group(0, &bind_group.0, &[]);
            pass.dispatch_workgroups(workgroups, 1, 1);
            info!("Info Node: DISPATCH pressure N = {n}, groups = {workgroups}");
        } else {
            info!("Info Node: pressure SKIPPED (pipeline not working/not ready)");
        }

        if let Some(forces) = world.get_resource::<ForcesPipeline>() {
            pass.set_pipeline(&forces.0);
            pass.set_bind_group(0, &bind_group.0, &[]);
            pass.dispatch_workgroups(workgroups, 1, 1);
            info!("Info Node: DISPATCH forces N = {n}, groups = {workgroups}");
        } else {
            info!("Info Node: forces SKIPPED (pipeline not working/not ready)");
        }

        if let Some(integrate) = world.get_resource::<IntegratePipeline>() {
            pass.set_pipeline(&integrate.0);
            pass.set_bind_group(0, &bind_group.0, &[]);
            pass.dispatch_workgroups(workgroups, 1, 1);
            info!("Info Node: DISPATCH integrate N = {n}, groups = {workgroups}");
        } else {
            info!("Info Node: integrate SKIPPED (pipeline not ready)");
        }

        drop(pass); // pass must end before encoding copies
        let Some(readback) = world.get_resource::<ExtractedReadbackBuffer>() else {
            return Ok(());
        };

        let allow_copy = world
            .get_resource::<ExtractedAllowCopy>()
            .map(|f| f.0)
            .unwrap_or(true);

        if allow_copy {
            render_context.command_encoder().copy_buffer_to_buffer(
                &extracted.buffer,
                0,
                &readback.buffer,
                0,
                readback.size_bytes,
            );
            info!(
                "Info Node: COPY particles -> readback ({} bytes)",
                readback.size_bytes
            );
        } else {
            info!("Info Node: copy is SKIPPED");
        }

        Ok(())
    }
}

pub fn add_density_node_to_graph(render_app: &mut bevy::app::SubApp) {
    let mut graph = render_app.world_mut().resource_mut::<RenderGraph>();
    graph.add_node(DensityPassLabel, DensityNode::default());
    graph.add_node_edge(DensityPassLabel, CameraDriverLabel);
}
