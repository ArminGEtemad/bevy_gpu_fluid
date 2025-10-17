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
use crate::gpu::grid_build::{
    AddBackBindGroup, AddBackBindGroupLayout, BlockSumsScanBindGroup, BlockSumsScanBindGroupLayout,
    GridBlockScanBindGroup, GridBlockScanBindGroupLayout, GridBlockSumsBuffer, GridBuildBindGroup,
    GridBuildBindGroupLayout, GridBuildParamsBuffer, GridCountsToStartsBindGroup,
    GridCountsToStartsBindGroupLayout, GridHistogramBindGroup, GridHistogramBindGroupLayout,
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

#[derive(Resource)]
pub struct ClearCountsPipeline(pub CachedComputePipelineId);

#[derive(Debug, Hash, PartialEq, Eq, Clone, RenderLabel)]
pub struct ClearCountsLabel;

#[derive(Default)]
pub struct ClearCountsNode;

#[derive(Resource)]
pub struct HistogramPipeline(pub CachedComputePipelineId);

#[derive(Debug, Hash, PartialEq, Eq, Clone, RenderLabel)]
pub struct HistogramPassLabel;

#[derive(Default)]
pub struct HistogramNode;

#[derive(Resource)]
pub struct PrefixSumNaivePipeline(pub CachedComputePipelineId);

#[derive(Debug, Hash, PartialEq, Eq, Clone, RenderLabel)]
pub struct PrefixSumNaivePassLabel;

#[derive(Default)]
pub struct PrefixSumNaiveNode;

#[derive(Resource)]
pub struct BlockScanPipeline(pub CachedComputePipelineId);

#[derive(Debug, Hash, PartialEq, Eq, Clone, RenderLabel)]
pub struct BlockScanPassLabel;

#[derive(Default)]
pub struct BlockScanNode;
#[derive(Resource)]
pub struct BlockSumsScanPipeline(pub CachedComputePipelineId);

#[derive(Debug, Hash, PartialEq, Eq, Clone, RenderLabel)]
pub struct BlockSumsScanPassLabel;

#[derive(Default)]
pub struct BlockSumsScanNode;

#[derive(Resource)]
pub struct AddBackPipeline(pub CachedComputePipelineId);

#[derive(Debug, Hash, PartialEq, Eq, Clone, RenderLabel)]
pub struct AddBackPassLabel;

#[derive(Default)]
pub struct AddBackNode;
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
            entry_point: Cow::Borrowed("main"),
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
            entry_point: Cow::Borrowed("pressure_main"),
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

pub fn prepare_clear_counts_pipeline(
    mut commands: Commands,
    pipeline_cache: Res<PipelineCache>,
    layout: Option<Res<GridBuildBindGroupLayout>>,
    assets: Res<AssetServer>,
    mut cached: Local<Option<CachedComputePipelineId>>,
    mut printed: Local<u8>, // 0 = none, 1 = queued, 2 = ready
) {
    let Some(layout) = layout else {
        // layout not ready this frame; normal on startup
        return;
    };

    if cached.is_none() {
        let shader: Handle<Shader> = assets.load("shaders/grid_build.wgsl");
        let desc = ComputePipelineDescriptor {
            label: Some("clear_counts_pipeline".into()),
            layout: vec![layout.0.clone()],
            push_constant_ranges: vec![],
            shader_defs: vec![],
            entry_point: Cow::Borrowed("clear_counts"),
            shader,
            zero_initialize_workgroup_memory: true,
        };
        let id = pipeline_cache.queue_compute_pipeline(desc);
        *cached = Some(id);
        commands.insert_resource(ClearCountsPipeline(id));
        if *printed == 0 {
            info!("Info Prepare: clear_counts QUEUED");
            *printed = 1;
        }
        return;
    }

    if let Some(id) = *cached {
        if pipeline_cache.get_compute_pipeline(id).is_some() && *printed < 2 {
            info!("Info Prepare: clear_counts READY");
            *printed = 2;
        }
    }
}
pub fn prepare_histogram_pipeline(
    mut commands: Commands,
    pipeline_cache: Res<PipelineCache>,
    layout: Option<Res<GridHistogramBindGroupLayout>>,
    assets: Res<AssetServer>,
    mut cached: Local<Option<CachedComputePipelineId>>,
    mut printed: Local<u8>,
) {
    let Some(layout) = layout else {
        return;
    };

    if cached.is_none() {
        let shader: Handle<Shader> = assets.load("shaders/grid_build.wgsl");
        let desc = ComputePipelineDescriptor {
            label: Some("grid_histogram_pipeline".into()),
            layout: vec![layout.0.clone()],
            push_constant_ranges: vec![],
            shader_defs: vec![],
            entry_point: Cow::Borrowed("histogram"),
            shader,
            zero_initialize_workgroup_memory: true,
        };
        let id = pipeline_cache.queue_compute_pipeline(desc);
        *cached = Some(id);
        commands.insert_resource(HistogramPipeline(id));
        if *printed == 0 {
            info!("Info Prepare: histogram QUEUED");
            *printed = 1;
        }
        return;
    }

    if let Some(id) = *cached {
        if pipeline_cache.get_compute_pipeline(id).is_some() && *printed < 2 {
            info!("Info Prepare: histogram READY");
            *printed = 2;
        }
    }
}

impl Node for ClearCountsNode {
    fn update(&mut self, _world: &mut World) {}

    fn run(
        &self,
        _graph: &mut RenderGraphContext,
        render_context: &mut RenderContext,
        world: &World,
    ) -> Result<(), NodeRunError> {
        if world.get_resource::<ClearCountsPipeline>().is_none() {
            info!("Info Node: clear_counts SKIPPED (pipeline not ready)");
            return Ok(());
        }
        if world.get_resource::<GridBuildBindGroup>().is_none() {
            info!("Info Node: clear_counts SKIPPED (no grid-build bind group)");
            return Ok(());
        }
        if world.get_resource::<GridBuildParamsBuffer>().is_none() {
            info!("Info Node: clear_counts SKIPPED (no grid-build params)");
            return Ok(());
        }

        let pipeline_res = world.get_resource::<ClearCountsPipeline>().unwrap();
        let bind_group = world.get_resource::<GridBuildBindGroup>().unwrap();
        let gb = world.get_resource::<GridBuildParamsBuffer>().unwrap();

        if gb.value.num_cells == 0 {
            info!("Info Node: clear_counts SKIPPED (num_cells = 0)");
            return Ok(());
        }

        let pipeline_cache = world.resource::<PipelineCache>();
        let Some(pipeline) = pipeline_cache.get_compute_pipeline(pipeline_res.0) else {
            info!("Info Node: clear_counts SKIPPED (pipeline compiling)");
            return Ok(());
        };

        let groups = ((gb.value.num_cells + 255) / 256).max(1);
        info!(
            "Info Node: clear_counts DISPATCH, cells = {}, groups = {}",
            gb.value.num_cells, groups
        );

        let mut pass =
            render_context
                .command_encoder()
                .begin_compute_pass(&ComputePassDescriptor {
                    label: Some("ClearCountsPass"),
                    timestamp_writes: None,
                });

        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &bind_group.0, &[]);
        pass.dispatch_workgroups(groups, 1, 1);

        Ok(())
    }
}

pub fn add_clear_counts_node_to_graph(render_app: &mut bevy::app::SubApp) {
    let mut graph = render_app.world_mut().resource_mut::<RenderGraph>();
    graph.add_node(ClearCountsLabel, ClearCountsNode::default());

    let _ = graph.add_node_edge(ClearCountsLabel, DensityPassLabel);
}

impl Node for HistogramNode {
    fn update(&mut self, _world: &mut World) {}

    fn run(
        &self,
        _graph: &mut RenderGraphContext,
        render_context: &mut RenderContext,
        world: &World,
    ) -> Result<(), NodeRunError> {
        // === debugging style consistent with your Density node ===
        if world.get_resource::<HistogramPipeline>().is_none() {
            info!("Info Node: histogram SKIPPED (pipeline not ready)");
            return Ok(());
        }
        if world.get_resource::<GridHistogramBindGroup>().is_none() {
            info!("Info Node: histogram SKIPPED (no histogram bind group)");
            return Ok(());
        }
        if world.get_resource::<ExtractedParticleBuffer>().is_none() {
            info!("Info Node: histogram SKIPPED (no particle buffer)");
            return Ok(());
        }

        let pipeline_res = world.get_resource::<HistogramPipeline>().unwrap();
        let bind_group = world.get_resource::<GridHistogramBindGroup>().unwrap();
        let extracted = world.get_resource::<ExtractedParticleBuffer>().unwrap();

        let n = extracted.num_particles.max(1);
        let workgroups = (n + 255) / 256;

        let pipeline_cache = world.resource::<PipelineCache>();
        let Some(pipeline) = pipeline_cache.get_compute_pipeline(pipeline_res.0) else {
            info!("Info Node: histogram SKIPPED (pipeline compiling)");
            return Ok(());
        };

        info!(
            "Info Node: histogram DISPATCH, N = {}, groups = {}",
            n, workgroups
        );

        let mut pass =
            render_context
                .command_encoder()
                .begin_compute_pass(&ComputePassDescriptor {
                    label: Some("HistogramPass"),
                    timestamp_writes: None,
                });

        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &bind_group.0, &[]);
        pass.dispatch_workgroups(workgroups, 1, 1);

        Ok(())
    }
}

pub fn add_histogram_node_to_graph(render_app: &mut bevy::app::SubApp) {
    let mut graph = render_app.world_mut().resource_mut::<RenderGraph>();
    graph.add_node(HistogramPassLabel, HistogramNode::default());

    // Run order: ClearCounts -> Histogram -> Density

    let _ = graph.add_node_edge(ClearCountsLabel, HistogramPassLabel);
    let _ = graph.add_node_edge(HistogramPassLabel, DensityPassLabel);
}

pub fn _prepare_prefix_sum_naive_pipeline(
    mut commands: Commands,
    pipeline_cache: Res<PipelineCache>,
    layout: Option<Res<GridCountsToStartsBindGroupLayout>>,
    assets: Res<AssetServer>,
    mut cached: Local<Option<CachedComputePipelineId>>,
) {
    let Some(layout) = layout else {
        return;
    };
    if cached.is_some() {
        return;
    }

    let shader: Handle<Shader> = assets.load("shaders/grid_build.wgsl");
    let desc = ComputePipelineDescriptor {
        label: Some("grid_prefix_sum_naive_pipeline".into()),
        layout: vec![layout.0.clone()], // counts (ro), starts (rw)
        push_constant_ranges: vec![],
        shader_defs: vec![],
        entry_point: Cow::Borrowed("prefix_sum_naive"),
        shader,
        zero_initialize_workgroup_memory: true,
    };
    let id = pipeline_cache.queue_compute_pipeline(desc);
    *cached = Some(id);
    commands.insert_resource(PrefixSumNaivePipeline(id));
}

impl Node for PrefixSumNaiveNode {
    fn update(&mut self, _world: &mut World) {}

    fn run(
        &self,
        _graph: &mut RenderGraphContext,
        render_context: &mut RenderContext,
        world: &World,
    ) -> Result<(), NodeRunError> {
        if world.get_resource::<PrefixSumNaivePipeline>().is_none() {
            info!("Info Node: prefix_sum_naive SKIPPED (pipeline not ready)");
            return Ok(());
        }
        if world
            .get_resource::<GridCountsToStartsBindGroup>()
            .is_none()
        {
            info!("Info Node: prefix_sum_naive SKIPPED (no bind group)");
            return Ok(());
        }
        if world.get_resource::<GridBuildParamsBuffer>().is_none() {
            info!("Info Node: prefix_sum_naive SKIPPED (no params)");
            return Ok(());
        }

        let pip_id = world.get_resource::<PrefixSumNaivePipeline>().unwrap().0;
        let bg = &world
            .get_resource::<GridCountsToStartsBindGroup>()
            .unwrap()
            .0;
        let gb = &world.get_resource::<GridBuildParamsBuffer>().unwrap().value;

        if gb.num_cells == 0 {
            info!("Info Node: prefix_sum_naive SKIPPED (num_cells = 0)");
            return Ok(());
        }

        let cache = world.resource::<PipelineCache>();
        let Some(pipeline) = cache.get_compute_pipeline(pip_id) else {
            info!("Info Node: prefix_sum_naive SKIPPED (pipeline compiling)");
            return Ok(());
        };

        let groups = ((gb.num_cells + 255) / 256).max(1);
        info!(
            "Info Node: prefix_sum_naive DISPATCH, cells = {}, groups = {}",
            gb.num_cells, groups
        );

        let mut pass =
            render_context
                .command_encoder()
                .begin_compute_pass(&ComputePassDescriptor {
                    label: Some("PrefixSumNaivePass"),
                    timestamp_writes: None,
                });

        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, bg, &[]);
        pass.dispatch_workgroups(groups, 1, 1);

        Ok(())
    }
}
pub fn _add_prefix_sum_naive_node_to_graph(render_app: &mut bevy::app::SubApp) {
    let mut graph = render_app.world_mut().resource_mut::<RenderGraph>();
    graph.add_node(PrefixSumNaivePassLabel, PrefixSumNaiveNode::default());

    // Order: ClearCounts -> Histogram -> PrefixSumNaive -> Density

    let _ = graph.add_node_edge(ClearCountsLabel, PrefixSumNaivePassLabel);
    let _ = graph.add_node_edge(HistogramPassLabel, PrefixSumNaivePassLabel);
    let _ = graph.add_node_edge(PrefixSumNaivePassLabel, DensityPassLabel);
}
pub fn prepare_block_scan_pipeline(
    mut commands: Commands,
    pipeline_cache: Res<PipelineCache>,
    layout: Option<Res<GridBlockScanBindGroupLayout>>,
    assets: Res<AssetServer>,
    mut cached: Local<Option<CachedComputePipelineId>>,
) {
    let Some(layout) = layout else {
        return;
    };
    if cached.is_some() {
        return;
    }

    let shader: Handle<Shader> = assets.load("shaders/grid_build.wgsl");
    let desc = ComputePipelineDescriptor {
        label: Some("grid_block_scan_pipeline".into()),
        layout: vec![layout.0.clone()],
        push_constant_ranges: vec![],
        shader_defs: vec![],
        entry_point: Cow::Borrowed("block_scan"),
        shader,
        zero_initialize_workgroup_memory: true,
    };
    let id = pipeline_cache.queue_compute_pipeline(desc);
    *cached = Some(id);
    commands.insert_resource(BlockScanPipeline(id));
}

impl Node for BlockScanNode {
    fn update(&mut self, _world: &mut World) {}

    fn run(
        &self,
        _graph: &mut RenderGraphContext,
        render_context: &mut RenderContext,
        world: &World,
    ) -> Result<(), NodeRunError> {
        if world.get_resource::<BlockScanPipeline>().is_none() {
            info!("Info Node: block_scan SKIPPED (pipeline not ready)");
            return Ok(());
        }
        if world.get_resource::<GridBlockScanBindGroup>().is_none() {
            info!("Info Node: block_scan SKIPPED (no bind group)");
            return Ok(());
        }
        if world.get_resource::<GridBuildParamsBuffer>().is_none() {
            info!("Info Node: block_scan SKIPPED (no params)");
            return Ok(());
        }

        let pip_id = world.get_resource::<BlockScanPipeline>().unwrap().0;
        let bg = &world.get_resource::<GridBlockScanBindGroup>().unwrap().0;
        let gb = &world.get_resource::<GridBuildParamsBuffer>().unwrap().value;

        if gb.num_cells == 0 {
            info!("Info Node: block_scan SKIPPED (num_cells = 0)");
            return Ok(());
        }

        let cache = world.resource::<PipelineCache>();
        let Some(pipeline) = cache.get_compute_pipeline(pip_id) else {
            info!("Info Node: block_scan SKIPPED (pipeline compiling)");
            return Ok(());
        };

        // one workgroup per block of 256 cells
        let groups = ((gb.num_cells + 255) / 256).max(1);
        info!(
            "Info Node: block_scan DISPATCH, blocks = {}, cells = {}",
            groups, gb.num_cells
        );

        let mut pass =
            render_context
                .command_encoder()
                .begin_compute_pass(&ComputePassDescriptor {
                    label: Some("BlockScanPass"),
                    timestamp_writes: None,
                });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, bg, &[]);
        pass.dispatch_workgroups(groups, 1, 1);
        Ok(())
    }
}

pub fn add_block_scan_node_to_graph(render_app: &mut bevy::app::SubApp) {
    let mut graph = render_app.world_mut().resource_mut::<RenderGraph>();
    graph.add_node(BlockScanPassLabel, BlockScanNode::default());

    // Order: ClearCounts -> Histogram -> BlockScan -> PrefixSumNaive (or Density later)

    let _ = graph.add_node_edge(ClearCountsLabel, BlockScanPassLabel);
    let _ = graph.add_node_edge(HistogramPassLabel, BlockScanPassLabel);
    //let _ = graph.add_node_edge(BlockScanPassLabel, PrefixSumNaivePassLabel);
}

pub fn prepare_block_sums_scan_pipeline(
    mut commands: Commands,
    cache: Res<PipelineCache>,
    layout: Option<Res<BlockSumsScanBindGroupLayout>>,
    assets: Res<AssetServer>,
    mut cached: Local<Option<CachedComputePipelineId>>,
) {
    let Some(layout) = layout else {
        return;
    };
    if cached.is_some() {
        return;
    }

    let shader: Handle<Shader> = assets.load("shaders/grid_build.wgsl");
    let desc = ComputePipelineDescriptor {
        label: Some("grid_block_sums_scan_pipeline".into()),
        layout: vec![layout.0.clone()],
        push_constant_ranges: vec![],
        shader_defs: vec![],
        entry_point: Cow::Borrowed("block_sums_scan"),
        shader,
        zero_initialize_workgroup_memory: true,
    };
    let id = cache.queue_compute_pipeline(desc);
    *cached = Some(id);
    commands.insert_resource(BlockSumsScanPipeline(id));
}

impl Node for BlockSumsScanNode {
    fn update(&mut self, _world: &mut World) {}

    fn run(
        &self,
        _graph: &mut RenderGraphContext,
        render_context: &mut RenderContext,
        world: &World,
    ) -> Result<(), NodeRunError> {
        if world.get_resource::<BlockSumsScanPipeline>().is_none() {
            info!("Info Node: block_sums_scan SKIPPED (pipeline not ready)");
            return Ok(());
        }
        if world.get_resource::<BlockSumsScanBindGroup>().is_none() {
            info!("Info Node: block_sums_scan SKIPPED (no bind group)");
            return Ok(());
        }
        if world.get_resource::<GridBlockSumsBuffer>().is_none() {
            info!("Info Node: block_sums_scan SKIPPED (no block sums)");
            return Ok(());
        }

        let pip_id = world.get_resource::<BlockSumsScanPipeline>().unwrap().0;
        let bg = &world.get_resource::<BlockSumsScanBindGroup>().unwrap().0;
        let bs = world.get_resource::<GridBlockSumsBuffer>().unwrap();

        // derive workgroups from number of blocks
        let blocks = bs.num_blocks.max(1);
        let groups = ((blocks + 255) / 256).max(1);

        let cache = world.resource::<PipelineCache>();
        let Some(pipeline) = cache.get_compute_pipeline(pip_id) else {
            info!("Info Node: block_sums_scan SKIPPED (pipeline compiling)");
            return Ok(());
        };

        info!(
            "Info Node: block_sums_scan DISPATCH, blocks = {}, groups = {}",
            blocks, groups
        );

        let mut pass =
            render_context
                .command_encoder()
                .begin_compute_pass(&ComputePassDescriptor {
                    label: Some("BlockSumsScanPass"),
                    timestamp_writes: None,
                });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, bg, &[]);
        pass.dispatch_workgroups(groups, 1, 1);
        Ok(())
    }
}

pub fn add_block_sums_scan_node_to_graph(render_app: &mut bevy::app::SubApp) {
    let mut graph = render_app.world_mut().resource_mut::<RenderGraph>();
    graph.add_node(BlockSumsScanPassLabel, BlockSumsScanNode::default());

    // Order: ClearCounts -> Histogram -> BlockScan -> BlockSumsScan -> Density
    let _ = graph.add_node_edge(ClearCountsLabel, BlockSumsScanPassLabel);
    let _ = graph.add_node_edge(HistogramPassLabel, BlockSumsScanPassLabel);
    let _ = graph.add_node_edge(BlockScanPassLabel, BlockSumsScanPassLabel);
    let _ = graph.add_node_edge(BlockSumsScanPassLabel, DensityPassLabel);
}

pub fn prepare_add_back_pipeline(
    mut commands: Commands,
    cache: Res<PipelineCache>,
    layout: Option<Res<AddBackBindGroupLayout>>,
    assets: Res<AssetServer>,
    mut cached: Local<Option<CachedComputePipelineId>>,
) {
    let Some(layout) = layout else {
        return;
    };
    if cached.is_some() {
        return;
    }

    let shader: Handle<Shader> = assets.load("shaders/grid_build.wgsl");
    let desc = ComputePipelineDescriptor {
        label: Some("grid_add_back_pipeline".into()),
        layout: vec![layout.0.clone()],
        push_constant_ranges: vec![],
        shader_defs: vec![],
        entry_point: Cow::Borrowed("add_back_block_offsets"),
        shader,
        zero_initialize_workgroup_memory: true,
    };
    let id = cache.queue_compute_pipeline(desc);
    *cached = Some(id);
    commands.insert_resource(AddBackPipeline(id));
}

impl Node for AddBackNode {
    fn update(&mut self, _world: &mut World) {}

    fn run(
        &self,
        _graph: &mut RenderGraphContext,
        render_context: &mut RenderContext,
        world: &World,
    ) -> Result<(), NodeRunError> {
        if world.get_resource::<AddBackPipeline>().is_none() {
            info!("Info Node: add_back SKIPPED (pipeline not ready)");
            return Ok(());
        }
        if world.get_resource::<AddBackBindGroup>().is_none() {
            info!("Info Node: add_back SKIPPED (no bind group)");
            return Ok(());
        }
        if world.get_resource::<GridBuildParamsBuffer>().is_none() {
            info!("Info Node: add_back SKIPPED (no params)");
            return Ok(());
        }

        let pip_id = world.get_resource::<AddBackPipeline>().unwrap().0;
        let bg = &world.get_resource::<AddBackBindGroup>().unwrap().0;
        let gb = &world.get_resource::<GridBuildParamsBuffer>().unwrap().value;

        if gb.num_cells == 0 {
            info!("Info Node: add_back SKIPPED (num_cells = 0)");
            return Ok(());
        }

        let cache = world.resource::<PipelineCache>();
        let Some(pipeline) = cache.get_compute_pipeline(pip_id) else {
            info!("Info Node: add_back SKIPPED (pipeline compiling)");
            return Ok(());
        };

        let groups = ((gb.num_cells + 255) / 256).max(1);
        info!(
            "Info Node: add_back DISPATCH, cells = {}, groups = {}",
            gb.num_cells, groups
        );

        let mut pass =
            render_context
                .command_encoder()
                .begin_compute_pass(&ComputePassDescriptor {
                    label: Some("AddBackPass"),
                    timestamp_writes: None,
                });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, bg, &[]);
        pass.dispatch_workgroups(groups, 1, 1);

        Ok(())
    }
}

pub fn add_add_back_node_to_graph(render_app: &mut bevy::app::SubApp) {
    let mut graph = render_app.world_mut().resource_mut::<RenderGraph>();
    graph.add_node(AddBackPassLabel, AddBackNode::default());

    // Order: ClearCounts -> Histogram -> BlockScan -> BlockSumsScan -> AddBack -> Density
    let _ = graph.add_node_edge(ClearCountsLabel, AddBackPassLabel);
    let _ = graph.add_node_edge(HistogramPassLabel, AddBackPassLabel);
    let _ = graph.add_node_edge(BlockScanPassLabel, AddBackPassLabel);
    let _ = graph.add_node_edge(BlockSumsScanPassLabel, AddBackPassLabel);
    let _ = graph.add_node_edge(AddBackPassLabel, DensityPassLabel);
}
