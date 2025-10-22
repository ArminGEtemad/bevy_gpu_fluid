use bevy::prelude::*;
use bevy::render::render_graph::{NodeRunError, RenderGraphContext, RenderLabel, ViewNode};
use bevy::render::renderer::RenderContext;
use bevy::render::view::ViewTarget;

use crate::gpu::buffers::ExtractedParticleBuffer;
use crate::gpu::draw_buffers::{DrawBindGroup, QuadVertexBuffer};
use crate::gpu::draw_pipeline::DrawPipeline;

#[derive(Debug, Hash, PartialEq, Eq, Clone, RenderLabel)]
pub struct ParticlesDrawPassLabel;

#[derive(Default)]
pub struct ParticlesDrawNode;

impl ViewNode for ParticlesDrawNode {
    // 0.16.1: ViewNode runs *per view*; fetch the camera's ViewTarget directly
    type ViewQuery = (&'static ViewTarget,);

    fn run(
        &self,
        _graph: &mut RenderGraphContext,
        rcx: &mut RenderContext,
        (view_target,): <Self::ViewQuery as bevy::ecs::query::QueryData>::Item<'_>,
        world: &World,
    ) -> Result<(), NodeRunError> {
        // Pipeline (from PipelineCache)
        let Some(dp) = world.get_resource::<DrawPipeline>() else {
            return Ok(());
        };
        let cache = world.resource::<bevy::render::render_resource::PipelineCache>();
        let Some(pipeline) = cache.get_render_pipeline(dp.0) else {
            return Ok(());
        };

        // Bind group, quad VB, and instance count (number of particles)
        let Some(bg) = world.get_resource::<DrawBindGroup>() else {
            return Ok(());
        };
        let Some(vb) = world.get_resource::<QuadVertexBuffer>() else {
            return Ok(());
        };
        let Some(particles) = world.get_resource::<ExtractedParticleBuffer>() else {
            return Ok(());
        };
        if particles.num_particles == 0 {
            return Ok(());
        }
        info!(
            "ParticlesDrawPass: drawing {} instances",
            particles.num_particles
        );

        let mut pass =
            rcx.begin_tracked_render_pass(bevy::render::render_resource::RenderPassDescriptor {
                label: Some("ParticlesDrawPass"),
                color_attachments: &[Some(view_target.get_color_attachment())],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            }); // uses the correct load/store ops for this view

        pass.set_render_pipeline(pipeline);
        pass.set_bind_group(0, &bg.0, &[]);
        pass.set_vertex_buffer(0, vb.buffer.slice(..));
        //pass.draw(0..6, 0..1);
        pass.draw(0..6, 0..particles.num_particles);
        Ok(())
    }
}
