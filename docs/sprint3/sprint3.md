Sprint 3, the GPU fundamentals, was the most complex sprint until now. Mostly because it is my first time writing GPU pipelines. 

# Goals

The high-level goal of this sprint was using WGSL compute shaders to simulate fluid behavior. So we had to replace the CPU-based calculation loop with GPU and use the results for Bevy to render the simulation.
- CPU generates and/or updates particle state (position, velocity, etc.)
- Send this data to GPU
- loads the correct WGSL shader
- Bind the buffers and dispatch GPU threads
- Each GPU thread does math
- Store results in output buffers
- Readback and use results for rendering

**Buffers** are data lifeline between simulation logic and the compute shader
 - creates GPU-side storage buffers
 - Upload CPU data into those buffers
 - Prepare Uniform buffers for constants
   - Uniform buffers are read-only buffers that store constant data
   - `var<uniform>` is read only and shared across all threads
   - `var<storage>` has read and write access

**Pipelines** are GPU work that is being scheduled and executed per frame.

## I Foreign Function Interface
Or ffi is where we build CPU to GPU bridge. I am not going to layout everything written in `ffi.rs` at once. Step by step whenever it is needed.

```rust
#[repr(C)] 
#[derive(Clone, Copy, Debug, Pod, Zeroable)] 
pub struct GPUParticle {  
    pub pos: [f32; 2], 
    pub vel: [f32; 2], 
    pub acc: [f32; 2], 
    pub rho: f32, 
    pub p: f32, 
}
```
GPU buffers are just raw memory. Meaning WGSL expects structs to be packed in a predictable way and `#[repr(C)]` guarantees that the struct matches C-style layout leading to safe data exchange between Rust and WGSL.
The reason why I didn't use `glam::Vec2` is that it is not `Pod` and GPU doesn't understand Rust types. 
 

## Buffers
The purpose is, as mentioned, to creat and manage GPU memory for simulation data
### Startup
These funtions are called once at start by Bevy
```rust
#[derive(Resource)]
pub struct ParticleBuffers {
    pub particle_buffer: Buffer,
    pub num_particles: u32,
}

fn init_gpu_buffers(mut commands: Commands, render_device: Res<RenderDevice>, sph: Res<SPHState>) {
    let particle_buffers = ParticleBuffers::new(&render_device, &sph);
    commands.insert_resource(particle_buffers);
}

impl ParticleBuffers {
    pub fn new(render_device: &RenderDevice, sph: &SPHState) -> Self {
        // converting the cpu particle to gpu
        let mut gpu_particles = Vec::with_capacity(sph.particles.len());
        for particle in &sph.particles {
            gpu_particles.push(GPUParticle {
                pos: [particle.pos.x, particle.pos.y],
                vel: [particle.vel.x, particle.vel.y],
                acc: [particle.acc.x, particle.acc.y],
                rho: particle.rho,
                p: particle.p,
            });
        }

        // storage buffer with the init data
        let particle_buffer = render_device.create_buffer_with_data(&BufferInitDescriptor {
            label: Some("Particle Buffer"),
            contents: bytemuck::cast_slice(&gpu_particles),
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
        });

        Self {
            particle_buffer,
            num_particles: gpu_particles.len() as u32,
        }
    }
}
```
`ParticleBuffers`
`init_gpu_buffers` allocates GPU buffer for the particles
`Commands` is how we insert resources. 
`RenderDevive` is responsible for the creation of most rendering and compute resources.
`SPHState` contains the simulation state.

The implmentation `ParticleBuffers` is here to create a GPU buffer that holds all SPU partocles ready to be used by compute shaders.

First of all, a new vector in CPU memory is being created. However `sph.particles` has Rust only type data such as `glam`. That is why I made a for-loopto push every particle state into a GPU-safe format. 

Then I create a storage buffer labeled `Particle Buffer` with raw bytes using `bytemuck`. Such buffer can be later accessed by WGSL like
```wgsl
var<storage, read_write> particles: array<GPUParticle>;
```
At the end, a new ParticleBuffer is returned that stores the GPU buffer itself and the numbers of the particles. 

Next step is to tell GPU what the buffers are that we want to send and what it will do with them!
- what buffer will be available
- at which binding point
- whether they are read-only, read-write or uniform

The layout will be used to late actually build buffer bindings and the compute pipeline will rely on this layout.
```rust
fn init_particle_bind_group_layout(mut commands: Commands, render_device: Res<RenderDevice>) {
    let layout = render_device.create_bind_group_layout(
        Some("particle_bind_group_layout"),
        &[
            // binding 0: particles (read_write)
            BindGroupLayoutEntry {
                binding: 0,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            // binding 1: cell_starts (read-only)
            BindGroupLayoutEntry {
                binding: 1,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            // binding 2: cell_entries (read-only)
            BindGroupLayoutEntry {
                binding: 2,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            // binding 3: grid params (uniform)
            BindGroupLayoutEntry {
                binding: 3,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            // binding 4: integrate params (uniform)
            BindGroupLayoutEntry {
                binding: 4,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
    );
    commands.insert_resource(ParticleBindGroupLayout(layout));
}
```
meaning that we are already at the start of the program telling GPU that I wanna use a bind group with 5 bindings each one is...
1. for particle_buffer `binding: 0` and `read_only: false` means the shader can write to it. In WGSL, we access it `@binding(0)`
2. for cell_starts `binding: 1` and it is read-only since it 
3. cell entries 
and so on...

Now we define a function that is not needed for the program to run but it is 100% needed for us to debug and that the readback!

```rust
#[derive(Resource)]
pub struct ReadbackBuffer {
    pub buffer: Buffer,
    pub size_bytes: u64,
}

fn init_readback_buffer(
    mut commands: Commands,
    render_device: Res<RenderDevice>,
    particle_buffers: Option<Res<ParticleBuffers>>, // solving the panic problem
) {
    let Some(particle_buffers) = particle_buffers else {
        return;
    };
    let size_bytes =
        (particle_buffers.num_particles as u64) * (std::mem::size_of::<GPUParticle>() as u64);
    let buffer = render_device.create_buffer(&BufferDescriptor {
        label: Some("readback_buffer"),
        size: size_bytes,
        usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    commands.insert_resource(ReadbackBuffer { buffer, size_bytes });
}
```
We are creating a special GPU buffer that is used to read the result back onto the CPU.
At the beginning the let-else statement is written to make sure of the safety of the program and avoid panic if `ParticleBuffer` doesn't exist when the system is running this.
Then, we calculate the exact number of bytes `size_bytes` needed to store all the particles. This much memory will be allocated for the readback in GPU.
Finally, we create the actual buffer.
How exactly does this system work?
1. GPU runs and writes data to particle buffer
2. we copy from particle buffer to readback buffer, hence `COPY_DST`
3. we, then, map from readback and read the data on CPU, hence `MAP_READ`

The `mapped_at_creation: false,` flag means that we don't map it yet. we wait until we are ready!

In the next step, we make the buffer for the spatial structure but just quickly 

---
### II Foreign Function Interface
Just like the other ffi part of the code we need alignment with WGSL. 
```rust
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct GridParams {
    pub min_world: [f32; 2],
    pub _pad0: f32,
    pub dims: [u32; 2],
    pub _pad1: [u32; 2], 
}
```
---

back to buffers

```rust
#[derive(Resource)]
pub struct GridBuffers {
    pub params_buf: Buffer,  // UNIFORM
    pub starts_buf: Buffer,  // STORAGE
    pub entries_buf: Buffer, // STORAGE
    pub num_cells: usize,
    pub num_particles: usize,
}

pub fn init_grid_buffers(
    mut commands: Commands,
    render_device: Res<RenderDevice>,
    sph: Res<SPHState>,
) {
    commands.insert_resource(GridBuffers::new(&render_device, &sph));
}

fn cell_ix(pos: Vec2, h: f32) -> IVec2 {
    (pos / h).floor().as_ivec2()
}

fn build_compressed_grid(sph: &SPHState) -> (GridParams, Vec<u32>, Vec<u32>) {
    let h = sph.h;

    let mut min_c = IVec2::new(i32::MAX, i32::MAX);
    let mut max_c = IVec2::new(i32::MIN, i32::MIN);
    for p in &sph.particles {
        let c = cell_ix(p.pos, h);
        min_c = IVec2::new(min_c.x.min(c.x), min_c.y.min(c.y));
        max_c = IVec2::new(max_c.x.max(c.x), max_c.y.max(c.y));
    }
    let dims = IVec2::new(max_c.x - min_c.x + 1, max_c.y - min_c.y + 1);
    let nx = dims.x.max(1) as usize;
    let ny = dims.y.max(1) as usize;
    let num_cells = nx * ny;
    let n = sph.particles.len();

    let mut counts = vec![0u32; num_cells];
    for (_i, p) in sph.particles.iter().enumerate() {
        let c = cell_ix(p.pos, h);
        let ix = (c.x - min_c.x) as usize;
        let iy = (c.y - min_c.y) as usize;
        let id = ix + iy * nx;
        debug_assert!(id < num_cells);
        counts[id] += 1;
    }

    let mut starts = vec![0u32; num_cells + 1];
    for i in 0..num_cells {
        starts[i + 1] = starts[i] + counts[i];
    }

    let mut offsets = starts.clone();
    let mut entries = vec![0u32; n];
    for (pi, p) in sph.particles.iter().enumerate() {
        let c = cell_ix(p.pos, h);
        let ix = (c.x - min_c.x) as usize;
        let iy = (c.y - min_c.y) as usize;
        let id = ix + iy * nx;
        let dst = &mut offsets[id];
        let idx = *dst as usize;
        entries[idx] = pi as u32;
        *dst += 1;
    }

    let params = GridParams {
        min_world: [min_c.x as f32 * h, min_c.y as f32 * h],
        cell_size: h,
        _pad0: 0.0,
        dims: [nx as u32, ny as u32],
        _pad1: [0, 0],
    };

    (params, starts, entries)
}

impl GridBuffers {
    pub fn new(render_device: &RenderDevice, sph: &SPHState) -> Self {
        let (params, starts, entries) = build_compressed_grid(sph);

        let params_buf = render_device.create_buffer_with_data(&BufferInitDescriptor {
            label: Some("Grid Params"),
            contents: bytemuck::bytes_of(&params),
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        });

        let starts_buf = render_device.create_buffer_with_data(&BufferInitDescriptor {
            label: Some("Grid Starts"),
            contents: bytemuck::cast_slice(&starts),
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
        });

        let entries_buf = render_device.create_buffer_with_data(&BufferInitDescriptor {
            label: Some("Grid entries"),
            contents: bytemuck::cast_slice(&entries),
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
        });

        info!(
            "Grid Init: cells={} ({}x{}), starts.len={}, entries.len={}",
            (params.dims[0] as usize) * (params.dims[1] as usize),
            params.dims[0],
            params.dims[1],
            starts.len(),
            entries.len()
        );

        Self {
            params_buf,
            starts_buf,
            entries_buf,
            num_cells: starts.len() - 1,
            num_particles: entries.len(),
        }
    }

    pub fn update(&mut self, render_device: &RenderDevice, queue: &RenderQueue, sph: &SPHState) {
        let (params, starts, entries) = build_compressed_grid(sph);

        let new_num_cells = starts.len() - 1;
        let new_num_particles = entries.len();

        if new_num_cells != self.num_cells {
            self.starts_buf = render_device.create_buffer_with_data(&BufferInitDescriptor {
                label: Some("Grid Starts"),
                contents: bytemuck::cast_slice(&starts),
                usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
            });
            self.num_cells = new_num_cells;
        } else {
            queue.write_buffer(&self.starts_buf, 0, bytemuck::cast_slice(&starts));
        }

        if new_num_particles != self.num_particles {
            self.entries_buf = render_device.create_buffer_with_data(&BufferInitDescriptor {
                label: Some("Grid Entries"),
                contents: bytemuck::cast_slice(&entries),
                usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
            });
            self.num_particles = new_num_particles;
        } else {
            queue.write_buffer(&self.entries_buf, 0, bytemuck::cast_slice(&entries));
        }

        queue.write_buffer(&self.params_buf, 0, bytemuck::bytes_of(&params));

        let nx = params.dims[0];
        let ny = params.dims[1];

        info!(
            "grid update: dims=({}x{}), cells={}, entries={}, starts[0..5]={:?}",
            nx,
            ny,
            self.num_cells,
            self.num_particles,
            &starts[0..starts.len().min(5)]
        );
    }
}
```
In itself, it has three buffers. The function reads the CPU side and allocates and sets up the memory of correct size. Before going to the implementation we need two helper funcitons `cell_ix` and `build_compressed_grid`. `cell_ix` takes the position and the smoothing length and returns a 2D interger ID of the grid cell that the particle belongs to. So if a particle is at position $(2.2, 3.8)$ and $h=1.0$ then this particle must live in the cell $(2, 3)$.

Then `build_compressed_grid` function converts a list of particles into `GridParams` and two vectors. Then we determine the spatial bounding. How? we calculate the smallest and largest cell index that contains any particle. 

Then we compute the grid dimensions and total number of cells. We want to have a grid of cells that starts at `min_c` (top-left-most cell with any particle) and ends with `max_c` (bottom-right-most cell with any particle in it) meaning if $c_{min}=(3, 7)$ and $c_{max}=(6, 9)$ then the needed $x$ axis is $3, 4, 5, 6$ and $y$ axis will be $7, 8, 9$ meaning 4 cells in x direction and 3 cells in y direction $6 - 3 + 1$ and $9 - 7 + 1$. It is then trivial that the number of cells are given by multiplying the number of cells in x direction to y direction.

Now, it is time for some memory work. First of all, we have to count how many particles are in each grid cell. We assign each particle to a grid cell indexed by an integer. Meaning 2D cell coordinates $(ix, iy)$ are converted to a 1D array $id$. At the end we see how many particles are in the same cell. We get somethin like `counts = [3, 1, 2, 3, 5, ...]` meaning that there are 3 particles in cell with index 0 and so on. 

The start array runs like
`starts[0] = 0`
`starts[1] = starts[0] + counts[0] = 0 + 3 = 3` (I use the already mentioned example for `counts`)
`starts[2] = starts[1] + counts[1] = 3 + 1 = 4`
`starts[3] = starts[2] + counts[2] = 4 + 2 = 6`
We need this array because in the next step we want to put all particles into one big flat array called `entries`.

| Cell | `starts[i]` | `starts[i+1]` | Entries range                 |
| ---- | ----------- | ------------- | ----------------------------- |
| 0    | 0           | 3             | `entries[0..3]` → 3 particles |
| 1    | 3           | 4             | `entries[3..4]` → 1 particle  |
| 2    | 4           | 6             | `entries[4..6]` → 2 particle  |

At the end the parameters of `GridParams` are found.
Now we can look at `impl GridBuffers`
First we define the three buffers and we log it for debugging. Its update function is called every frame. to refresh the buffer with the latest data. 
1. it rebuilds the grid from CPU
2. checks for size changes
3. updates all the buffers

It logs again.

The last pieces of startups are incluing the ffi

---
### III Foreign Function Interface
```rust
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct IntegrateParams {
    pub dt: f32,
    pub x_min: f32,
    pub x_max: f32,
    pub bounce: f32, // mast be negative
}
```
---

```rust
#[derive(Resource)]
pub struct IntegrateParamsBuffer {
    pub buffer: Buffer,
}
fn init_integrate_params_buffer(
    mut commands: Commands,
    render_device: Res<RenderDevice>,
    config: Res<IntegrateConfig>,
) {
    let params = IntegrateParams {
        dt: config.dt,
        x_min: config.x_min,
        x_max: config.x_max,
        bounce: config.bounce,
    };
    let buffer = render_device.create_buffer_with_data(&BufferInitDescriptor {
        label: Some("integrate_params_uniform"),
        contents: bytemuck::bytes_of(&params),
        usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
    });
    commands.insert_resource(IntegrateParamsBuffer { buffer });
}

fn init_use_gpu_integration(mut commands: Commands) {
    commands.insert_resource(UseGpuIntegration(true)); 
}
```

this code initializes the buffer needed for intergration. builds the shaders interface (the contract between CPU and GPU!)

### Update
Update systems run every frame starting with 

```rust
fn queue_particle_buffer(
    sph: Res<SPHState>,
    particle_buffers: Option<Res<ParticleBuffers>>,
    render_queue: Res<RenderQueue>,
    use_gpu_integration: Res<UseGpuIntegration>,
) {
    let Some(particle_buffers) = particle_buffers else {
        return;
    };
    if use_gpu_integration.0 {
        return;
    }
    let mut gpu_particles = Vec::with_capacity(sph.particles.len());
    for particle in &sph.particles {
        gpu_particles.push(GPUParticle {
            pos: [particle.pos.x, particle.pos.y],
            vel: [particle.vel.x, particle.vel.y],
            acc: [particle.acc.x, particle.acc.y],
            rho: particle.rho,
            p: particle.p,
        });
    }

    render_queue.write_buffer(
        &particle_buffers.particle_buffer,
        0,
        bytemuck::cast_slice(&gpu_particles),
    );
}
```
I wrote this function back when I wanted to compare GPU and CPU using the readback. At the end of sprint 3, GPU itself evolves the particles and no CPU overwrite is needed. So this function is not used anymore. It converted `Vec<Particle>` to `Vec<GPUParticle>`. Then just send the particle data to the GPU. 

Next is 
```rust
#[derive(Resource, Clone, Copy, Debug)]
pub struct IntegrateConfig {
    pub dt: f32,
    pub x_min: f32,
    pub x_max: f32,
    pub bounce: f32,
}

impl Default for IntegrateConfig {
    fn default() -> Self {
        Self {
            dt: 0.0005,
            x_min: -5.0,
            x_max: 3.0,
            bounce: -3.0,
        }
    }
}
fn update_integrate_params_buffer(
    render_queue: Res<RenderQueue>,
    ub: Res<IntegrateParamsBuffer>,
    config: Res<IntegrateConfig>,
) {
    let params = IntegrateParams {
        dt: config.dt,
        x_min: config.x_min,
        x_max: config.x_max,
        bounce: config.bounce,
    };
    render_queue.write_buffer(&ub.buffer, 0, bytemuck::bytes_of(&params));
}
```
with which we upload the updated integration parameters for the GPU every frame. I did it every frame because I wasn't sure if I want to stay with a static simulation or if I want to make the wall oscillate in the future. So I thought it just makes it more flexible for the future. Also it was a fast way of getting rid of hardcoded values in the earlier version of the code. 

the last system in Update is 
```rust
pub fn update_grid_buffers(
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
    sph: Res<SPHState>,
    mut grid: ResMut<GridBuffers>,
) {
    grid.update(&render_device, &render_queue, &sph);
}
```
which calls the method we already discussed `GridBuffer::update()`. In every frame, we are giving GPU the latest particle map and needed information for further calculations. 

## Extractions
Main world and render world do not live together. Render world does not have access to the main world resources directly. 
> `ExtracSchedule` extracts data from the main world and inserts it into the render world.
> This step should be kept as short as possible to increase the “pipelining potential” for running the next frame while rendering the current frame.

I have this information from [here](https://docs.rs/bevy/latest/bevy/render/struct.ExtractSchedule.html)

First extraction is 
```rust
// Rendering world Copy
#[derive(Resource, Clone, ExtractResource)]
pub struct ExtractedParticleBuffer {
    pub buffer: Buffer,
    pub num_particles: u32,
}

fn extract_particle_buffer(
    mut commands: Commands,
    particle_buffers: Extract<Res<ParticleBuffers>>,
) {
    commands.insert_resource(ExtractedParticleBuffer {
        buffer: particle_buffers.particle_buffer.clone(),
        num_particles: particle_buffers.num_particles,
    });
}
```
Main job of this function is to bring over the `ParticleBUffers` from the main app into the render app (like one way road being `Extract<Res<ParticleBuffers>>`). 

Then 
```rust
#[derive(Resource, Clone)]
pub struct ParticleBindGroupLayout(pub BindGroupLayout);

fn extract_bind_group_layout(
    mut commands: Commands,
    layout: Extract<Res<ParticleBindGroupLayout>>,
) {
    commands.insert_resource(ParticleBindGroupLayout(layout.0.clone()));
}
```
Remember that we made a bung group layout. However, the main world only knew about the contract and GPU wasn't aware. Now, with this function we exract it to the render world. 

The next extraction is that of readback. GPU cannot just write to CPU memory. So we make:
```rust
#[derive(Resource, Clone, ExtractResource)]
pub struct ExtractedReadbackBuffer {
    pub buffer: Buffer,
    pub size_bytes: u64,
}
fn extract_readback_buffer(mut commands: Commands, readback: Extract<Res<ReadbackBuffer>>) {
    commands.insert_resource(ExtractedReadbackBuffer {
        buffer: readback.buffer.clone(),
        size_bytes: readback.size_bytes,
    });
}
```
I am not sure explaining all of the extraction functions is necessary since they have all only one job and that is to copy recources from the main world into the render world without changing them.

## Prepare Systems: Render
Starting with 
```rust
fn prepare_particle_bind_group(
    mut commands: Commands,
    render_device: Res<RenderDevice>,
    layout: Res<ParticleBindGroupLayout>,
    extracted: Res<ExtractedParticleBuffer>,
    grid: Res<ExtractedGrid>,
    integ: Res<ExtractedIntegrateParamsBuffer>,
) {
    let bind_group = render_device.create_bind_group(
        Some("particle_bind_group"),
        &layout.0,
        &[
            BindGroupEntry {
                binding: 0,
                resource: extracted.buffer.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 1,
                resource: grid.starts_buf.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 2,
                resource: grid.entries_buf.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 3,
                resource: grid.params_buf.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 4,
                resource: integ.buffer.as_entire_binding(),
            },
        ],
    );
    commands.insert_resource(ParticleBindGroup(bind_group));
    info!("particle_bind_group is READY");
}

```
All the inputs are GPU-side resources, already extracted from the main world. Here, we pass the GPU buffers into a compute shader in the exact layout it expects. Remember `BindGroupEntry`. For anything that follows we need to look at the pipelines first!

### Pipelines

Density Pipeline: 
```rust
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
        return;
    }

    if let Some(id) = *pipeline_id {
        if let Some(pipeline) = pipeline_cache.get_compute_pipeline(id) {
            info!("density_pipe_line is READY");

            commands.insert_resource(DensityPipeline(pipeline.clone()));
        }
    }
}
```
The cache is the central store for all WGPU pipeline comilation results. The Compilation starts from the `assets` folder.  We queue the pipeline build and save the ID for later. 
Once the pipeline is compiled, we insert it. This is the same with every pipeline.

### Node
I named it `DensityNode` but the reality is that it is the node for everything. I have to change it in the future. 
```rust
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

        let n = extracted.num_particles.max(1);
        let workgroups = (n + 255) / 256;
        info!("Info Node: DISPATCH, N = {}, groups = {}", n, workgroups);

        let mut pass = render_context
            .command_encoder()
            .begin_compute_pass(&ComputePassDescriptor::default());

        pass.set_pipeline(&pipeline.0); 
        pass.set_bind_group(0, &bind_group.0, &[]);
        pass.dispatch_workgroups(workgroups, 1, 1);

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

        drop(pass);
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
```
This is the actual GPU execution pipeline. 
1. waits until all pipelines are compiled and bind groups are ready
2. starts a compute pass and dispatches the SPH pipeline stages:
   - Density
   - Pressure
   - Forces
   - Integration
3. if allowed copies the result buffer back into a CPU readable buffer `readback` for debugging.

Here, I made `n` to be at least one even if there are no particles since dispatching 0 workgroups can crash the system.
This whole phase works as following:

| Phase       | System                         | Description |
|-------------|--------------------------------|-------------|
| Extract     | `extract_particle_buffer`      | Clones GPU buffer to Render World |
| Prepare     | `prepare_density_pipeline`     | Queues WGSL compute shader        |
| RenderGraph | `DensityNode::run`             | Sets pipeline, binds, dispatches  |
| Dispatch    | `dispatch_workgroups`          | Launches compute threads          |



