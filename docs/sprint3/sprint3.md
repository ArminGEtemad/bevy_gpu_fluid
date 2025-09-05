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

In the next step, we make the buffer for the spatial structure
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



