# bevy_gpu_fluid

> **GPU-accelerated Smoothed-Particle Hydrodynamics for Bevy 0.16**

> _Early work-in-progress – APIs will change until v0.1_

> _Currently a CPU prototype - GPU implementation in progress_

## Overview
This protype implements SPH (Smoothed-Particle Hydrodynamics) in Bevy.
The long term goal is to providfe a fully GPU-accelarated and rendered fluid dynamics.
The 2D CPU-based prototype is available for testing and algorithm validation.

![Demo 1](docs/sprint2/demo_scene.gif)

| ![Demo_SolidColor](docs/sprint2/solid_color.png) | ![Demo_DensityColor](docs/sprint2/density_map.png) |
|------------------------------------------|------------------------------------------|
| Solid Color                              | Density Color Mapping                    |

## Features (CPU prototype)
- 2D SPH simulation using Poly6, Spiky and Viscosity kernels normalized for 2D
- Interactive mouse-driven fluid manipulation
- Switching between visualization modes is possible:
  - Solid color
  - Density color mapping   
- The parameters are not hardcoded and can be changed easily:
  - smoothing length
  - stiffness
  - viscosity
  - mass
  - etc.

## Controls & Demos

| Action | Key / Mouse | Demo |
|--------|-------------|------|
| Switch view mode | `Space` | ![Demo_Toggle](docs/sprint2/toggle_demo.gif) |
| Click + drag to disturb fluid | Left mouse button | ![Demo_Mouse](docs/sprint2/mouse_drag_example.gif) |


## Quick start
```bash
git clone https://github.com/ArminGEtemad/bevy_gpu_fluid.git
cd bevy_gpu_fluid
cargo run --release --example sph2d_cpu_demo  # demo scene with density and solid color view
```

