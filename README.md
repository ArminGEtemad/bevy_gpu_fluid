# bevy_gpu_fluid

> **GPU-accelerated Smoothed-Particle Hydrodynamics for Bevy 0.16**  
> _Early work-in-progress – APIs will change until v0.1_
> _Currently a CPU prototype_ - GPU implementation in progress

## Overview
This protype implements SPH (Smoothed-Particle Hydrodynamics) in Bevy.
The long term goal is to providfe a fully GPU-accelarated and rendered fluid dynamics.
The 2D CPU-based prototype is available for testing and algorithm validation.

## Features (CPU ptototype)
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

## Controls
| Action                   | Key / Mouse |
|--------------------------|-------------|
| Switch view mode         | `Space`     |
| Click + drag to disturb fluid | Left mouse button |


## Quick start
```bash
git clone https://github.com/ArminGEtemad/bevy_gpu_fluid.git
cd bevy_gpu_fluid
cargo run --release --example sph2d_cpu_demo  # demo scene with density and solid color view
```

