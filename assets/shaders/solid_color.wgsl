@group(1) @binding(0) var<uniform> color : vec4<f32>;
// group 1 led to Handling wgpu error as fatal by defaul

@fragment 
fn fragment() -> @location(0) vec4<f32> {
    return color;
}
