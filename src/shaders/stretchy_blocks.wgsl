struct VertexInput {
    @location(0) position: vec3f,
    @location(1) normal: vec3f,
}

struct VertexOutput {
    @builtin(position) position: vec4f,
    @location(0) normal: vec3f,
    @location(1) uvw: vec3f,
}

@vertex
fn vertex_main(input: VertexInput) -> VertexOutput {
    // the cube was defined as [0, 1]^3, so shift it so it's [-0.5, 0.5]^3,
    // i.e. it partially fills clip space
    let position_clip = input.position - 0.5;

    var output: VertexOutput;
    output.position = vec4f(position_clip, 1.0);
    output.normal = input.normal;

    // Position also serves as uvw coordinates!
    output.uvw = input.position;
    return output;
}

@fragment
fn fragment_main(input: VertexOutput) -> @location(0) vec4f {
    // mango cube
    return vec4f(input.uvw, 1.0);
}