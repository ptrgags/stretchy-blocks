struct VertexInput {
    @builtin(instance_index) instance_index: u32,
    @location(0) position: vec3f,
    @location(1) normal: vec3f,
}

struct VertexOutput {
    @builtin(position) position: vec4f,
    @location(0) normal: vec3f,
    @location(1) uvw: vec3f,
    @location(2) @interpolate(flat) grid_coords: vec3u,
}

const DIMENSIONS = vec3u(10, 10, 10);
const DEG_45 = radians(45);
const ROTATE_45_Y = mat3x3f(
    cos(DEG_45), 0, -sin(DEG_45),
    0, 1, 0,
    sin(DEG_45), 0, cos(DEG_45),
);

const MAGIC_ANGLE = 0.615479709; // atan(1/sqrt(2))
const TILT = mat3x3f(
    1, 0, 0,
    0, cos(MAGIC_ANGLE), sin(MAGIC_ANGLE),
    0, -sin(MAGIC_ANGLE), cos(MAGIC_ANGLE),
);


fn get_grid_coords(index: u32) -> vec3u {
    let intermediate = index / DIMENSIONS.x;
    let x = index % DIMENSIONS.x;
    let y = intermediate % DIMENSIONS.y;
    let z = intermediate / DIMENSIONS.y;

    return vec3u(x, y, z);
}

@vertex
fn vertex_main(input: VertexInput) -> VertexOutput {
    // Position the cube instance. Each cube will be 1 unit wide, and
    // we want to center this based on the overall bounding box.
    let grid_coords = get_grid_coords(input.instance_index);
    let position_model = vec3f(grid_coords) + input.position - 0.5 * vec3f(DIMENSIONS);

    let rotated = TILT * ROTATE_45_Y * position_model;

    // The blocks fit within [-5, 5]^3.
    // Our view frustum will be orthographic, but the canvas size has a 5:7 aspect ratio
    // so let's pretend we're in a box that's 5:5:7 in aspect ratio.
    const VIEW_RADII = 2.0 * vec3f(5.0, 7.0, 5.0);
    var position_clip = rotated / VIEW_RADII;

    var output: VertexOutput;
    output.position = vec4f(position_clip, 1.0);
    output.normal = input.normal;
    output.grid_coords = grid_coords;

    // Position also serves as uvw coordinates!
    output.uvw = input.position;
    return output;
}

@fragment
fn fragment_main(input: VertexOutput) -> @location(0) vec4f {

    let shading = f32(input.grid_coords.z) / f32(DIMENSIONS.z);
    // mango cube
    return vec4f(shading * input.uvw, 1.0);
}