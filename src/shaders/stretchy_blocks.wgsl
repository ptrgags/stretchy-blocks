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

const MAGIC_ANGLE = 0.615479709; // atan(1/sqrt(2))
const TILT = mat3x3f(
    1, 0, 0,
    0, cos(MAGIC_ANGLE), sin(MAGIC_ANGLE),
    0, -sin(MAGIC_ANGLE), cos(MAGIC_ANGLE),
);

fn rotate_y(angle: f32) -> mat3x3f {
    let c = cos(angle);
    let s = sin(angle);
    return mat3x3f(
        c, 0, -s,
        0, 1, 0,
        s, 0, c,
    );
}


fn get_grid_coords(index: u32) -> vec3u {
    let intermediate = index / DIMENSIONS.x;
    let x = index % DIMENSIONS.x;
    let y = intermediate % DIMENSIONS.y;
    let z = intermediate / DIMENSIONS.y;

    return vec3u(x, y, z);
}

@group(0) @binding(0) var<uniform> time: f32;

@vertex
fn vertex_main(input: VertexInput) -> VertexOutput {
    // Position the cube instance. Each cube will be 1 unit wide, and
    // we want to center this based on the overall bounding box.
    let grid_coords = get_grid_coords(input.instance_index);
    let position_model = vec3f(grid_coords) + input.position - 0.5 * vec3f(DIMENSIONS);

    let rotated = TILT * rotate_y(time) * position_model;

    // Before rotation, the blocks fit within [-5, 5]^3.
    // After rotation, it's at most 5 * sqrt(3) in every direction, which is about 8.66
    // So let's be generous and say it fits within [-10, 10]^3
    // The canvas size has a 5:7 aspect ratio, so if our width is 20 units wide, then
    // the height is 20 / (5/7) = 28 units.
    // So we want to map the box [-10, 10] x [-14, 14] x [-10, 10] to the
    // normalized box [-1, 1], [-1, 1], [0, 1].
    // for x: divide by 10
    // for y: divide by 14
    // for z: divide by 2 * 10 (to get [-0.5, 0.5]), 
    //        multiply by -1 (since depth goes into the screen),
    //        then add 0.5 to get [0, 1]
    var position_clip = rotated / vec3f(10.0, 14.0, -20.0) + vec3f(0.0, 0.0, 0.5);

    let coords_from_center = grid_coords - DIMENSIONS / 2;
    const RADIUS = 3.5;
    let yeet_mask = step(RADIUS * RADIUS, f32(dot(coords_from_center, coords_from_center)));

    const YEET = 10000.0;
    position_clip.x += YEET * yeet_mask; //f32(dot(grid_coords, vec3u(1)) % 2 == 1);

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
    let dist = 2.0 * abs(input.uvw - 0.5);
    let edge_masks = step(vec3f(0.8), dist);
    let masks = edge_masks.yzx * edge_masks.zxy;
    let brightness = max(max(masks.x, masks.y), masks.z);
    let color = mix(input.uvw, vec3f(0.0), brightness);

    return vec4f(color, 1.0);
}