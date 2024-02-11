struct VertexInput {
    @builtin(instance_index) instance_index: u32,
    // Vertex position with components in the range [0, 1]
    @location(0) position: vec3f,
    // Vertex normal. For a cube this will always be one of the 6 cardinal
    // directions
    @location(1) normal: vec3f,
}

struct VertexOutput {
    @builtin(position) position: vec4f,
    @location(0) normal: vec3f,
    // UV coordinates within the current block.
    @location(1) uvw: vec3f,
    // Integer grid coordinates of the current block
    @location(2) @interpolate(flat) grid_coords: vec3u,
}

struct Uniforms {
    // Grid dimensions
    dimensions: vec3u,
    // Elapsed time in seconds
    time: f32,
}

@group(0) @binding(0) var<uniform> uniforms: Uniforms;

fn rotate_x(angle: f32) -> mat3x3f {
    let c = cos(angle);
    let s = sin(angle);
    return mat3x3f(
        1, 0, 0,
        0, c, s,
        0, -s, c,
    );
}

fn rotate_y(angle: f32) -> mat3x3f {
    let c = cos(angle);
    let s = sin(angle);
    return mat3x3f(
        c, 0, -s,
        0, 1, 0,
        s, 0, c,
    );
}

fn get_grid_coords(index: u32, dimensions: vec3u) -> vec3u {
    // index = z * WH + y * W + x
    // modulo the width gives x
    let x = index % dimensions.x;

    // using integer division, we have
    // intermediate = z * H + y
    // so the quotient is z, remainder is y
    let intermediate = index / dimensions.x;
    let y = intermediate % dimensions.y;
    let z = intermediate / dimensions.y;

    return vec3u(x, y, z);
}

fn visibility_mask(uvw: vec3f, time: f32) -> f32 {
    let height = 3.0 * sin(time);
    return clamp(dot(uvw, vec3f(1.0)) + height, 0.0, 1.0);
}

fn compute_visibility(grid_coords_normalized: vec3f, position: vec3f, time: f32) -> vec3f {
    // Compute a grayscale pattern to decide which blocks are visible
    let visibility = visibility_mask(grid_coords_normalized, time);

    // Scale the position relative to the center.
    // When the visibility is 1, this will be the full size, but
    // when visibility is < 1, the cube will shrink.
    return visibility * (position - 0.5) + 0.5;
}

fn position_instances(position_model: vec3f, grid_coords: vec3f, dimensions: vec3f) -> vec3f {
    // stack the cubes, this gives coordinates in the range [0, dimensions]
    let stacked = grid_coords + position_model;
    // rescale to [0, 1]
    return stacked / dimensions;
}

fn project_orthographic(rotated: vec3f) -> vec3f {
    // Before rotation, the blocks fit within [-1, 1]^3.
    // So the diagonal is cbrt(1 - -1) = cbrt(2)
    // so the cube fits in a sphere with radius cbrt(2)/2 which is about 0.63.
    // Let's allow a margin from [-2, 2].
    // The canvas is trading card
    const box_radius = 2.0;
    const aspect_ratio = 5.0 / 7.0;
    const dimensions = vec3f(box_radius, box_radius / aspect_ratio, box_radius);

    // Rescale to map to the volume [-1, 1]^3
    var rescaled = rotated / dimensions;
    // However, clip space has z reversed and in the range [0, 1], so
    // fix that.
    rescaled.z = -0.5 * rescaled.z + 0.5;

    return rescaled;

    
    //return rotated / dimensions * flip_z + vec3f(0.0, 0.0, 0.5);

    // The canvas is a trading card size, i.e. 5:7 aspect ratio. So if the
    // x and z coordinates have a width of 2, then the height will be 
    // 2 * 7/5 = 14/5
    // So we have a view volume of [-1, 1] x [-14/10, 14/10] x [-1, 1]
    // We want to map it to clip space, [-1, 1] x [-1, 1] x [0, 1]
    // (with z reversed since it's into the screen)
    //
    // for x: no change
    // for y: divide by 10/14
    // for z: 0.5 + 0.5 * x
    //return rotated * vec3f(1.0 * 1.0 / 2.0, 14.0 / 10.0, -0.5 * 1.0 / 2.0) + vec3f(0.0, 0.0, 0.5);
}

// Raise x to a power:
// k = 0 -> 0
// k = 0.5 -> 1
// k = 1 -> infinity
fn stretch_function(x: vec3f, k: vec3f) -> vec3f {
    let exponent = tan(radians(90) * k);
    return pow(x, exponent);
}

fn stretch_blocks(
    grid_coords: vec3f,
    dimensions: vec3f,
    position: vec3f,
    time: f32
) -> vec3f {
    let k = vec3f(0.5 + 0.2 * sin(2.0 * time));
    let min_coords = grid_coords / dimensions;
    let max_coords = (grid_coords + 1.0) / dimensions;

    let min_corner = stretch_function(min_coords, k);
    let max_corner = stretch_function(max_coords, k);
    let position_percent = mix(min_corner, max_corner, position);
    return position_percent * dimensions;
}

@vertex
fn vertex_main(input: VertexInput) -> VertexOutput {
    // Position the cube instance. Each cube will be 1 unit wide, and
    // we want to center this based on the overall bounding box.
    let grid_coords = get_grid_coords(input.instance_index, uniforms.dimensions);
    // Normalize so the the coords are in [0, 1]
    let grid_coords_normalized = vec3f(grid_coords) / vec3f(uniforms.dimensions - 1);

    // First, compute how visible each block is. Shrink blocks that are
    // only partially visible. This operates in the [0, 1] space of the block.
    let position_model = compute_visibility(
        grid_coords_normalized,
        input.position, uniforms.time
    );

    // now position and scale the instances to form a cube in [0, 1]
    let position_instance = position_instances(position_model, vec3f(grid_coords), vec3f(uniforms.dimensions)); 

    // TODO: apply a stretching function from [0, 1]^3 -> [0, 1]^3
    /*
    let model_coords = stretch_blocks(
        vec3f(grid_coords),
        vec3f(uniforms.dimensions),
        input.position,
        uniforms.time
    );
    */

    // Position the model in world space. Make the overall cube fit in [-1, 1]
    let position_world = 2.0 * position_instance - 1.0;

    // For isometric projection, rotate around the y axis as desired, then
    // tilt the model towards the camera so the top faces are showing
    const MAGIC_ANGLE = 0.615479709; // atan(1/sqrt(2))
    let TILT = rotate_x(MAGIC_ANGLE);
    let isometric = TILT * rotate_y(uniforms.time);

    let rotated = isometric * position_world;
    

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
    //var position_clip = rotated / vec3f(10.0, 14.0, -20.0) + vec3f(0.0, 0.0, 0.5);

    /*let coords_from_center = grid_coords - uniforms.dimensions / 2;
    const RADIUS = 3.5;
    let yeet_mask = step(RADIUS * RADIUS, f32(dot(coords_from_center, coords_from_center)));
    */

    let position_clip = project_orthographic(rotated);

    //const YEET = 10000.0;
    //position_clip.x += YEET * yeet_mask; //f32(dot(grid_coords, vec3u(1)) % 2 == 1);

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
