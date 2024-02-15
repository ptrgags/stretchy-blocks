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
    @location(0) position_world: vec3f,
    @location(1) normal: vec3f,
    // UV coordinates within the current block.
    @location(2) uvw: vec3f,
    // Integer grid coordinates of the current block
    @location(3) @interpolate(flat) grid_coords: vec3u,
    @location(4) global_uvw: vec3f,
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

// Because WGSL's mod uses truncate, not floor :(
fn floor_mod(x: f32, modulus: f32) -> f32 {
    return modulus * fract(x / modulus);
}

// Difference of remainders from two different moduli, with output 
// rescaled to be between [0, 1].
//
// This operation was inspired by the Bridges paper "Let the Numbers Do the
// Walking: Generating Turtle Dances on the Plane from Integer Sequences",
// though used for a very different purpose.
// 
// https://archive.bridgesmathart.org/2017/bridges2017-139.html
fn mod_diff(x: f32, a: f32, b: f32) -> f32 {
    let diff = floor_mod(x, a) - floor_mod(x, b);

    // the range of diff is [-(b-1), (a-1)], so remap to [0, 1].
    return (diff + (b - 1.0)) / (a + b - 2.0);
}

/**
 * Make a train of bell curves such that for every unit of the domain
 * there is one bell curve. The output range is slightly less than [0, 1],
 * 
 * This is the blue curve in this Desmos sketch. Note that this can be
 * combined with higher frequencies to make different shapes.
 * https://www.desmos.com/calculator/65xv3qteke
 */
fn bell_train(x: f32) -> f32 {
    let repeated = 4.0 * fract(x) - 2.0;
    return exp(-repeated * repeated);
}

fn visibility_mask(uvw: vec3f, time: f32) -> f32 {
    // Simulate some random blobs moving vertically through the space.

    // Advance time slowly
    let t = uvw.y - 0.1 * time;

    // Use difference of remainders to jitter the
    // The pattern is periodic, but it takes a while to go through.
    let center = vec2f(
        mod_diff(t, 3, 5),
        mod_diff(t, 3, 4),
    );

    // Compute distance from the center for this horizontal plane
    let dist = length(uvw.xz - center);

    // Make a train of bell curves to define the radius for each horizontal
    // slice. I combined two different frequencies to make a bimodal shape,
    // see here:
    // https://www.desmos.com/calculator/kto0shp4c4
    let radius = 1.0 / 5.0 * (0.5 * bell_train(t) + 2.0 * bell_train(2.0 * t));

    // Finally, combine the distance field and the radius to make a solid
    // of revolution
    return smoothstep(radius + 0.1, radius, dist);
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

/* 
 * warp the domain [0, 1] -> [0, 1] using a single 2D control point. This is
 * used as a helper to compute the stretch function.
 * 
 * Inspired by a generalization of the pulse-width-modulation functionality on
 * my ASM Hydrasynth synthesizer. It warps the domain by picking a control point
 * C = (a, b) and makes two line segments, one between (0, 0) and C and the
 * other between C and (1, 1). This works similarly to the power function x^p,
 * however the 2D control point is easier to control.
 *
 * in particular: when the control point is below y = x, it compresses the 
 * low end of the range while stretching out the high end of the range. When
 * the control point is above y = x, it compresses the high end and stretches out
 * the low end. When the control point is on y = x, the range is unchanged
 *
 * This is the blue curve f(x) in the Desmos graph https://www.desmos.com/calculator/rid2vsvpou
 */
fn pwm_warp(control_point: vec2f, x: f32) -> f32 {
    let a = control_point.x;
    let b = control_point.y;

    // Create a line from (0, 0) to control_point
    let line_0c = mix(0.0, b, x / a);
    // create a line from control_point to (1, 1)
    let line_c1 = mix(b, 1.0, (x - a) / (1.0 - a));

    // Switch between the lines depending on whether the x coordinate is
    // to the left or right of a.
    //
    // Why the step function and not just x? while the latter would give a
    // single quadratic curve, it sometimes goes outside the [0, 1] range
    // which is undesirable for the stretch function.
    return mix(line_0c, line_c1, step(a, x));
}

/*
 * pwm_warp only stretches values to one side of the unit square. I want
 * to bunch symmetrically about the center, so this makes the curve have
 * rotation symmetry about (0.5, 0.5).
 *
 * If the control point is below y = x, the curve looks like a linear
 * approximation of smoothstep(), but the control point allows more variation,
 * e.g. moving it above y = x makes it work like an approximate inverse
 * smoothstep without needing inverse trig functions!
 *
 * see the purple curve h(x) in the Desmos graph https://www.desmos.com/calculator/rid2vsvpou
 */
fn symmetric_warp(control_point: vec2f, x: f32) -> f32 {
    // Split the domain into two halves
    let halves = modf(2.0 * x);


    // The first half will use the warp function, the second half will
    // use a rotated copy
    let curve = pwm_warp(control_point, halves.fract);
    // mirroring the control point has the asame effect as rotating
    // the curve 180 degrees
    let rotated = pwm_warp(1.0 - control_point, halves.fract);
    let symmetric_curve = halves.whole + mix(curve, rotated, halves.whole);

    // The range of the curve is now [0, 2], we want [0, 1]
    return 0.5 * symmetric_curve;
}

// Triangle wave "cosine" and "sine". These have periods of 1
fn triangle_cos(x: f32) -> f32 {
    return abs(2.0 * x % 2.0 - 1.0);
}

fn triangle_sin(x: f32) -> f32 {
    return triangle_cos(x - 0.25);
}

fn diamond(t: f32) -> vec2f {
    return vec2f(triangle_cos(t), triangle_sin(t));
}

fn hesitate(t: f32, pause:f32) -> f32 {
    return max((t - pause) / (1.0 - pause), 0.0);
}

/* 
 * Normally time passes 1 second per second. This function
 * pauses the progression of time freq times per second. 
 * The "pause" parameter from [0, 1] 
 * lets you control how long each pause is as a fraction of a single period.
 *
 * See this graph in Desmos:
 * https://www.desmos.com/calculator/iwvpzqangh
 */
fn hesitate_repeat(t: f32, pause:f32, freq: f32) -> f32 {
    let scaled = freq * t;
    return (floor(scaled) + hesitate(fract(scaled), pause)) / freq;
}

fn stretch(uvw: vec3f, time: f32) -> vec3f {
    let pause_at_corners = hesitate_repeat(0.25 * time, 0.25, 4.0);
    let control_point = 0.5 + 0.375 * diamond(pause_at_corners);

    let x = symmetric_warp(control_point, uvw.x);
    let z = symmetric_warp(control_point, uvw.z);

    return vec3f(x, uvw.y, z);
    //return uvw;
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

// cube_vertices * instance_count -> 36 * 4096
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

    // Mess with the coordinates a bit to make the blocks stretch and squish.
    let position_stretched = stretch(position_instance, uniforms.time);

    // Position the model in world space. Make the overall cube fit in [-1, 1]
    let position_world = 2.0 * position_stretched - 1.0;

    // For isometric projection, rotate around the y axis as desired, then
    // tilt the model towards the camera so the top faces are showing
    const MAGIC_ANGLE = 0.615479709; // atan(1/sqrt(2))
    let TILT = rotate_x(MAGIC_ANGLE);
    let isometric = TILT * rotate_y(0.2 * uniforms.time);
    let rotated = isometric * position_world;

    let position_clip = project_orthographic(rotated);

    // the normal is unchanged
    // the only thing we do in world space is rotate it. But note that
    //
    // rotation^(-T) = (rotation^T)^T = rotation
    // 
    // for rotation matrices, so we just need to multiply the normal
    // by the isometric projection matrix
    let normal_world = isometric * input.normal;

    var output: VertexOutput;
    output.position = vec4f(position_clip, 1.0);
    output.position_world = rotated;
    output.normal = normal_world;
    output.grid_coords = grid_coords;

    // Position also serves as uvw coordinates!
    output.uvw = input.position;
    output.global_uvw = position_instance;
    return output;
}

fn wide_cos(x: f32, q: f32) -> f32 {
    return min(q * cos(x) + q, 1.0);
}


// Colors from Paletton. The output were percents, hence having to divide
// by 100.
//
// The palette was listed in reverse order of what I want. I'll fix it
// in the math below :P.
const PALETTE_LENGTH = 16;
const PALETTE = array<vec3f, PALETTE_LENGTH>(
    // green: brightest -> darkest
    0.01 * vec3f(87.1, 94.9, 29.8),
    0.01 * vec3f(80.4, 89.4, 12.9),
    0.01 * vec3f(64.3, 72.5, 4.7),
    0.01 * vec3f(50.6, 57.3, 0),

    // orange: brightest -> darkest
    0.01 * vec3f(97.3, 60.8, 30.6),
    0.01 * vec3f(91.8, 49, 13.3),
    0.01 * vec3f(74.1, 36.1, 4.7),
    0.01 * vec3f(58.8, 26.7, 0),

    // blue: brightest -> darkest
    0.01 * vec3f(18.4, 58.4, 58.4),
    0.01 * vec3f(7.8, 54.9, 54.9),
    0.01 * vec3f(2.7, 44.7, 44.7),
    0.01 * vec3f(0, 35.3, 35.3),

    // purple: brightest -> darkest
    0.01 * vec3f(57.6, 21.2, 63.9),
    0.01 * vec3f(52.9, 9.8, 60.4),
    0.01 * vec3f(42, 3.9, 48.6),
    0.01 * vec3f(32.9, 0.8, 38.4),
);

fn pick_color(grid_coords: vec3u) -> vec3f {
    // Divide the scene into 4 layers of 4 blocks thick. this will control
    // the hue
    let layer = grid_coords.y / 4;

    // Height within layer
    let height = layer % 4;

    // Compute a checkerboard pattern, every cell is either 0 or 1.
    let diagonal_plane = dot(grid_coords, vec3u(1));
    let checkerboard = diagonal_plane % 2;

    // The layer controls the primary color, but to make the checkerboard
    // pattern stand out, when the checkerboard pattern is 1, switch to the next
    // hue cyclically.
    let hue = (layer + checkerboard) % 4;

    // Within each layer, make a brightness gradient from 0-3
    let brightness = height;

    // Look up the color.
    // The colors are listed in reverse, hence the backwards indexing.
    let color_index = (4 * hue + brightness) % PALETTE_LENGTH;
    return PALETTE[(PALETTE_LENGTH - 1) - color_index];
}

@fragment
fn fragment_main(input: VertexOutput) -> @location(0) vec4f {
    // Diffuse lighting
    let diffuse_color = pick_color(input.grid_coords);
    const light_world = normalize(vec3f(-0.5, 0.5, 1.0));
    let normal_world = normalize(input.normal);
    let diffuse = diffuse_color * max(dot(normal_world, light_world), 0.0);

    // Highlight the edge black
    let dist = 2.0 * abs(input.uvw - 0.5);
    let edge_masks = step(vec3f(0.8), dist);
    let masks = edge_masks.yzx * edge_masks.zxy;
    let edge = max(max(masks.x, masks.y), masks.z);

    let color = mix(diffuse, vec3f(0.0), edge);

    return vec4f(color, 1.0);
}
