const GRID_SIZE: u32 = 32u;

struct Block {
    id: u32,
    visible: bool,
}

@group(0) @binding(1) var<storage, read_write> blocks: array<Block>;

@compute
@workgroup_size(4, 4, 4)
fn main(
    @builtin(global_invocation_id) global_id: vec3u
) {
    let index = (
        global_id.z * GRID_SIZE * GRID_SIZE +
        global_id.y * GRID_SIZE +
        global_id.x
    );

    // Constant color, but make a checkerboard pattern in 3D
    blocks[index].id = 0;
    blocks[index].visible = dot(global_id, vec3u(1, 1, 1)) % 2;
}