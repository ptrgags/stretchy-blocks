import { compile_shader } from "./compile_shader.js";
const SWAP_CHAIN_FORMAT = "bgra8unorm";
// 6 faces * 3 triangle vertices (unique because of normals)
const CUBE_VERTICES = 36;
const FACES = 6;
const SIZE_FLOAT = 4;
const SIZE_VEC3F = SIZE_FLOAT * 3;
const SIZE_POSITION = SIZE_VEC3F;
const SIZE_NORMAL = SIZE_VEC3F;
const SIZE_VERTEX = SIZE_POSITION + SIZE_NORMAL;
const N = 16;
const DIMENSIONS = [N, N, N];
const INSTANCE_COUNT = N * N * N;
const CUBE_POSITIONS = [
    [0, 0, 0],
    [1, 0, 0],
    [0, 1, 0],
    [1, 1, 0],
    [0, 0, 1],
    [1, 0, 1],
    [0, 1, 1],
    [1, 1, 1],
];
const CUBE_NORMALS = [
    [1, 0, 0],
    [-1, 0, 0],
    [0, 1, 0],
    [0, -1, 0],
    [0, 0, 1],
    [0, 0, -1],
];
// Each entry is listed in the same order as CUBE_NORMALS,
// but this indexes into CUBE_POSITIONS
const CUBE_QUADS = [
    // +x
    [1, 3, 7, 5],
    // -x
    [0, 4, 6, 2],
    // +y
    [3, 2, 6, 7],
    // -y
    [0, 1, 5, 4],
    // +z
    [4, 5, 7, 6],
    // -z
    [0, 2, 3, 1],
];
function* generate_cube_data() {
    for (let i = 0; i < FACES; i++) {
        const normal = CUBE_NORMALS[i];
        const [a, b, c, d] = CUBE_QUADS[i].map(index => CUBE_POSITIONS[index]);
        // Triangle abc
        yield* a;
        yield* normal;
        yield* b;
        yield* normal;
        yield* c;
        yield* normal;
        // Triangle cda
        yield* c;
        yield* normal;
        yield* d;
        yield* normal;
        yield* a;
        yield* normal;
    }
}
const CUBE_DATA = [...generate_cube_data()];
function configure_context(device, context) {
    context?.configure({
        device,
        // for compositing on the page
        alphaMode: "opaque",
        // swap chain format
        format: "bgra8unorm",
        usage: GPUTextureUsage.RENDER_ATTACHMENT
    });
}
function make_uniform_buffer(device) {
    const uniform_buffer = device.createBuffer({
        // dimensions is a vec3u but padded to 4 u32
        // time is a f32, but padded to 4 floats
        size: 4 * SIZE_FLOAT,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        mappedAtCreation: true,
    });
    let view = new DataView(uniform_buffer.getMappedRange());
    const little_endian = true;
    const [W, H, D] = DIMENSIONS;
    view.setUint32(0, W, little_endian);
    view.setUint32(4, H, little_endian);
    view.setUint32(8, D, little_endian);
    const time = 0.0;
    view.setFloat32(12, time, little_endian);
    uniform_buffer.unmap();
    return uniform_buffer;
}
function make_vertex_buffer(device) {
    const vertex_buffer = device.createBuffer({
        size: CUBE_VERTICES * SIZE_VERTEX,
        usage: GPUBufferUsage.VERTEX,
        mappedAtCreation: true
    });
    let typed_array = new Float32Array(vertex_buffer.getMappedRange());
    typed_array.set(CUBE_DATA);
    vertex_buffer.unmap();
    return vertex_buffer;
}
function make_bind_group(device, layout, uniform_buffer) {
    return device.createBindGroup({
        label: "stretchy_blocks",
        layout,
        entries: [
            {
                binding: 0,
                resource: {
                    buffer: uniform_buffer
                }
            }
        ]
    });
}
function make_bind_group_layout(device) {
    return device.createBindGroupLayout({
        label: "stretchy_blocks",
        entries: [
            {
                binding: 0,
                visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT,
                buffer: {
                    type: "uniform"
                }
            }
        ]
    });
}
function make_depth_texture(device, canvas) {
    return device.createTexture({
        size: [canvas.width, canvas.height],
        format: "depth24plus",
        usage: GPUTextureUsage.RENDER_ATTACHMENT
    });
}
async function make_render_pipeline(device, bind_group_layout) {
    const shader_module = await compile_shader(device, "shaders/stretchy_blocks.wgsl");
    const pipeline_layout = device.createPipelineLayout({
        bindGroupLayouts: [bind_group_layout]
    });
    const vertex_state = {
        module: shader_module,
        entryPoint: "vertex_main",
        buffers: [
            {
                arrayStride: SIZE_VERTEX,
                attributes: [
                    {
                        // position
                        format: "float32x3",
                        offset: 0,
                        shaderLocation: 0
                    },
                    {
                        // normal
                        format: "float32x3",
                        offset: SIZE_VEC3F,
                        shaderLocation: 1
                    }
                ]
            }
        ]
    };
    const fragment_state = {
        module: shader_module,
        entryPoint: "fragment_main",
        targets: [
            {
                format: SWAP_CHAIN_FORMAT
            }
        ]
    };
    const render_pipeline = device.createRenderPipeline({
        layout: pipeline_layout,
        vertex: vertex_state,
        fragment: fragment_state,
        primitive: {
            topology: "triangle-list",
            frontFace: "ccw",
            cullMode: "back"
        },
        depthStencil: {
            depthWriteEnabled: true,
            depthCompare: "less",
            format: "depth24plus"
        },
    });
    return render_pipeline;
}
export class RenderPipeline {
    constructor(render_pipeline, uniform_buffer, vertex_buffer, bind_group, depth_texture) {
        this.render_pipeline = render_pipeline;
        this.uniform_buffer = uniform_buffer;
        this.vertex_buffer = vertex_buffer;
        this.bind_group = bind_group;
        this.depth_texture = depth_texture;
    }
    update_uniforms(device, time) {
        const buffer = new ArrayBuffer(4 * SIZE_FLOAT);
        const view = new DataView(buffer);
        const little_endian = true;
        const [W, H, D] = DIMENSIONS;
        view.setUint32(0, W, little_endian);
        view.setUint32(4, H, little_endian);
        view.setUint32(8, D, little_endian);
        view.setFloat32(12, time, little_endian);
        device.queue.writeBuffer(this.uniform_buffer, 0, buffer);
    }
    render(encoder, context) {
        const pass_description = {
            colorAttachments: [
                {
                    view: context.getCurrentTexture().createView(),
                    loadOp: "clear",
                    storeOp: "store",
                    clearValue: [0, 0, 0, 1]
                }
            ],
            depthStencilAttachment: {
                view: this.depth_texture.createView(),
                depthClearValue: 1.0,
                depthLoadOp: "clear",
                depthStoreOp: "store"
            }
        };
        const render_pass = encoder.beginRenderPass(pass_description);
        render_pass.setPipeline(this.render_pipeline);
        render_pass.setVertexBuffer(0, this.vertex_buffer);
        render_pass.setBindGroup(0, this.bind_group);
        render_pass.draw(CUBE_VERTICES, INSTANCE_COUNT);
        render_pass.end();
    }
    static async build(device, canvas, context) {
        configure_context(device, context);
        const uniform_buffer = make_uniform_buffer(device);
        const vertex_buffer = make_vertex_buffer(device);
        const bind_group_layout = make_bind_group_layout(device);
        const bind_group = make_bind_group(device, bind_group_layout, uniform_buffer);
        const depth_texture = make_depth_texture(device, canvas);
        const render_pipeline = await make_render_pipeline(device, bind_group_layout);
        return new RenderPipeline(render_pipeline, uniform_buffer, vertex_buffer, bind_group, depth_texture);
    }
}
