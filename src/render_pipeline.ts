import { compile_shader } from "./compile_shader.js";

const SWAP_CHAIN_FORMAT = "bgra8unorm"

// 6 faces * 3 triangle vertices (unique because of normals)
const CUBE_VERTICES = 36;
const FACES = 6;
const SIZE_VEC3F = 4 * 3;
const SIZE_POSITION = SIZE_VEC3F;
const SIZE_NORMAL = SIZE_VEC3F;
const SIZE_VERTEX = SIZE_POSITION + SIZE_NORMAL;

const INSTANCE_COUNT = 10 * 10 * 10;

const CUBE_POSITIONS = [
    [0, 0, 0],
    [1, 0, 0],
    [0, 1, 0],
    [1, 1, 0],
    [0, 0, 1],
    [1, 0, 1],
    [0, 1, 1],
    [1, 1, 1],
]

const CUBE_NORMALS = [
    [1, 0, 0],
    [-1, 0, 0],
    [0, 1, 0],
    [0, -1, 0],
    [0, 0, 1],
    [0, 0, -1],
]

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
]

function * generate_cube_data(): Generator<number, void, undefined> {
    for (let i = 0; i < FACES; i++) {
        const normal: number[] = CUBE_NORMALS[i];
        const [a, b, c, d] = CUBE_QUADS[i].map(index => CUBE_POSITIONS[index]);
        
        // Triangle abc
        yield *a
        yield *normal
        yield *b
        yield *normal
        yield *c
        yield *normal

        // Triangle cda
        yield *c
        yield *normal
        yield *d
        yield *normal
        yield *a
        yield *normal
    }
}

const CUBE_DATA: number[] = [...generate_cube_data()]

function configure_context(device: GPUDevice, context: GPUCanvasContext) {
    context?.configure({
        device,
        // for compositing on the page
        alphaMode: "opaque",
        // swap chain format
        format: "bgra8unorm",
        usage: GPUTextureUsage.RENDER_ATTACHMENT
    })
}

function make_vertex_buffer(device: GPUDevice): GPUBuffer {
    const vertex_buffer = device.createBuffer({
        size: CUBE_VERTICES * SIZE_VERTEX,
        usage: GPUBufferUsage.VERTEX,
        mappedAtCreation: true
    })

    let typed_array = new Float32Array(vertex_buffer.getMappedRange())
    typed_array.set(CUBE_DATA)

    vertex_buffer.unmap()

    return vertex_buffer
}

async function make_render_pipeline(device: GPUDevice): Promise<GPURenderPipeline> {
    const shader_module = await compile_shader(device, "shaders/stretchy_blocks.wgsl");
    
    const pipeline_layout = device.createPipelineLayout({
        bindGroupLayouts: []
    })

    const vertex_state: GPUVertexState = {
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
    }

    const fragment_state: GPUFragmentState = {
        module: shader_module,
        entryPoint: "fragment_main",
        targets: [
            {
                format: SWAP_CHAIN_FORMAT
            }
        ]
    }

    const render_pipeline = device.createRenderPipeline({
        layout: pipeline_layout,
        vertex: vertex_state,
        fragment: fragment_state,
    })

    return render_pipeline
}

export class RenderPipeline {
    render_pipeline: GPURenderPipeline
    vertex_buffer: GPUBuffer

    constructor(
        render_pipeline: GPURenderPipeline,
        vertex_buffer: GPUBuffer
    ) {
        this.render_pipeline = render_pipeline
        this.vertex_buffer = vertex_buffer
    }

    render(encoder: GPUCommandEncoder, context: GPUCanvasContext) {
        const pass_description: GPURenderPassDescriptor = {
            colorAttachments: [
                {
                    view: context.getCurrentTexture().createView(),
                    loadOp: "clear",
                    storeOp: "store",
                    clearValue: [0, 0, 0, 1]
                }
            ]
        }
        
        const render_pass = encoder.beginRenderPass(pass_description)
        render_pass.setPipeline(this.render_pipeline)
        render_pass.setVertexBuffer(0, this.vertex_buffer)
        render_pass.draw(CUBE_VERTICES, INSTANCE_COUNT);
        render_pass.end();
    }

    static async build(device: GPUDevice, context: GPUCanvasContext): Promise<RenderPipeline> {
        configure_context(device, context)
        const vertex_buffer = make_vertex_buffer(device);
        const render_pipeline = await make_render_pipeline(device);
        return new RenderPipeline(render_pipeline, vertex_buffer)
    }
}