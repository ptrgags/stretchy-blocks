interface GenerateBlocksBuffers {

}

class GenerateBlocks {
    buffers: GenerateBlocksBuffers;

    constructor(buffers: GenerateBlocksBuffers) {
        this.buffers = buffers;
    }

    compute_pass(encoder: GPUCommandEncoder) {

    }
}