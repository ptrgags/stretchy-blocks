async function get_device(): Promise<GPUDevice> {
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) {
        throw new Error("Your browser doesn't support WebGPU :(");
    }

    return await adapter.requestDevice();
}

async function make_shader(device: GPUDevice, filename: string): Promise<GPUShaderModule> {
    const response = await fetch(filename);
    const code = await response.text();

    const shader_module = device.createShaderModule({code});
    const compilation_info = await shader_module.getCompilationInfo();
    if (compilation_info.messages.length > 0) {
        let had_error = false;
        console.log("Shader compilation log:");
        for (const msg of compilation_info.messages) {
            console.log(`${msg.lineNum}:${msg.linePos} - ${msg.message}`);
            had_error ||= msg.type === "error";

            if (had_error) {
                throw new Error("Shader failed to compile!");
            }
        }
    }

    return shader_module;
}

async function main() {
    const device = await get_device();
    const render = () => {
        const commandEncoder = device.createCommandEncoder();
        device.queue.submit([commandEncoder.finish()]);
        requestAnimationFrame(render);
    }
    requestAnimationFrame(render);
}

main();