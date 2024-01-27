import { RenderPipeline } from "./render_pipeline.js";

async function get_device(): Promise<GPUDevice> {
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) {
        throw new Error("Your browser doesn't support WebGPU :(");
    }

    return await adapter.requestDevice()
}

async function main() {
    const device = await get_device()

    const canvas = document.getElementById("webgpu-canvas") as HTMLCanvasElement
    const context = canvas.getContext("webgpu")

    if (context === null) {
        return;
    }

    const render_pipeline = await RenderPipeline.build(device, canvas, context)

    const start = performance.now()
    const render = () => {
        const elapsed_time = (performance.now() - start) / 1000.0
        const encoder = device.createCommandEncoder()

        render_pipeline.update_uniforms(device, elapsed_time)
        render_pipeline.render(encoder, context)
        
        device.queue.submit([encoder.finish()])
        requestAnimationFrame(render)
    }
    requestAnimationFrame(render)
}

main()

function export_screenshot(data_url: string) {
    const a = document.createElement('a')
    a.href = data_url
    a.download = 'screenshot.png'
    document.body.appendChild(a);
    a.click();
    a.remove();
}

// This works, just gotta 
document.addEventListener('keyup', event => {
    if (event.code === 'Space') {
        const canvas = document.getElementById('webgpu-canvas') as HTMLCanvasElement
        const data_url = canvas?.toDataURL();
        if (data_url !== undefined) {
            export_screenshot(data_url);
        }
    }
})