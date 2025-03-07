import Metal
import Foundation  

@_cdecl("run_matmul")
public func run_matmul(inA: UnsafePointer<Float>,
                       inB: UnsafePointer<Float>,
                       outC: UnsafeMutablePointer<Float>,
                       size: Int) {

    let devices = MTLCopyAllDevices()

    guard let device = devices.first else {
        print("Error: No Metal-compatible GPU found.")
        return
    }

    print("Using Metal device: \(device.name)")

    let metalLibPath = "/Users/ashleyczumak/metaltest/matmul.metallib"
    let metalLibURL = URL(fileURLWithPath: metalLibPath)

    guard let library = try? device.makeLibrary(URL: metalLibURL) else {
        print("Error: Failed to load Metal library from path: \(metalLibPath)")
        return
    }

    guard let function = library.makeFunction(name: "matmul") else {
        print("Error: Function 'matmul' not found in Metal library.")
        return
    }

    guard let pipelineState = try? device.makeComputePipelineState(function: function) else {
        print("Error: Failed to create pipeline state.")
        return
    }

    let commandQueue = device.makeCommandQueue()!

    let bufferA = device.makeBuffer(bytes: inA, length: size * size * MemoryLayout<Float>.size, options: [])
    let bufferB = device.makeBuffer(bytes: inB, length: size * size * MemoryLayout<Float>.size, options: [])
    let bufferC = device.makeBuffer(length: size * size * MemoryLayout<Float>.size, options: []) // Empty output buffer

    guard bufferA != nil, bufferB != nil, bufferC != nil else {
        print("Error: Failed to create Metal buffers.")
        return
    }

    let commandBuffer = commandQueue.makeCommandBuffer()!
    let encoder = commandBuffer.makeComputeCommandEncoder()!
    encoder.setComputePipelineState(pipelineState)
    encoder.setBuffer(bufferA, offset: 0, index: 0)
    encoder.setBuffer(bufferB, offset: 0, index: 1)
    encoder.setBuffer(bufferC, offset: 0, index: 2)

    var mutableSize = size
    encoder.setBytes(&mutableSize, length: MemoryLayout<Int>.size, index: 3) // Pass size

    let gridSize = MTLSize(width: size, height: size, depth: 1)
    let threadgroupSize = MTLSize(width: min(16, size), height: min(16, size), depth: 1)
    encoder.dispatchThreads(gridSize, threadsPerThreadgroup: threadgroupSize)

    encoder.endEncoding()
    commandBuffer.commit()
    commandBuffer.waitUntilCompleted()

    let outputPointer = bufferC!.contents().assumingMemoryBound(to: Float.self)
    for i in 0..<(size * size) {
        outC[i] = outputPointer[i]
    }

    print("GPU Computation Output (First 10 Values) (debugging):", Array(UnsafeBufferPointer(start: outC, count: 10)))
}
