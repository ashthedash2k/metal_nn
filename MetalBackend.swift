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
    let bufferC = device.makeBuffer(length: size * size * MemoryLayout<Float>.size, options: []) 

    guard bufferA != nil, bufferB != nil, bufferC != nil else {
        print("Error: Failed to create Metal buffers.")
        return
    }

    let commandBuffer = commandQueue.makeCommandBuffer()!

    let computeEncoder = commandBuffer.makeComputeCommandEncoder()!
    computeEncoder.setComputePipelineState(pipelineState)
    computeEncoder.setBuffer(bufferA, offset: 0, index: 0)
    computeEncoder.setBuffer(bufferB, offset: 0, index: 1)
    computeEncoder.setBuffer(bufferC, offset: 0, index: 2)

    var mutableSize = size
    computeEncoder.setBytes(&mutableSize, length: MemoryLayout<Int>.size, index: 3)

    let gridSize = MTLSize(width: size, height: size, depth: 1)
    let threadgroupSize = MTLSize(width: min(16, size), height: min(16, size), depth: 1)

    let startTime = CFAbsoluteTimeGetCurrent()  // Start CPU Timer
    computeEncoder.dispatchThreads(gridSize, threadsPerThreadgroup: threadgroupSize)
    computeEncoder.endEncoding()

    commandBuffer.addCompletedHandler { _ in
        let endTime = CFAbsoluteTimeGetCurrent()  // End CPU Timer
        let gpuElapsedTime = endTime - startTime  // Compute Execution Time
        gpuTimePtr.pointee = gpuElapsedTime * 1000  // Convert to milliseconds
        print("GPU Execution Time: \(gpuElapsedTime * 1000) ms")
    }

    commandBuffer.commit()
    commandBuffer.waitUntilCompleted()

    let outputPointer = bufferC!.contents().assumingMemoryBound(to: Float.self)
    for i in 0..<(size * size) {
        outC[i] = outputPointer[i]
    }

    print("GPU Computation Output (First 10 Values) (debugging):", Array(UnsafeBufferPointer(start: outC, count: 10)))
}

var gpuTimePtr = UnsafeMutablePointer<Double>.allocate(capacity: 1)

@_cdecl("get_gpu_time")
public func get_gpu_time(outTime: UnsafeMutablePointer<Double>) {
    outTime.pointee = gpuTimePtr.pointee
}
