import Foundation
import Metal

public enum ScanKernelDataType {
    case uint32
}

/**
 * Builds the exclusive prefix sum over UInt32 buffers.
 */
public class ScanKernel {
    private static let BlockSize = 256
    
    private let capacity: Int
    private let scratchBuffer: MTLBuffer
    private let scanKernel: MTLComputePipelineState
    private let uniformAddKernel: MTLComputePipelineState
    
    private static func buildScratchPyramid(capacity: Int) -> [Int] {
        var current = capacity
        var pyramid = [ 0 ]
        while true {
            current = (current + BlockSize - 1) / BlockSize
            pyramid.append(pyramid.last! + current)
            if current <= 1 {
                return pyramid
            }
        }
    }
    
    public init(on device: MTLDevice, type: ScanKernelDataType, capacity: Int) throws {
        self.capacity = capacity
        let scratchElementCount = ScanKernel.buildScratchPyramid(capacity: capacity).last!
        self.scratchBuffer = device.makeBuffer(
            length: MemoryLayout<UInt32>.stride * scratchElementCount)!
        
        let library = try device.makeDefaultLibrary(bundle: Bundle.module)
        self.scanKernel = try device.makeComputePipelineState(
            function: library.makeFunction(name: "scan_\(ScanKernel.BlockSize)")!)
        self.uniformAddKernel = try device.makeComputePipelineState(
            function: library.makeFunction(name: "uniform_add_\(ScanKernel.BlockSize)")!)
    }
    
    public func encodeScan(
        commandBuffer: MTLCommandBuffer,
        input: MTLBuffer,
        count: Int
    ) {
        let elementStride = MemoryLayout<UInt32>.stride
        let pyramid = ScanKernel.buildScratchPyramid(capacity: count)
        
        struct Stage {
            let inputBuffer: MTLBuffer
            let inputOffset: Int
            let outputBuffer: MTLBuffer
            let outputOffset: Int
            let elementCount: Int
        }
        let stages = (0..<(pyramid.count - 1)).map { level in
            Stage.init(
                inputBuffer: level == 0 ? input : scratchBuffer,
                inputOffset: level == 0 ? 0 : elementStride * pyramid[level - 1],
                outputBuffer: scratchBuffer,
                outputOffset: elementStride * pyramid[level],
                elementCount: level == 0 ? count : (pyramid[level] - pyramid[level - 1]))
        }
        
        if let encoder = commandBuffer.makeBlitCommandEncoder() {
            let rangeStart = elementStride * pyramid[0]
            let rangeEnd = elementStride * pyramid[pyramid.count - 2]
            encoder.fill(buffer: scratchBuffer, range: rangeStart..<rangeEnd, value: 0)
            encoder.endEncoding()
        }
        
        for stage in stages {
            if let encoder = commandBuffer.makeComputeCommandEncoder() {
                encoder.setComputePipelineState(scanKernel)
                encoder.setBuffer(stage.inputBuffer, offset: stage.inputOffset, index: 0)
                encoder.setBuffer(stage.outputBuffer, offset: stage.outputOffset, index: 1)
                encoder.dispatchThreads(
                    MTLSize(width: stage.elementCount, height: 1, depth: 1),
                    threadsPerThreadgroup: MTLSize(width: ScanKernel.BlockSize, height: 1, depth: 1))
                encoder.endEncoding()
            }
        }
        
        for stage in stages.reversed() {
            if let encoder = commandBuffer.makeComputeCommandEncoder() {
                encoder.setComputePipelineState(uniformAddKernel)
                encoder.setBuffer(stage.inputBuffer, offset: stage.inputOffset, index: 0)
                encoder.setBuffer(stage.outputBuffer, offset: stage.outputOffset, index: 1)
                encoder.dispatchThreads(
                    MTLSize(width: stage.elementCount, height: 1, depth: 1),
                    threadsPerThreadgroup: MTLSize(width: ScanKernel.BlockSize, height: 1, depth: 1))
                encoder.endEncoding()
            }
        }
    }
}