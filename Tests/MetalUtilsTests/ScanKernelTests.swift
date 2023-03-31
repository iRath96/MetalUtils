import XCTest
@testable import MetalUtils

final class ScanKernelTests: XCTestCase {
    func testScan() throws {
        let N = 123457
        
        let device = MTLCopyAllDevices()[0]
        
        let inputBuffer = device.makeBuffer(
            length: N * MemoryLayout<uint>.stride,
            options: .storageModeShared)!
        let expectedBuffer = device.makeBuffer(
            length: N * MemoryLayout<uint>.stride,
            options: .storageModeShared)!

        var actualSum: UInt32 = 0
        let ptrI = inputBuffer.contents().assumingMemoryBound(to: uint.self)
        let ptrE = expectedBuffer.contents().assumingMemoryBound(to: uint.self)
        for i in 0..<N {
            let element: UInt32 = (UInt32(i) % 3) ^ (UInt32(i) % 5)
            ptrI.advanced(by: i).pointee = element
            ptrE.advanced(by: i).pointee = actualSum
            actualSum &+= element
        }
        
        // setup pipeline
        
        let kernel = try ScanKernel(on: device, type: .uint32, capacity: N)
        
        let commandQueue = device.makeCommandQueue()!
        let commandBuffer = commandQueue.makeCommandBuffer()!
        kernel.encodeScan(commandBuffer: commandBuffer, input: inputBuffer, count: N)
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        
        // check results
        
        for i in 0..<N {
            let actual = ptrI.advanced(by: i).pointee
            let expected = ptrE.advanced(by: i).pointee
            XCTAssertEqual(actual, expected)
        }
    }
}
