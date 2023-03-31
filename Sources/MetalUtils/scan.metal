#include <metal_stdlib>
using namespace metal;

template<typename T, ushort Length>
static inline T threadScan(threadgroup T *values) {
    for (ushort i = 1; i < Length; i++) values[i] += values[i - 1];
    T sum = values[Length - 1];
    for (ushort i = Length; i > 0; i--) values[i] = values[i - 1];
    values[0] = 0;
    return sum;
}

template<typename T, ushort Length>
static inline void threadUniformAdd(threadgroup T *values, T operand) {
    for (ushort i = 0; i < Length; i++) values[i] += operand;
}

template<typename T, ushort BlockSize>
static inline T threadgroupRakingScan(
    T value,
    threadgroup T *shared,
    ushort index
) {
    shared[index] = value;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    if (index < 32) {
        T partialSum = threadScan<T, BlockSize / 32>(shared + index * (BlockSize / 32));
        T prefix = simd_prefix_exclusive_sum(partialSum);
        threadUniformAdd<T, BlockSize / 32>(shared + index * (BlockSize / 32), prefix);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    return shared[index];
}

template<typename T, ushort BlockSize>
kernel void scan(
    device T *input,
    device T *partials,
    uint groupIndex [[threadgroup_position_in_grid]],
    ushort localIndex [[thread_position_in_threadgroup]]
) {
    const uint baseIndex = groupIndex * BlockSize;
    threadgroup T scratch[BlockSize];
    
    T value = input[baseIndex + localIndex];
    T prefix = threadgroupRakingScan<T, BlockSize>(value, scratch, localIndex);
    input[baseIndex + localIndex] = prefix;
    
    if (localIndex == BlockSize - 1) {
        partials[groupIndex] = value + prefix;
    }
}

template<typename T, ushort BlockSize>
kernel void uniformAdd(
    device T *input,
    device T *partials,
    uint groupIndex [[threadgroup_position_in_grid]],
    ushort localIndex [[thread_position_in_threadgroup]]
) {
    threadgroup T scratch;
    if (localIndex == 0) {
        scratch = partials[groupIndex];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    input[groupIndex * BlockSize + localIndex] += scratch;
}

template [[host_name("scan_256")]]
kernel void scan<uint, 256>(device uint *, device uint *, uint, ushort);

template [[host_name("uniform_add_256")]]
kernel void uniformAdd<uint, 256>(device uint *, device uint *, uint, ushort);
