const std = @import("std");

pub const cudaError_t = c_int;
pub const cudaSuccess: cudaError_t = 0;
pub const cudaStream_t = *anyopaque;
pub const cudaEvent_t = *anyopaque;
pub const cudaMemcpyKind = c_int;
pub const cudaMemcpyHostToHost: cudaMemcpyKind = 0;
pub const cudaMemcpyHostToDevice: cudaMemcpyKind = 1;
pub const cudaMemcpyDeviceToHost: cudaMemcpyKind = 2;
pub const cudaMemcpyDeviceToDevice: cudaMemcpyKind = 3;
pub const cudaMemcpyDefault: cudaMemcpyKind = 4;

pub extern "c" fn cudaMalloc(devPtr: **anyopaque, size: usize) cudaError_t;
pub extern "c" fn cudaFree(devPtr: *anyopaque) cudaError_t;
pub extern "c" fn cudaMallocHost(ptr: **anyopaque, size: usize) cudaError_t;
pub extern "c" fn cudaFreeHost(ptr: *anyopaque) cudaError_t;
pub extern "c" fn cudaMemcpy(dst: *anyopaque, src: *const anyopaque, count: usize, kind: cudaMemcpyKind) cudaError_t;
pub extern "c" fn cudaMemcpyAsync(dst: *anyopaque, src: *const anyopaque, count: usize, kind: cudaMemcpyKind, stream: cudaStream_t) cudaError_t;
pub extern "c" fn cudaMemcpyPeer(dst: *anyopaque, dstDevice: c_int, src: *const anyopaque, srcDevice: c_int, count: usize) cudaError_t;
pub extern "c" fn cudaStreamCreate(stream: *cudaStream_t) cudaError_t;
pub extern "c" fn cudaStreamDestroy(stream: cudaStream_t) cudaError_t;
pub extern "c" fn cudaStreamSynchronize(stream: cudaStream_t) cudaError_t;
pub extern "c" fn cudaEventCreate(event: *cudaEvent_t) cudaError_t;
pub extern "c" fn cudaEventDestroy(event: cudaEvent_t) cudaError_t;
pub extern "c" fn cudaEventRecord(event: cudaEvent_t, stream: cudaStream_t) cudaError_t;
pub extern "c" fn cudaEventSynchronize(event: cudaEvent_t) cudaError_t;
pub extern "c" fn cudaEventElapsedTime(ms: *f32, start: cudaEvent_t, end: cudaEvent_t) cudaError_t;
pub extern "c" fn cudaDeviceSynchronize() cudaError_t;
pub extern "c" fn cudaGetDeviceCount(count: *c_int) cudaError_t;
pub extern "c" fn cudaSetDevice(device: c_int) cudaError_t;
pub extern "c" fn cudaGetDeviceProperties(prop: *cudaDeviceProp, device: c_int) cudaError_t;
pub extern "c" fn cudaLaunchKernel(func: *const anyopaque, gridDim: dim3, blockDim: dim3, args: [*]*anyopaque, sharedMem: usize, stream: cudaStream_t) cudaError_t;
pub extern "c" fn cudaOccupancyMaxActiveBlocksPerMultiprocessor(numBlocks: *c_int, func: *const anyopaque, blockSize: c_int, dynamicSMemSize: usize) cudaError_t;

pub const cudaDeviceProp = extern struct {
    name: [256]u8,
    totalGlobalMem: usize,
    sharedMemPerBlock: usize,
    regsPerBlock: c_int,
    warpSize: c_int,
    memPitch: usize,
    maxThreadsPerBlock: c_int,
    maxThreadsDim: [3]c_int,
    maxGridSize: [3]c_int,
    clockRate: c_int,
    totalConstMem: usize,
    major: c_int,
    minor: c_int,
    textureAlignment: usize,
    texturePitchAlignment: usize,
    deviceOverlap: c_int,
    multiProcessorCount: c_int,
    kernelExecTimeoutEnabled: c_int,
    integrated: c_int,
    canMapHostMemory: c_int,
    computeMode: c_int,
    maxTexture1D: c_int,
    maxTexture1DMipmap: c_int,
    maxTexture1DLinear: c_int,
    maxTexture2D: [2]c_int,
    maxTexture2DMipmap: [2]c_int,
    maxTexture2DLinear: [3]c_int,
    maxTexture2DGather: [2]c_int,
    maxTexture3D: [3]c_int,
    maxTexture3DAlt: [3]c_int,
    maxTextureCubemap: c_int,
    maxTexture1DLayered: [2]c_int,
    maxTexture2DLayered: [3]c_int,
    maxTextureCubemapLayered: [2]c_int,
    maxSurface1D: c_int,
    maxSurface2D: [2]c_int,
    maxSurface3D: [3]c_int,
    maxSurface1DLayered: [2]c_int,
    maxSurface2DLayered: [3]c_int,
    maxSurfaceCubemap: c_int,
    maxSurfaceCubemapLayered: [2]c_int,
    surfaceAlignment: usize,
    concurrentKernels: c_int,
    ECCEnabled: c_int,
    pciBusID: c_int,
    pciDeviceID: c_int,
    pciDomainID: c_int,
    tccDriver: c_int,
    asyncEngineCount: c_int,
    unifiedAddressing: c_int,
    memoryClockRate: c_int,
    memoryBusWidth: c_int,
    l2CacheSize: c_int,
    persistingL2CacheMaxSize: c_int,
    maxThreadsPerMultiProcessor: c_int,
    streamPrioritiesSupported: c_int,
    globalL1CacheSupported: c_int,
    localL1CacheSupported: c_int,
    sharedMemPerMultiprocessor: usize,
    regsPerMultiprocessor: c_int,
    managedMemory: c_int,
    isMultiGpuBoard: c_int,
    multiGpuBoardGroupID: c_int,
    hostNativeAtomicSupported: c_int,
    singleToDoublePrecisionPerfRatio: c_int,
    pageableMemoryAccess: c_int,
    concurrentManagedAccess: c_int,
    computePreemptionSupported: c_int,
    canUseHostPointerForRegisteredMem: c_int,
    cooperativeLaunch: c_int,
    cooperativeMultiDeviceLaunch: c_int,
    sharedMemPerBlockOptin: usize,
    pageableMemoryAccessUsesHostPageTables: c_int,
    directManagedMemAccessFromHost: c_int,
    maxBlocksPerMultiProcessor: c_int,
    accessPolicyMaxWindowSize: c_int,
    reservedSharedMemPerBlock: usize,
};

pub const dim3 = extern struct {
    x: c_uint,
    y: c_uint,
    z: c_uint,
    
    pub fn init(x: c_uint, y: c_uint, z: c_uint) dim3 {
        return dim3{ .x = x, .y = y, .z = z };
    }
};

pub fn cudaInit() !void {
    var count: c_int = 0;
    const err = cudaGetDeviceCount(&count);
    if (err != cudaSuccess) {
        return error.CudaInitFailed;
    }
    if (count < 8) {
        return error.InsufficientDevices;
    }
}

pub fn cudaCleanup() void {
    _ = cudaDeviceSynchronize();
}

pub fn cudaMallocHost(size: usize) ![]u8 {
    var ptr: *anyopaque = undefined;
    const err = cudaMallocHost(&ptr, size);
    if (err != cudaSuccess) {
        return error.CudaMallocFailed;
    }
    return @as([*]u8, @ptrCast(ptr))[0..size];
}

pub fn cudaFreeHost(ptr: []u8) void {
    _ = cudaFreeHost(ptr.ptr);
}

pub fn cudaMalloc(size: usize) ![]u8 {
    var ptr: *anyopaque = undefined;
    const err = cudaMalloc(&ptr, size);
    if (err != cudaSuccess) {
        return error.CudaMallocFailed;
    }
    return @as([*]u8, @ptrCast(ptr))[0..size];
}

pub fn cudaFree(ptr: []u8) void {
    _ = cudaFree(ptr.ptr);
}

pub fn cudaMemcpy(dst: *anyopaque, src: *const anyopaque, count: usize, kind: cudaMemcpyKind) !void {
    const err = cudaMemcpy(dst, src, count, kind);
    if (err != cudaSuccess) {
        return error.CudaMemcpyFailed;
    }
}

pub fn cudaMemcpyAsync(dst: *anyopaque, src: *const anyopaque, count: usize, kind: cudaMemcpyKind, stream: cudaStream_t) !void {
    const err = cudaMemcpyAsync(dst, src, count, kind, stream);
    if (err != cudaSuccess) {
        return error.CudaMemcpyFailed;
    }
}

pub fn cudaMemcpyPeer(dst: *anyopaque, dstDevice: c_int, src: *const anyopaque, srcDevice: c_int, count: usize) !void {
    const err = cudaMemcpyPeer(dst, dstDevice, src, srcDevice, count);
    if (err != cudaSuccess) {
        return error.CudaMemcpyFailed;
    }
}

pub fn cudaStreamCreate() !cudaStream_t {
    var stream: cudaStream_t = undefined;
    const err = cudaStreamCreate(&stream);
    if (err != cudaSuccess) {
        return error.CudaStreamCreateFailed;
    }
    return stream;
}

pub fn cudaStreamDestroy(stream: cudaStream_t) void {
    _ = cudaStreamDestroy(stream);
}

pub fn cudaStreamSynchronize(stream: cudaStream_t) !void {
    const err = cudaStreamSynchronize(stream);
    if (err != cudaSuccess) {
        return error.CudaStreamSyncFailed;
    }
}

pub fn cudaDeviceSynchronize() !void {
    const err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        return error.CudaDeviceSyncFailed;
    }
}

pub fn cudaSetDevice(device: c_int) !void {
    const err = cudaSetDevice(device);
    if (err != cudaSuccess) {
        return error.CudaSetDeviceFailed;
    }
}

pub fn cudaGetDeviceProperties(device: c_int) !cudaDeviceProp {
    var prop: cudaDeviceProp = undefined;
    const err = cudaGetDeviceProperties(&prop, device);
    if (err != cudaSuccess) {
        return error.CudaGetDevicePropFailed;
    }
    return prop;
}

pub fn cudaLaunchKernel(func: *const anyopaque, gridDim: dim3, blockDim: dim3, args: [*]*anyopaque, sharedMem: usize, stream: cudaStream_t) !void {
    const err = cudaLaunchKernel(func, gridDim, blockDim, args, sharedMem, stream);
    if (err != cudaSuccess) {
        return error.CudaLaunchKernelFailed;
    }
}
