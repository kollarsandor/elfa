const std = @import("std");
const cuda = @import("cuda_bindings.zig");

pub const MemoryPool = struct {
    allocator: std.mem.Allocator,
    host_buffer: []u8,
    device_buffers: [8][]u8,
    offset: usize,
    capacity: usize,
    
    pub fn init(allocator: std.mem.Allocator, capacity: usize) !MemoryPool {
        const host_buffer = try allocator.alloc(u8, capacity);
        errdefer allocator.free(host_buffer);
        
        var device_buffers: [8][]u8 = undefined;
        for (0..8) |i| {
            device_buffers[i] = try cuda.cudaMalloc(capacity);
        }
        
        return MemoryPool{
            .allocator = allocator,
            .host_buffer = host_buffer,
            .device_buffers = device_buffers,
            .offset = 0,
            .capacity = capacity,
        };
    }
    
    pub fn deinit(self: *MemoryPool) void {
        self.allocator.free(self.host_buffer);
        for (0..8) |i| {
            cuda.cudaFree(self.device_buffers[i]);
        }
    }
    
    pub fn alloc(self: *MemoryPool, size: usize, alignment: usize) ![]u8 {
        const aligned_offset = (self.offset + alignment - 1) & ~(alignment - 1);
        if (aligned_offset + size > self.capacity) {
            return error.OutOfMemory;
        }
        const ptr = self.host_buffer[aligned_offset .. aligned_offset + size];
        self.offset = aligned_offset + size;
        return ptr;
    }
    
    pub fn allocDevice(self: *MemoryPool, gpu_id: usize, size: usize, alignment: usize) ![]u8 {
        const aligned_offset = (self.offset + alignment - 1) & ~(alignment - 1);
        if (aligned_offset + size > self.capacity) {
            return error.OutOfMemory;
        }
        const ptr = self.device_buffers[gpu_id][aligned_offset .. aligned_offset + size];
        self.offset = aligned_offset + size;
        return ptr;
    }
    
    pub fn reset(self: *MemoryPool) void {
        self.offset = 0;
    }
    
    pub fn copyHostToDevice(self: *MemoryPool, gpu_id: usize, host_ptr: []const u8, device_ptr: []u8, size: usize) !void {
        try cuda.cudaMemcpy(device_ptr.ptr, host_ptr.ptr, size, cuda.cudaMemcpyHostToDevice);
    }
    
    pub fn copyDeviceToHost(self: *MemoryPool, gpu_id: usize, device_ptr: []const u8, host_ptr: []u8, size: usize) !void {
        try cuda.cudaMemcpy(host_ptr.ptr, device_ptr.ptr, size, cuda.cudaMemcpyDeviceToHost);
    }
    
    pub fn copyDeviceToDevice(self: *MemoryPool, src_gpu: usize, dst_gpu: usize, src_ptr: []const u8, dst_ptr: []u8, size: usize) !void {
        try cuda.cudaMemcpyPeer(dst_ptr.ptr, dst_gpu, src_ptr.ptr, src_gpu, size);
    }
};

pub const PinnedMemoryPool = struct {
    allocator: std.mem.Allocator,
    buffer: []u8,
    offset: usize,
    capacity: usize,
    
    pub fn init(allocator: std.mem.Allocator, capacity: usize) !PinnedMemoryPool {
        const buffer = try cuda.cudaMallocHost(capacity);
        return PinnedMemoryPool{
            .allocator = allocator,
            .buffer = buffer,
            .offset = 0,
            .capacity = capacity,
        };
    }
    
    pub fn deinit(self: *PinnedMemoryPool) void {
        cuda.cudaFreeHost(self.buffer);
    }
    
    pub fn alloc(self: *PinnedMemoryPool, size: usize, alignment: usize) ![]u8 {
        const aligned_offset = (self.offset + alignment - 1) & ~(alignment - 1);
        if (aligned_offset + size > self.capacity) {
            return error.OutOfMemory;
        }
        const ptr = self.buffer[aligned_offset .. aligned_offset + size];
        self.offset = aligned_offset + size;
        return ptr;
    }
    
    pub fn reset(self: *PinnedMemoryPool) void {
        self.offset = 0;
    }
};
