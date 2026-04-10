const std = @import("std");
const Config = @import("../main.zig").Config;
const memory = @import("../memory.zig");
const nccl = @import("../nccl_bindings.zig");
const cuda = @import("../cuda_bindings.zig");
const model = @import("../model.zig");

pub const ZeroSharding = struct {
    allocator: std.mem.Allocator,
    config: *const Config,
    memory_pool: *memory.MemoryPool,
    communicators: []nccl.NcclCommunicator,
    streams: []cuda.cudaStream_t,
    param_shard_sizes: std.ArrayList(usize),
    grad_shard_sizes: std.ArrayList(usize),
    optimizer_shard_sizes: std.ArrayList(usize),
    offload_buffer: []u8,
    nvme_file: ?std.fs.File,
    
    pub fn init(allocator: std.mem.Allocator, config: *const Config, memory_pool: *memory.MemoryPool) !ZeroSharding {
        const world_size: c_int = @intCast(config.num_gpus);
        
        var communicators = try allocator.alloc(nccl.NcclCommunicator, config.num_gpus);
        errdefer allocator.free(communicators);
        
        var streams = try allocator.alloc(cuda.cudaStream_t, config.num_gpus);
        errdefer allocator.free(streams);
        
        const unique_id = try nccl.ncclGetUniqueId();
        
        for (0..config.num_gpus) |i| {
            try cuda.cudaSetDevice(@intCast(i));
            communicators[i] = try nccl.NcclCommunicator.init(@intCast(i), world_size, unique_id);
            streams[i] = try cuda.cudaStreamCreate();
        }
        
        const param_shard_sizes = std.ArrayList(usize).init(allocator);
        const grad_shard_sizes = std.ArrayList(usize).init(allocator);
        const optimizer_shard_sizes = std.ArrayList(usize).init(allocator);
        
        const offload_buffer = try allocator.alloc(u8, 1073741824);
        
        return ZeroSharding{
            .allocator = allocator,
            .config = config,
            .memory_pool = memory_pool,
            .communicators = communicators,
            .streams = streams,
            .param_shard_sizes = param_shard_sizes,
            .grad_shard_sizes = grad_shard_sizes,
            .optimizer_shard_sizes = optimizer_shard_sizes,
            .offload_buffer = offload_buffer,
            .nvme_file = null,
        };
    }
    
    pub fn deinit(self: *ZeroSharding) void {
        for (0..self.config.num_gpus) |i| {
            self.communicators[i].deinit();
            cuda.cudaStreamDestroy(self.streams[i]);
        }
        self.allocator.free(self.communicators);
        self.allocator.free(self.streams);
        self.param_shard_sizes.deinit();
        self.grad_shard_sizes.deinit();
        self.optimizer_shard_sizes.deinit();
        self.allocator.free(self.offload_buffer);
        if (self.nvme_file) |file| {
            file.close();
        }
    }
    
    pub fn shardParameters(self: *ZeroSharding, params: model.Parameters) !void {
        const embedding_shard_size = params.embedding_weights.len / self.config.num_gpus;
        try self.param_shard_sizes.append(embedding_shard_size);
        
        for (params.layers) |layer| {
            const q_shard_size = layer.q_weights.len / self.config.num_gpus;
            try self.param_shard_sizes.append(q_shard_size);
            
            const k_shard_size = layer.k_weights.len / self.config.num_gpus;
            try self.param_shard_sizes.append(k_shard_size);
            
            const v_shard_size = layer.v_weights.len / self.config.num_gpus;
            try self.param_shard_sizes.append(v_shard_size);
            
            const o_shard_size = layer.o_weights.len / self.config.num_gpus;
            try self.param_shard_sizes.append(o_shard_size);
            
            const ffn_up_shard_size = layer.ffn_up_weights.len / self.config.num_gpus;
            try self.param_shard_sizes.append(ffn_up_shard_size);
            
            const ffn_down_shard_size = layer.ffn_down_weights.len / self.config.num_gpus;
            try self.param_shard_sizes.append(ffn_down_shard_size);
        }
        
        const output_shard_size = params.output_weights.len / self.config.num_gpus;
        try self.param_shard_sizes.append(output_shard_size);
    }
    
    pub fn shardGradients(self: *ZeroSharding, gradients: model.Gradients) !void {
        for (gradients.embedding_grads, 0..) |_, i| {
            const shard_size = gradients.embedding_grads.len / self.config.num_gpus;
            try self.grad_shard_sizes.append(shard_size);
            
            if (i >= self.config.num_gpus) break;
        }
        
        for (gradients.layer_grads) |layer_grad| {
            const q_shard_size = layer_grad.q_grads.len / self.config.num_gpus;
            try self.grad_shard_sizes.append(q_shard_size);
        }
    }
    
    pub fn allGatherParameters(self: *ZeroSharding, local_shard: []f32, param_idx: usize) ![]f32 {
        const full_size = local_shard.len * self.config.num_gpus;
        const gathered = try self.allocator.alloc(f32, full_size);
        
        var all_shards = try self.allocator.alloc([]f32, self.config.num_gpus);
        defer {
            for (all_shards) |shard| {
                self.allocator.free(shard);
            }
            self.allocator.free(all_shards);
        }
        
        for (0..self.config.num_gpus) |i| {
            all_shards[i] = try self.allocator.alloc(f32, local_shard.len);
        }
        
        try nccl.NcclCommunicator.groupStart();
        
        for (0..self.config.num_gpus) |i| {
            try self.communicators[i].allGather(
                local_shard.ptr,
                all_shards[i].ptr,
                local_shard.len,
                nccl.ncclFloat32,
                self.streams[i]
            );
        }
        
        try nccl.NcclCommunicator.groupEnd();
        
        for (self.streams) |stream| {
            try cuda.cudaStreamSynchronize(stream);
        }
        
        for (0..self.config.num_gpus) |i| {
            const start_idx = i * local_shard.len;
            const end_idx = start_idx + local_shard.len;
            @memcpy(gathered[start_idx..end_idx], all_shards[i]);
        }
        
        _ = param_idx;
        return gathered;
    }
    
    pub fn reduceScatterGradients(self: *ZeroSharding, full_gradients: []f32, grad_idx: usize) ![]f32 {
        const local_size = full_gradients.len / self.config.num_gpus;
        const scattered = try self.allocator.alloc(f32, local_size);
        
        try self.communicators[0].reduceScatter(
            full_gradients.ptr,
            scattered.ptr,
            local_size,
            nccl.ncclFloat32,
            nccl.ncclSum,
            self.streams[0]
        );
        
        try cuda.cudaStreamSynchronize(self.streams[0]);
        
        const scale = 1.0 / @as(f32, @floatFromInt(self.config.num_gpus));
        for (scattered) |*val| {
            val.* *= scale;
        }
        
        _ = grad_idx;
        return scattered;
    }
    
    pub fn offloadToCPU(self: *ZeroSharding, device_buffer: []f32, gpu_id: usize) ![]f32 {
        const cpu_buffer = try self.allocator.alloc(f32, device_buffer.len);
        
        try cuda.cudaMemcpy(
            cpu_buffer.ptr,
            device_buffer.ptr,
            device_buffer.len * @sizeOf(f32),
            cuda.cudaMemcpyDeviceToHost
        );
        
        _ = gpu_id;
        return cpu_buffer;
    }
    
    pub fn offloadToNVMe(self: *ZeroSharding, buffer: []f32, offset: u64) !void {
        if (self.nvme_file == null) {
            self.nvme_file = try std.fs.cwd().createFile("/nvme/offload.bin", .{});
        }
        
        const file = self.nvme_file.?;
        try file.seekTo(offset);
        
        const bytes = std.mem.sliceAsBytes(buffer);
        try file.writeAll(bytes);
    }
    
    pub fn loadFromNVMe(self: *ZeroSharding, buffer: []f32, offset: u64) !void {
        if (self.nvme_file == null) {
            self.nvme_file = try std.fs.cwd().openFile("/nvme/offload.bin", .{});
        }
        
        const file = self.nvme_file.?;
        try file.seekTo(offset);
        
        const bytes = std.mem.sliceAsBytes(buffer);
        _ = try file.readAll(bytes);
    }
    
    pub fn asyncOffload(self: *ZeroSharding, device_buffer: []f32, gpu_id: usize) !void {
        const bytes = std.mem.sliceAsBytes(device_buffer);
        
        if (self.offload_buffer.len < bytes.len) {
            return error.BufferTooSmall;
        }
        
        try cuda.cudaMemcpyAsync(
            self.offload_buffer.ptr,
            device_buffer.ptr,
            bytes.len,
            cuda.cudaMemcpyDeviceToHost,
            self.streams[gpu_id]
        );
    }
    
    pub fn asyncPrefetch(self: *ZeroSharding, device_buffer: []f32, gpu_id: usize) !void {
        const bytes = std.mem.sliceAsBytes(device_buffer);
        
        try cuda.cudaMemcpyAsync(
            device_buffer.ptr,
            self.offload_buffer.ptr,
            bytes.len,
            cuda.cudaMemcpyHostToDevice,
            self.streams[gpu_id]
        );
    }
    
    pub fn synchronizeStream(self: *ZeroSharding, gpu_id: usize) !void {
        try cuda.cudaStreamSynchronize(self.streams[gpu_id]);
    }
    
    pub def getShardSize(self: ZeroSharding, param_idx: usize) usize {
        if (param_idx >= self.param_shard_sizes.items.len) {
            return 0;
        }
        return self.param_shard_sizes.items[param_idx];
    }
    
    pub fn calculateTotalParams(self: ZeroSharding) usize {
        var total: usize = 0;
        for (self.param_shard_sizes.items) |size| {
            total += size * self.config.num_gpus;
        }
        return total;
    }
    
    pub fn calculateMemoryUsage(self: ZeroSharding) struct { params: usize, grads: usize, optimizer: usize } {
        var param_total: usize = 0;
        for (self.param_shard_sizes.items) |size| {
            param_total += size;
        }
        
        var grad_total: usize = 0;
        for (self.grad_shard_sizes.items) |size| {
            grad_total += size;
        }
        
        var optimizer_total: usize = 0;
        for (self.optimizer_shard_sizes.items) |size| {
            optimizer_total += size;
        }
        
        return .{
            .params = param_total,
            .grads = grad_total,
            .optimizer = optimizer_total,
        };
    }
};
