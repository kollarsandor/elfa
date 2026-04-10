const std = @import("std");
const Config = @import("../main.zig").Config;
const nccl = @import("../nccl_bindings.zig");
const cuda = @import("../cuda_bindings.zig");

pub const TensorParallel = struct {
    allocator: std.mem.Allocator,
    config: *const Config,
    communicators: []nccl.NcclCommunicator,
    streams: []cuda.cudaStream_t,
    rank: c_int,
    world_size: c_int,
    
    pub fn init(allocator: std.mem.Allocator, config: *const Config) !TensorParallel {
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
        
        return TensorParallel{
            .allocator = allocator,
            .config = config,
            .communicators = communicators,
            .streams = streams,
            .rank = 0,
            .world_size = world_size,
        };
    }
    
    pub fn deinit(self: *TensorParallel) void {
        for (0..self.config.num_gpus) |i| {
            self.communicators[i].deinit();
            cuda.cudaStreamDestroy(self.streams[i]);
        }
        self.allocator.free(self.communicators);
        self.allocator.free(self.streams);
    }
    
    pub fn shardTensor(self: *TensorParallel, tensor: []f32, dim: usize) ![]f32 {
        const local_size = tensor.len / self.config.num_gpus;
        const shard_size = local_size;
        
        var local_shards = try self.allocator.alloc([]f32, self.config.num_gpus);
        defer self.allocator.free(local_shards);
        
        for (0..self.config.num_gpus) |i| {
            local_shards[i] = try self.allocator.alloc(f32, shard_size);
            
            const start_idx = i * shard_size;
            const end_idx = start_idx + shard_size;
            
            if (dim == 0) {
                @memcpy(local_shards[i], tensor[start_idx..end_idx]);
            } else {
                for (0..shard_size) |j| {
                    const src_idx = j * self.config.num_gpus + i;
                    local_shards[i][j] = tensor[src_idx];
                }
            }
        }
        
        return local_shards[0];
    }
    
    pub fn gatherTensor(self: *TensorParallel, local_shard: []f32, dim: usize) ![]f32 {
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
            if (i == 0) {
                try self.communicators[i].allGather(
                    local_shard.ptr,
                    all_shards[i].ptr,
                    local_shard.len,
                    nccl.ncclFloat32,
                    self.streams[i]
                );
            }
        }
        
        try nccl.NcclCommunicator.groupEnd();
        
        for (self.streams) |stream| {
            try cuda.cudaStreamSynchronize(stream);
        }
        
        if (dim == 0) {
            for (0..self.config.num_gpus) |i| {
                const start_idx = i * local_shard.len;
                const end_idx = start_idx + local_shard.len;
                @memcpy(gathered[start_idx..end_idx], all_shards[i]);
            }
        } else {
            for (0..local_shard.len) |j| {
                for (0..self.config.num_gpus) |i| {
                    const dst_idx = j * self.config.num_gpus + i;
                    gathered[dst_idx] = all_shards[i][j];
                }
            }
        }
        
        return gathered;
    }
    
    pub fn allReduceGradients(self: *TensorParallel, gradients: []f32) !void {
        try nccl.NcclCommunicator.groupStart();
        
        for (0..self.config.num_gpus) |i| {
            const grad_slice = gradients[i * gradients.len / self.config.num_gpus .. (i + 1) * gradients.len / self.config.num_gpus];
            
            try self.communicators[i].allReduce(
                grad_slice.ptr,
                grad_slice.ptr,
                grad_slice.len,
                nccl.ncclFloat32,
                nccl.ncclSum,
                self.streams[i]
            );
        }
        
        try nccl.NcclCommunicator.groupEnd();
        
        for (self.streams) |stream| {
            try cuda.cudaStreamSynchronize(stream);
        }
    }
    
    pub fn allReduceTensor(self: *TensorParallel, tensor: []f32) !void {
        try self.communicators[0].allReduce(
            tensor.ptr,
            tensor.ptr,
            tensor.len,
            nccl.ncclFloat32,
            nccl.ncclSum,
            self.streams[0]
        );
        
        try cuda.cudaStreamSynchronize(self.streams[0]);
        
        const scale = 1.0 / @as(f32, @floatFromInt(self.world_size));
        for (tensor) |*val| {
            val.* *= scale;
        }
    }
    
    pub fn broadcastParameter(self: *TensorParallel, param: []f32, root: c_int) !void {
        try self.communicators[0].broadcast(
            param.ptr,
            param.ptr,
            param.len,
            nccl.ncclFloat32,
            root,
            self.streams[0]
        );
        
        try cuda.cudaStreamSynchronize(self.streams[0]);
    }
    
    pub fn syncAll(self: *TensorParallel) !void {
        for (self.streams) |stream| {
            try cuda.cudaStreamSynchronize(stream);
        }
    }
    
    pub fn getLocalShardSize(self: TensorParallel, global_size: usize) usize {
        return global_size / self.config.num_gpus;
    }
    
    pub fn getGlobalIndex(self: TensorParallel, local_idx: usize) usize {
        return local_idx * self.config.num_gpus + @as(usize, @intCast(self.rank));
    }
    
    pub fn getLocalIndex(self: TensorParallel, global_idx: usize) usize {
        return global_idx / self.config.num_gpus;
    }
};
