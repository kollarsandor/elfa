const std = @import("std");
const Config = @import("../main.zig").Config;
const nccl = @import("../nccl_bindings.zig");
const cuda = @import("../cuda_bindings.zig");

pub const SequenceParallel = struct {
    allocator: std.mem.Allocator,
    config: *const Config,
    communicators: []nccl.NcclCommunicator,
    streams: []cuda.cudaStream_t,
    rank: c_int,
    world_size: c_int,
    local_seq_len: usize,
    
    pub fn init(allocator: std.mem.Allocator, config: *const Config) !SequenceParallel {
        const world_size: c_int = @intCast(config.num_gpus);
        const local_seq_len = config.max_seq_len / config.num_gpus;
        
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
        
        return SequenceParallel{
            .allocator = allocator,
            .config = config,
            .communicators = communicators,
            .streams = streams,
            .rank = 0,
            .world_size = world_size,
            .local_seq_len = local_seq_len,
        };
    }
    
    pub fn deinit(self: *SequenceParallel) void {
        for (0..self.config.num_gpus) |i| {
            self.communicators[i].deinit();
            cuda.cudaStreamDestroy(self.streams[i]);
        }
        self.allocator.free(self.communicators);
        self.allocator.free(self.streams);
    }
    
    pub fn splitSequence(self: *SequenceParallel, full_sequence: []f32) ![][]f32 {
        const hidden_dim = self.config.hidden_dim;
        const tokens_per_gpu = self.local_seq_len;
        
        var shards = try self.allocator.alloc([]f32, self.config.num_gpus);
        
        for (0..self.config.num_gpus) |i| {
            const start_token = i * tokens_per_gpu;
            const end_token = start_token + tokens_per_gpu;
            const shard_size = tokens_per_gpu * hidden_dim;
            
            shards[i] = try self.allocator.alloc(f32, shard_size);
            
            for (start_token..end_token) |t| {
                const src_start = t * hidden_dim;
                const src_end = src_start + hidden_dim;
                const dst_start = (t - start_token) * hidden_dim;
                
                @memcpy(shards[i][dst_start..dst_start + hidden_dim], full_sequence[src_start..src_end]);
            }
        }
        
        return shards;
    }
    
    pub fn gatherSequence(self: *SequenceParallel, local_shard: []f32) ![]f32 {
        const hidden_dim = self.config.hidden_dim;
        const full_seq_len = self.config.max_seq_len;
        const gathered = try self.allocator.alloc(f32, full_seq_len * hidden_dim);
        
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
        
        for (0..self.config.num_gpus) |i| {
            const start_token = i * self.local_seq_len;
            
            for (0..self.local_seq_len) |t| {
                const src_start = t * hidden_dim;
                const dst_start = (start_token + t) * hidden_dim;
                
                @memcpy(gathered[dst_start..dst_start + hidden_dim], all_shards[i][src_start..src_start + hidden_dim]);
            }
        }
        
        return gathered;
    }
    
    pub fn ringExchange(self: *SequenceParallel, local_activation: []f32, step: usize) ![]f32 {
        const next_rank = (@as(usize, @intCast(self.rank)) + step) % self.config.num_gpus;
        const prev_rank = (@as(usize, @intCast(self.rank)) + self.config.num_gpus - step) % self.config.num_gpus;
        
        const received = try self.allocator.alloc(f32, local_activation.len);
        
        try nccl.NcclCommunicator.groupStart();
        
        try self.communicators[@intCast(self.rank)].send(
            local_activation.ptr,
            local_activation.len,
            nccl.ncclFloat32,
            @intCast(next_rank),
            self.streams[@intCast(self.rank)]
        );
        
        try self.communicators[@intCast(self.rank)].recv(
            received.ptr,
            received.len,
            nccl.ncclFloat32,
            @intCast(prev_rank),
            self.streams[@intCast(self.rank)]
        );
        
        try nccl.NcclCommunicator.groupEnd();
        
        try cuda.cudaStreamSynchronize(self.streams[@intCast(self.rank)]);
        
        return received;
    }
    
    pub fn allReduceGradients(self: *SequenceParallel, gradients: []f32) !void {
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
    
    pub fn reduceScatterGradients(self: *SequenceParallel, full_gradients: []f32) ![]f32 {
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
        
        return scattered;
    }
    
    pub fn syncCrossBoundary(self: *SequenceParallel, activations: []f32, boundary_size: usize) !void {
        const hidden_dim = self.config.hidden_dim;
        const boundary_data_size = boundary_size * hidden_dim;
        
        var send_buffer = try self.allocator.alloc(f32, boundary_data_size * 2);
        defer self.allocator.free(send_buffer);
        
        var recv_buffer = try self.allocator.alloc(f32, boundary_data_size * 2);
        defer self.allocator.free(recv_buffer);
        
        const local_start = activations.len - boundary_data_size;
        @memcpy(send_buffer[0..boundary_data_size], activations[local_start..]);
        @memcpy(send_buffer[boundary_data_size..], activations[0..boundary_data_size]);
        
        const left_neighbor = (@as(usize, @intCast(self.rank)) + self.config.num_gpus - 1) % self.config.num_gpus;
        const right_neighbor = (@as(usize, @intCast(self.rank)) + 1) % self.config.num_gpus;
        
        try nccl.NcclCommunicator.groupStart();
        
        try self.communicators[@intCast(self.rank)].send(
            send_buffer.ptr,
            boundary_data_size,
            nccl.ncclFloat32,
            @intCast(left_neighbor),
            self.streams[@intCast(self.rank)]
        );
        
        try self.communicators[@intCast(self.rank)].recv(
            recv_buffer.ptr,
            boundary_data_size,
            nccl.ncclFloat32,
            @intCast(right_neighbor),
            self.streams[@intCast(self.rank)]
        );
        
        try nccl.NcclCommunicator.groupEnd();
        
        try cuda.cudaStreamSynchronize(self.streams[@intCast(self.rank)]);
    }
    
    pub fn getLocalSeqRange(self: SequenceParallel) struct { start: usize, end: usize } {
        const start = @as(usize, @intCast(self.rank)) * self.local_seq_len;
        const end = start + self.local_seq_len;
        return .{ .start = start, .end = end };
    }
    
    pub fn getGlobalSeqIdx(self: SequenceParallel, local_idx: usize) usize {
        return @as(usize, @intCast(self.rank)) * self.local_seq_len + local_idx;
    }
};
