const std = @import("std");
const Config = @import("main.zig").Config;
const cuda = @import("cuda_bindings.zig");

pub const AsyncIO = struct {
    allocator: std.mem.Allocator,
    read_queue: std.ArrayList(IORequest),
    write_queue: std.ArrayList(IORequest),
    streams: []cuda.cudaStream_t,
    pinned_buffers: std.ArrayList([]u8),
    
    const IORequest = struct {
        buffer: []u8,
        file_path: []const u8,
        offset: u64,
        size: usize,
        callback: ?*const fn ([]u8) void,
    };
    
    pub fn init(allocator: std.mem.Allocator, num_streams: usize) !AsyncIO {
        const read_queue = std.ArrayList(IORequest).init(allocator);
        const write_queue = std.ArrayList(IORequest).init(allocator);
        
        const streams = try allocator.alloc(cuda.cudaStream_t, num_streams);
        errdefer allocator.free(streams);
        
        for (0..num_streams) |i| {
            streams[i] = try cuda.cudaStreamCreate();
        }
        
        const pinned_buffers = std.ArrayList([]u8).init(allocator);
        
        return AsyncIO{
            .allocator = allocator,
            .read_queue = read_queue,
            .write_queue = write_queue,
            .streams = streams,
            .pinned_buffers = pinned_buffers,
        };
    }
    
    pub fn deinit(self: *AsyncIO) void {
        for (self.streams) |stream| {
            cuda.cudaStreamDestroy(stream);
        }
        self.allocator.free(self.streams);
        
        for (self.pinned_buffers.items) |buf| {
            cuda.cudaFreeHost(buf);
        }
        self.pinned_buffers.deinit();
        
        self.read_queue.deinit();
        self.write_queue.deinit();
    }
    
    pub fn allocatePinnedBuffer(self: *AsyncIO, size: usize) ![]u8 {
        const buffer = try cuda.cudaMallocHost(size);
        try self.pinned_buffers.append(buffer);
        return buffer;
    }
    
    pub fn queueRead(self: *AsyncIO, file_path: []const u8, offset: u64, size: usize, callback: ?*const fn ([]u8) void) ![]u8 {
        const buffer = try self.allocator.alloc(u8, size);
        
        const request = IORequest{
            .buffer = buffer,
            .file_path = try self.allocator.dupe(u8, file_path),
            .offset = offset,
            .size = size,
            .callback = callback,
        };
        
        try self.read_queue.append(request);
        return buffer;
    }
    
    pub fn queueWrite(self: *AsyncIO, buffer: []const u8, file_path: []const u8, offset: u64, callback: ?*const fn () void) !void {
        _ = callback;
        
        const buffer_copy = try self.allocator.dupe(u8, buffer);
        
        const request = IORequest{
            .buffer = buffer_copy,
            .file_path = try self.allocator.dupe(u8, file_path),
            .offset = offset,
            .size = buffer.len,
            .callback = null,
        };
        
        try self.write_queue.append(request);
    }
    
    pub fn processReads(self: *AsyncIO) !void {
        for (self.read_queue.items) |request| {
            const file = try std.fs.cwd().openFile(request.file_path, .{});
            defer file.close();
            
            try file.seekTo(request.offset);
            _ = try file.readAll(request.buffer);
            
            if (request.callback) |callback| {
                callback(request.buffer);
            }
            
            self.allocator.free(request.file_path);
        }
        
        self.read_queue.clearRetainingCapacity();
    }
    
    pub fn processWrites(self: *AsyncIO) !void {
        for (self.write_queue.items) |request| {
            const file = try std.fs.cwd().openFile(request.file_path, .{});
            defer file.close();
            
            try file.seekTo(request.offset);
            try file.writeAll(request.buffer);
            
            self.allocator.free(request.buffer);
            self.allocator.free(request.file_path);
        }
        
        self.write_queue.clearRetainingCapacity();
    }
    
    pub fn synchronize(self: *AsyncIO) !void {
        for (self.streams) |stream| {
            try cuda.cudaStreamSynchronize(stream);
        }
    }
    
    pub fn prefetchToDevice(self: *AsyncIO, host_buffer: []const u8, device_buffer: []u8, stream_idx: usize) !void {
        try cuda.cudaMemcpyAsync(
            device_buffer.ptr,
            host_buffer.ptr,
            host_buffer.len,
            cuda.cudaMemcpyHostToDevice,
            self.streams[stream_idx]
        );
    }
    
    pub fn offloadToHost(self: *AsyncIO, device_buffer: []const u8, host_buffer: []u8, stream_idx: usize) !void {
        try cuda.cudaMemcpyAsync(
            host_buffer.ptr,
            device_buffer.ptr,
            device_buffer.len,
            cuda.cudaMemcpyDeviceToHost,
            self.streams[stream_idx]
        );
    }
};

pub const MemoryMappedFile = struct {
    allocator: std.mem.Allocator,
    file: std.fs.File,
    data: []align(std.mem.page_size) u8,
    size: usize,
    
    pub fn init(allocator: std.mem.Allocator, path: []const u8, size: usize) !MemoryMappedFile {
        const file = try std.fs.cwd().createFile(path, .{ .read = true });
        
        try file.seekTo(size - 1);
        try file.writeAll(&[_]u8{0});
        try file.seekTo(0);
        
        const data = try std.posix.mmap(
            null,
            size,
            std.posix.PROT.READ | std.posix.PROT.WRITE,
            std.posix.MAP.SHARED,
            file.handle,
            0
        );
        
        return MemoryMappedFile{
            .allocator = allocator,
            .file = file,
            .data = data,
            .size = size,
        };
    }
    
    pub fn deinit(self: *MemoryMappedFile) void {
        std.posix.munmap(self.data);
        self.file.close();
    }
    
    pub fn read(self: MemoryMappedFile, offset: usize, buffer: []u8) !void {
        if (offset + buffer.len > self.size) {
            return error.OutOfBounds;
        }
        @memcpy(buffer, self.data[offset .. offset + buffer.len]);
    }
    
    pub fn write(self: MemoryMappedFile, offset: usize, buffer: []const u8) !void {
        if (offset + buffer.len > self.size) {
            return error.OutOfBounds;
        }
        @memcpy(self.data[offset .. offset + buffer.len], buffer);
    }
    
    pub fn sync(self: MemoryMappedFile) !void {
        try std.posix.msync(self.data, std.posix.MS_SYNC);
    }
};

pub const CheckpointManager = struct {
    allocator: std.mem.Allocator,
    checkpoint_dir: []const u8,
    max_checkpoints: usize,
    checkpoint_history: std.ArrayList(usize),
    
    pub fn init(allocator: std.mem.Allocator, checkpoint_dir: []const u8, max_checkpoints: usize) !CheckpointManager {
        try std.fs.cwd().makePath(checkpoint_dir);
        
        const checkpoint_history = std.ArrayList(usize).init(allocator);
        
        return CheckpointManager{
            .allocator = allocator,
            .checkpoint_dir = checkpoint_dir,
            .max_checkpoints = max_checkpoints,
            .checkpoint_history = checkpoint_history,
        };
    }
    
    pub fn deinit(self: *CheckpointManager) void {
        self.checkpoint_history.deinit();
    }
    
    pub fn saveCheckpoint(self: *CheckpointManager, step: usize, data: []const u8) !void {
        const checkpoint_path = try std.fmt.allocPrint(self.allocator, "{s}/checkpoint_{d}.bin", .{ self.checkpoint_dir, step });
        defer self.allocator.free(checkpoint_path);
        
        const file = try std.fs.cwd().createFile(checkpoint_path, .{});
        defer file.close();
        
        try file.writeAll(data);
        
        try self.checkpoint_history.append(step);
        
        if (self.checkpoint_history.items.len > self.max_checkpoints) {
            const oldest_step = self.checkpoint_history.items[0];
            const old_path = try std.fmt.allocPrint(self.allocator, "{s}/checkpoint_{d}.bin", .{ self.checkpoint_dir, oldest_step });
            defer self.allocator.free(old_path);
            
            std.fs.cwd().deleteFile(old_path) catch {};
            _ = self.checkpoint_history.orderedRemove(0);
        }
    }
    
    pub fn loadCheckpoint(self: CheckpointManager, step: usize) ![]u8 {
        const checkpoint_path = try std.fmt.allocPrint(self.allocator, "{s}/checkpoint_{d}.bin", .{ self.checkpoint_dir, step });
        defer self.allocator.free(checkpoint_path);
        
        const file = try std.fs.cwd().openFile(checkpoint_path, .{});
        defer file.close();
        
        return try file.readToEndAlloc(self.allocator, 107374182400);
    }
    
    pub fn getLatestCheckpoint(self: CheckpointManager) ?usize {
        if (self.checkpoint_history.items.len == 0) {
            return null;
        }
        return self.checkpoint_history.items[self.checkpoint_history.items.len - 1];
    }
    
    pub fn listCheckpoints(self: CheckpointManager) ![]usize {
        return try self.allocator.dupe(usize, self.checkpoint_history.items);
    }
};
