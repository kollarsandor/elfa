const std = @import("std");
const Config = @import("main.zig").Config;
const tokenizer = @import("tokenizer.zig");

pub const StreamingPacker = struct {
    allocator: std.mem.Allocator,
    config: *const Config,
    buffer: std.ArrayList(u32),
    documents: std.ArrayList([]u8),
    current_doc_idx: usize,
    eod_token: u32,
    
    pub fn init(allocator: std.mem.Allocator, config: *const Config) !StreamingPacker {
        const buffer = std.ArrayList(u32).init(allocator);
        const documents = std.ArrayList([]u8).init(allocator);
        
        return StreamingPacker{
            .allocator = allocator,
            .config = config,
            .buffer = buffer,
            .documents = documents,
            .current_doc_idx = 0,
            .eod_token = 4,
        };
    }
    
    pub fn deinit(self: *StreamingPacker) void {
        for (self.documents.items) |doc| {
            self.allocator.free(doc);
        }
        self.documents.deinit();
        self.buffer.deinit();
    }
    
    pub fn addDocument(self: *StreamingPacker, text: []const u8) !void {
        const doc_copy = try self.allocator.dupe(u8, text);
        try self.documents.append(doc_copy);
    }
    
    pub fn loadFromDirectory(self: *StreamingPacker, dir_path: []const u8) !void {
        var dir = try std.fs.cwd().openDir(dir_path, .{ .iterate = true });
        defer dir.close();
        
        var iter = dir.iterate();
        while (try iter.next()) |entry| {
            if (entry.kind == .file) {
                const file_path = try std.fs.path.join(self.allocator, &[_][]const u8{ dir_path, entry.name });
                defer self.allocator.free(file_path);
                
                const file = try std.fs.cwd().openFile(file_path, .{});
                defer file.close();
                
                const content = try file.readToEndAlloc(self.allocator, 1073741824);
                try self.addDocument(content);
                self.allocator.free(content);
            }
        }
    }
    
    pub fn getNextBatch(self: *StreamingPacker) !tokenizer.TokenizedBatch {
        const target_len = self.config.max_seq_len;
        
        self.buffer.clearRetainingCapacity();
        
        while (self.buffer.items.len < target_len and self.current_doc_idx < self.documents.items.len) {
            const doc = self.documents.items[self.current_doc_idx];
            
            var tok = try tokenizer.EntropyTokenizer.init(self.allocator, 100000);
            defer tok.deinit();
            
            const doc_tokens = try tok.encode(doc);
            defer self.allocator.free(doc_tokens);
            
            const remaining_space = target_len - self.buffer.items.len;
            const tokens_to_add = @min(doc_tokens.len, remaining_space);
            
            try self.buffer.appendSlice(doc_tokens[0..tokens_to_add]);
            
            if (self.buffer.items.len < target_len) {
                try self.buffer.append(self.eod_token);
            }
            
            self.current_doc_idx += 1;
        }
        
        const actual_len = self.buffer.items.len;
        const tokens = try self.allocator.dupe(u32, self.buffer.items);
        
        const targets = try self.allocator.alloc(u32, actual_len);
        for (0..actual_len - 1) |i| {
            targets[i] = tokens[i + 1];
        }
        targets[actual_len - 1] = 0;
        
        return tokenizer.TokenizedBatch{
            .tokens = tokens,
            .targets = targets,
            .seq_len = actual_len,
        };
    }
    
    pub fn reset(self: *StreamingPacker) void {
        self.current_doc_idx = 0;
        self.buffer.clearRetainingCapacity();
    }
    
    pub fn getProgress(self: StreamingPacker) f32 {
        if (self.documents.items.len == 0) return 0.0;
        return @as(f32, @floatFromInt(self.current_doc_idx)) / @as(f32, @floatFromInt(self.documents.items.len));
    }
    
    pub fn shuffleDocuments(self: *StreamingPacker, seed: u64) void {
        var rng = std.rand.DefaultPrng.init(seed);
        const random = rng.random();
        
        var i: usize = self.documents.items.len;
        while (i > 1) {
            i -= 1;
            const j = random.intRangeAtMost(usize, 0, i);
            const temp = self.documents.items[i];
            self.documents.items[i] = self.documents.items[j];
            self.documents.items[j] = temp;
        }
    }
    
    pub fn streamFromIterator(self: *StreamingPacker, iterator: *DocumentIterator) !void {
        while (try iterator.next()) |doc| {
            try self.addDocument(doc);
            self.allocator.free(doc);
        }
    }
};

pub const DocumentIterator = struct {
    allocator: std.mem.Allocator,
    file: std.fs.File,
    buffer: [1048576]u8,
    
    pub fn init(allocator: std.mem.Allocator, file_path: []const u8) !DocumentIterator {
        const file = try std.fs.cwd().openFile(file_path, .{});
        return DocumentIterator{
            .allocator = allocator,
            .file = file,
            .buffer = undefined,
        };
    }
    
    pub fn deinit(self: *DocumentIterator) void {
        self.file.close();
    }
    
    pub fn next(self: *DocumentIterator) !?[]u8 {
        const line = try self.file.reader().readUntilDelimiterOrEof(&self.buffer, '\n');
        if (line) |l| {
            return try self.allocator.dupe(u8, l);
        }
        return null;
    }
};

pub const BinaryDataLoader = struct {
    allocator: std.mem.Allocator,
    file: std.fs.File,
    config: *const Config,
    
    pub fn init(allocator: std.mem.Allocator, file_path: []const u8, config: *const Config) !BinaryDataLoader {
        const file = try std.fs.cwd().openFile(file_path, .{});
        return BinaryDataLoader{
            .allocator = allocator,
            .file = file,
            .config = config,
        };
    }
    
    pub fn deinit(self: *BinaryDataLoader) void {
        self.file.close();
    }
    
    pub fn readBatch(self: *BinaryDataLoader) !tokenizer.TokenizedBatch {
        const tokens = try self.allocator.alloc(u32, self.config.max_seq_len);
        errdefer self.allocator.free(tokens);
        
        const reader = self.file.reader();
        const bytes_read = try reader.read(std.mem.sliceAsBytes(tokens));
        const tokens_read = bytes_read / 4;
        
        if (tokens_read == 0) {
            self.allocator.free(tokens);
            return error.EndOfFile;
        }
        
        const targets = try self.allocator.alloc(u32, tokens_read);
        for (0..tokens_read - 1) |i| {
            targets[i] = tokens[i + 1];
        }
        targets[tokens_read - 1] = 0;
        
        return tokenizer.TokenizedBatch{
            .tokens = tokens,
            .targets = targets,
            .seq_len = tokens_read,
        };
    }
    
    pub fn seek(self: *BinaryDataLoader, offset: u64) !void {
        try self.file.seekTo(offset);
    }
    
    pub fn getPosition(self: *BinaryDataLoader) u64 {
        return self.file.getPos() catch 0;
    }
};

pub const DataPackerWriter = struct {
    allocator: std.mem.Allocator,
    file: std.fs.File,
    
    pub fn init(allocator: std.mem.Allocator, file_path: []const u8) !DataPackerWriter {
        const file = try std.fs.cwd().createFile(file_path, .{});
        return DataPackerWriter{
            .allocator = allocator,
            .file = file,
        };
    }
    
    pub fn deinit(self: *DataPackerWriter) void {
        self.file.close();
    }
    
    pub fn writeBatch(self: *DataPackerWriter, tokens: []const u32) !void {
        const writer = self.file.writer();
        try writer.writeAll(std.mem.sliceAsBytes(tokens));
    }
    
    pub fn writeDocument(self: *DataPackerWriter, tokens: []const u32, eod_token: u32) !void {
        const writer = self.file.writer();
        try writer.writeAll(std.mem.sliceAsBytes(tokens));
        try writer.writeInt(u32, eod_token, .little);
    }
};
