const std = @import("std");

pub const EntropyTokenizer = struct {
    allocator: std.mem.Allocator,
    vocab_size: usize,
    vocab: std.StringHashMap(u32),
    merges: std.ArrayList(Merge),
    special_tokens: SpecialTokens,
    
    const Merge = struct {
        first: []u8,
        second: []u8,
        priority: u32,
    };
    
    const SpecialTokens = struct {
        pad: u32 = 0,
        unk: u32 = 1,
        bos: u32 = 2,
        eos: u32 = 3,
        eod: u32 = 4,
        mask: u32 = 5,
    };
    
    pub fn init(allocator: std.mem.Allocator, vocab_size: usize) !EntropyTokenizer {
        var vocab = std.StringHashMap(u32).init(allocator);
        var merges = std.ArrayList(Merge).init(allocator);
        
        const special_tokens = SpecialTokens{};
        
        try vocab.put("<pad>", special_tokens.pad);
        try vocab.put("<unk>", special_tokens.unk);
        try vocab.put("<bos>", special_tokens.bos);
        try vocab.put("<eos>", special_tokens.eos);
        try vocab.put("<eod>", special_tokens.eod);
        try vocab.put("<mask>", special_tokens.mask);
        
        var byte_tokens: usize = 6;
        var i: u8 = 0;
        while (byte_tokens < 300 and i < 256) : (i += 1) {
            const byte_str = try std.fmt.allocPrint(allocator, "<byte_{d}>", .{i});
            try vocab.put(byte_str, @as(u32, @intCast(byte_tokens)));
            byte_tokens += 1;
        }
        
        return EntropyTokenizer{
            .allocator = allocator,
            .vocab_size = vocab_size,
            .vocab = vocab,
            .merges = merges,
            .special_tokens = special_tokens,
        };
    }
    
    pub fn deinit(self: *EntropyTokenizer) void {
        var iter = self.vocab.keyIterator();
        while (iter.next()) |key| {
            self.allocator.free(key.*);
        }
        self.vocab.deinit();
        
        for (self.merges.items) |merge| {
            self.allocator.free(merge.first);
            self.allocator.free(merge.second);
        }
        self.merges.deinit();
    }
    
    pub fn encode(self: *EntropyTokenizer, text: []const u8) ![]u32 {
        var tokens = std.ArrayList(u32).init(self.allocator);
        errdefer tokens.deinit();
        
        try tokens.append(self.special_tokens.bos);
        
        var words = std.ArrayList([]u8).init(self.allocator);
        defer {
            for (words.items) |word| {
                self.allocator.free(word);
            }
            words.deinit();
        }
        
        var word_start: usize = 0;
        for (text, 0..) |char, i| {
            if (char == ' ' or char == '\n' or char == '\t' or char == '.' or char == ',' or char == '!' or char == '?' or char == ';' or char == ':') {
                if (i > word_start) {
                    const word = try self.allocator.dupe(u8, text[word_start..i]);
                    try words.append(word);
                }
                const punct = try self.allocator.dupe(u8, text[i..i+1]);
                try words.append(punct);
                word_start = i + 1;
            }
        }
        if (word_start < text.len) {
            const word = try self.allocator.dupe(u8, text[word_start..]);
            try words.append(word);
        }
        
        for (words.items) |word| {
            const word_tokens = try self.encodeWord(word);
            defer self.allocator.free(word_tokens);
            try tokens.appendSlice(word_tokens);
        }
        
        try tokens.append(self.special_tokens.eos);
        
        return tokens.toOwnedSlice();
    }
    
    fn encodeWord(self: *EntropyTokenizer, word: []const u8) ![]u32 {
        var tokens = std.ArrayList(u32).init(self.allocator);
        errdefer tokens.deinit();
        
        var subwords = std.ArrayList([]u8).init(self.allocator);
        defer {
            for (subwords.items) |sw| {
                self.allocator.free(sw);
            }
            subwords.deinit();
        }
        
        for (word) |byte| {
            const byte_str = try std.fmt.allocPrint(self.allocator, "<byte_{d}>", .{byte});
            try subwords.append(byte_str);
        }
        
        while (subwords.items.len > 1) {
            var best_merge_idx: ?usize = null;
            var best_priority: u32 = 0;
            
            for (self.merges.items, 0..) |merge, i| {
                for (0..subwords.items.len - 1) |j| {
                    if (std.mem.eql(u8, subwords.items[j], merge.first) and
                        std.mem.eql(u8, subwords.items[j + 1], merge.second)) {
                        if (merge.priority > best_priority) {
                            best_priority = merge.priority;
                            best_merge_idx = i;
                        }
                    }
                }
            }
            
            if (best_merge_idx) |idx| {
                const merge = self.merges.items[idx];
                var new_subwords = std.ArrayList([]u8).init(self.allocator);
                
                var i: usize = 0;
                while (i < subwords.items.len) {
                    if (i < subwords.items.len - 1 and
                        std.mem.eql(u8, subwords.items[i], merge.first) and
                        std.mem.eql(u8, subwords.items[i + 1], merge.second)) {
                        const merged = try std.mem.concat(self.allocator, u8, &[_][]const u8{ merge.first, merge.second });
                        try new_subwords.append(merged);
                        self.allocator.free(subwords.items[i]);
                        self.allocator.free(subwords.items[i + 1]);
                        i += 2;
                    } else {
                        try new_subwords.append(subwords.items[i]);
                        i += 1;
                    }
                }
                
                subwords.deinit();
                subwords = new_subwords;
            } else {
                break;
            }
        }
        
        for (subwords.items) |subword| {
            if (self.vocab.get(subword)) |token_id| {
                try tokens.append(token_id);
            } else {
                try tokens.append(self.special_tokens.unk);
            }
        }
        
        return tokens.toOwnedSlice();
    }
    
    pub fn decode(self: *EntropyTokenizer, tokens: []const u32) ![]u8 {
        var result = std.ArrayList(u8).init(self.allocator);
        errdefer result.deinit();
        
        var vocab_reverse = std.AutoHashMap(u32, []u8).init(self.allocator);
        defer vocab_reverse.deinit();
        
        var iter = self.vocab.iterator();
        while (iter.next()) |entry| {
            try vocab_reverse.put(entry.value_ptr.*, entry.key_ptr.*);
        }
        
        for (tokens) |token| {
            if (token == self.special_tokens.bos or
                token == self.special_tokens.eos or
                token == self.special_tokens.pad) {
                continue;
            }
            
            if (vocab_reverse.get(token)) |token_str| {
                if (std.mem.startsWith(u8, token_str, "<byte_")) {
                    const byte_val = try std.fmt.parseInt(u8, token_str[6..token_str.len-1], 10);
                    try result.append(byte_val);
                } else {
                    try result.appendSlice(token_str);
                }
            }
        }
        
        return result.toOwnedSlice();
    }
    
    pub fn train(self: *EntropyTokenizer, texts: [][]const u8, num_merges: usize) !void {
        var pair_counts = std.StringHashMap(u32).init(self.allocator);
        defer pair_counts.deinit();
        
        var merge_iter: usize = 0;
        while (merge_iter < num_merges and self.vocab.count() < self.vocab_size) : (merge_iter += 1) {
            pair_counts.clearRetainingCapacity();
            
            for (texts) |text| {
                const tokens = try self.encode(text);
                defer self.allocator.free(tokens);
                
                for (0..tokens.len - 1) |i| {
                    const pair = try std.fmt.allocPrint(self.allocator, "{d}_{d}", .{ tokens[i], tokens[i + 1] });
                    defer self.allocator.free(pair);
                    
                    const count = pair_counts.get(pair) orelse 0;
                    try pair_counts.put(pair, count + 1);
                }
            }
            
            var best_pair: ?[]u8 = null;
            var best_count: u32 = 0;
            
            var iter = pair_counts.iterator();
            while (iter.next()) |entry| {
                if (entry.value_ptr.* > best_count) {
                    best_count = entry.value_ptr.*;
                    best_pair = entry.key_ptr.*;
                }
            }
            
            if (best_pair) |pair| {
                var parts = std.mem.split(u8, pair, "_");
                const first_id = try std.fmt.parseInt(u32, parts.next().?, 10);
                const second_id = try std.fmt.parseInt(u32, parts.next().?, 10);
                
                var vocab_reverse = std.AutoHashMap(u32, []u8).init(self.allocator);
                defer vocab_reverse.deinit();
                
                var vocab_iter = self.vocab.iterator();
                while (vocab_iter.next()) |entry| {
                    try vocab_reverse.put(entry.value_ptr.*, entry.key_ptr.*);
                }
                
                const first_str = vocab_reverse.get(first_id) orelse "<unk>";
                const second_str = vocab_reverse.get(second_id) orelse "<unk>";
                
                const merged_str = try std.mem.concat(self.allocator, u8, &[_][]const u8{ first_str, second_str });
                const new_id: u32 = @intCast(self.vocab.count());
                
                try self.vocab.put(merged_str, new_id);
                
                const first_copy = try self.allocator.dupe(u8, first_str);
                const second_copy = try self.allocator.dupe(u8, second_str);
                try self.merges.append(Merge{
                    .first = first_copy,
                    .second = second_copy,
                    .priority = @intCast(self.merges.items.len),
                });
            }
        }
    }
    
    pub fn save(self: *EntropyTokenizer, path: []const u8) !void {
        const file = try std.fs.cwd().createFile(path, .{});
        defer file.close();
        
        const writer = file.writer();
        
        try writer.writeInt(u32, @intCast(self.vocab.count()), .little);
        
        var iter = self.vocab.iterator();
        while (iter.next()) |entry| {
            try writer.writeInt(u32, entry.value_ptr.*, .little);
            try writer.writeInt(u32, @intCast(entry.key_ptr.*.len), .little);
            try writer.writeAll(entry.key_ptr.*);
        }
        
        try writer.writeInt(u32, @intCast(self.merges.items.len), .little);
        for (self.merges.items) |merge| {
            try writer.writeInt(u32, @intCast(merge.first.len), .little);
            try writer.writeAll(merge.first);
            try writer.writeInt(u32, @intCast(merge.second.len), .little);
            try writer.writeAll(merge.second);
            try writer.writeInt(u32, merge.priority, .little);
        }
    }
    
    pub fn load(self: *EntropyTokenizer, path: []const u8) !void {
        const file = try std.fs.cwd().openFile(path, .{});
        defer file.close();
        
        const reader = file.reader();
        
        const vocab_count = try reader.readInt(u32, .little);
        
        var i: u32 = 0;
        while (i < vocab_count) : (i += 1) {
            const id = try reader.readInt(u32, .little);
            const len = try reader.readInt(u32, .little);
            const token_str = try self.allocator.alloc(u8, len);
            _ = try reader.readAll(token_str);
            try self.vocab.put(token_str, id);
        }
        
        const merge_count = try reader.readInt(u32, .little);
        
        var j: u32 = 0;
        while (j < merge_count) : (j += 1) {
            const first_len = try reader.readInt(u32, .little);
            const first = try self.allocator.alloc(u8, first_len);
            _ = try reader.readAll(first);
            
            const second_len = try reader.readInt(u32, .little);
            const second = try self.allocator.alloc(u8, second_len);
            _ = try reader.readAll(second);
            
            const priority = try reader.readInt(u32, .little);
            
            try self.merges.append(Merge{
                .first = first,
                .second = second,
                .priority = priority,
            });
        }
    }
    
    pub fn getEODToken(self: EntropyTokenizer) u32 {
        return self.special_tokens.eod;
    }
};

pub const TokenizedBatch = struct {
    tokens: []u32,
    targets: []u32,
    seq_len: usize,
    
    pub fn deinit(self: *TokenizedBatch, allocator: std.mem.Allocator) void {
        allocator.free(self.tokens);
        allocator.free(self.targets);
    }
};
