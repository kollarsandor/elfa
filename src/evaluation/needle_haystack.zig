const std = @import("std");
const Config = @import("../main.zig").Config;

pub const NeedleHaystack = struct {
    allocator: std.mem.Allocator,
    config: *const Config,
    needles: [][]const u8,
    haystack_templates: [][]const u8,
    
    pub const TestCase = struct {
        needle: []const u8,
        haystack: []u8,
        position: usize,
    };
    
    pub fn init(allocator: std.mem.Allocator, config: *const Config) !NeedleHaystack {
        const needles = try allocator.alloc([]const u8, 100);
        for (0..100) |i| {
            needles[i] = try std.fmt.allocPrint(allocator, "SPECIAL_NEEDLE_TOKEN_{d}", .{i});
        }
        
        const templates = try allocator.alloc([]const u8, 10);
        templates[0] = "The quick brown fox jumps over the lazy dog. ";
        templates[1] = "In the beginning, the universe was created. ";
        templates[2] = "Machine learning is a subset of artificial intelligence. ";
        templates[3] = "The capital of France is Paris. ";
        templates[4] = "Water boils at 100 degrees Celsius at sea level. ";
        templates[5] = "The Earth revolves around the Sun. ";
        templates[6] = "DNA stands for deoxyribonucleic acid. ";
        templates[7] = "Photosynthesis converts light energy into chemical energy. ";
        templates[8] = "The Great Wall of China is one of the Seven Wonders. ";
        templates[9] = "Quantum mechanics describes nature at the smallest scales. ";
        
        return NeedleHaystack{
            .allocator = allocator,
            .config = config,
            .needles = needles,
            .haystack_templates = templates,
        };
    }
    
    pub fn deinit(self: *NeedleHaystack) void {
        for (self.needles) |needle| {
            self.allocator.free(needle);
        }
        self.allocator.free(self.needles);
        self.allocator.free(self.haystack_templates);
    }
    
    pub fn generateTestCase(self: *NeedleHaystack, seed: usize) !TestCase {
        const needle_idx = seed % self.needles.len;
        const needle = self.needles[needle_idx];
        
        const target_tokens = self.config.max_seq_len;
        const needle_tokens = 10;
        const filler_tokens_needed = target_tokens - needle_tokens;
        
        var haystack = std.ArrayList(u8).init(self.allocator);
        errdefer haystack.deinit();
        
        var rng = std.rand.DefaultPrng.init(seed);
        const random = rng.random();
        
        const needle_position = random.intRangeAtMost(usize, 0, filler_tokens_needed);
        
        var current_tokens: usize = 0;
        while (current_tokens < needle_position) {
            const template_idx = random.intRangeAtMost(usize, 0, self.haystack_templates.len - 1);
            const template = self.haystack_templates[template_idx];
            try haystack.appendSlice(template);
            current_tokens += 10;
        }
        
        try haystack.appendSlice(needle);
        current_tokens += needle_tokens;
        
        while (current_tokens < target_tokens) {
            const template_idx = random.intRangeAtMost(usize, 0, self.haystack_templates.len - 1);
            const template = self.haystack_templates[template_idx];
            try haystack.appendSlice(template);
            current_tokens += 10;
        }
        
        return TestCase{
            .needle = needle,
            .haystack = try haystack.toOwnedSlice(),
            .position = needle_position,
        };
    }
    
    pub fn freeTestCase(self: *NeedleHaystack, test_case: TestCase) void {
        self.allocator.free(test_case.haystack);
    }
    
    pub fn verifyNeedlePresence(self: *NeedleHaystack, generated_text: []const u8, needle: []const u8) bool {
        _ = self;
        return std.mem.indexOf(u8, generated_text, needle) != null;
    }
    
    pub fn calculateDepthAccuracy(self: *NeedleHaystack, test_cases: []const TestCase, results: []const bool) f32 {
        _ = self;
        
        if (test_cases.len == 0 or results.len == 0) {
            return 0.0;
        }
        
        var correct: usize = 0;
        for (results) |found| {
            if (found) correct += 1;
        }
        
        return @as(f32, @floatFromInt(correct)) / @as(f32, @floatFromInt(results.len));
    }
    
    pub fn generateMultiNeedleTestCase(self: *NeedleHaystack, seed: usize, num_needles: usize) !struct { needles: [][]const u8, haystack: []u8 } {
        const target_tokens = self.config.max_seq_len;
        const needle_tokens = 10 * num_needles;
        const filler_tokens_needed = target_tokens - needle_tokens;
        
        var selected_needles = std.ArrayList([]const u8).init(self.allocator);
        errdefer selected_needles.deinit();
        
        var haystack = std.ArrayList(u8).init(self.allocator);
        errdefer haystack.deinit();
        
        var rng = std.rand.DefaultPrng.init(seed);
        const random = rng.random();
        
        var current_tokens: usize = 0;
        var needles_placed: usize = 0;
        
        const positions = try self.allocator.alloc(usize, num_needles);
        defer self.allocator.free(positions);
        
        for (0..num_needles) |i| {
            positions[i] = random.intRangeAtMost(usize, 0, filler_tokens_needed);
        }
        
        std.sort.insertion(usize, positions, {}, std.sort.asc(usize));
        
        for (0..num_needles) |i| {
            const needle_idx = (seed + i) % self.needles.len;
            try selected_needles.append(self.needles[needle_idx]);
            
            while (current_tokens < positions[i]) {
                const template_idx = random.intRangeAtMost(usize, 0, self.haystack_templates.len - 1);
                const template = self.haystack_templates[template_idx];
                try haystack.appendSlice(template);
                current_tokens += 10;
            }
            
            try haystack.appendSlice(self.needles[needle_idx]);
            current_tokens += 10;
            needles_placed += 1;
        }
        
        while (current_tokens < target_tokens) {
            const template_idx = random.intRangeAtMost(usize, 0, self.haystack_templates.len - 1);
            const template = self.haystack_templates[template_idx];
            try haystack.appendSlice(template);
            current_tokens += 10;
        }
        
        _ = needles_placed;
        
        return .{
            .needles = try selected_needles.toOwnedSlice(),
            .haystack = try haystack.toOwnedSlice(),
        };
    }
    
    pub fn freeMultiNeedleTestCase(self: *NeedleHaystack, test_case: struct { needles: [][]const u8, haystack: []u8 }) void {
        self.allocator.free(test_case.needles);
        self.allocator.free(test_case.haystack);
    }
};
