const std = @import("std");
const Config = @import("../main.zig").Config;

pub const OptimizerQuant = struct {
    allocator: std.mem.Allocator,
    config: *const Config,
    scale_factors: std.ArrayList(f32),
    zero_points: std.ArrayList(f32),
    
    pub fn init(allocator: std.mem.Allocator, config: *const Config) !OptimizerQuant {
        const scale_factors = std.ArrayList(f32).init(allocator);
        const zero_points = std.ArrayList(f32).init(allocator);
        
        return OptimizerQuant{
            .allocator = allocator,
            .config = config,
            .scale_factors = scale_factors,
            .zero_points = zero_points,
        };
    }
    
    pub fn deinit(self: *OptimizerQuant) void {
        self.scale_factors.deinit();
        self.zero_points.deinit();
    }
    
    pub fn quantizeStates(self: *OptimizerQuant, states: []const []f32) !void {
        for (states) |state| {
            try self.quantizeState(state, self.config.optimizer_state_bits);
        }
    }
    
    fn quantizeState(self: *OptimizerQuant, state: []f32, bits: u8) !void {
        if (bits >= 16) return;
        
        const num_levels = @as(u32, 1) << @as(u5, @intCast(bits));
        const qmax = @as(f32, @floatFromInt(num_levels - 1));
        
        var min_val: f32 = state[0];
        var max_val: f32 = state[0];
        
        for (state) |val| {
            if (val < min_val) min_val = val;
            if (val > max_val) max_val = val;
        }
        
        const abs_max = @max(@abs(min_val), @abs(max_val));
        const scale = abs_max / (qmax / 2.0);
        const zero_point = qmax / 2.0;
        
        try self.scale_factors.append(scale);
        try self.zero_points.append(zero_point);
        
        for (state) |*val| {
            const quantized = @round(val.* / scale + zero_point);
            const clamped = @max(0.0, @min(qmax, quantized));
            val.* = (clamped - zero_point) * scale;
        }
    }
    
    pub fn quantizeMomentum(self: *OptimizerQuant, momentum: []f32) ![]i8 {
        const quantized = try self.allocator.alloc(i8, momentum.len);
        
        var max_abs: f32 = 0.0;
        for (momentum) |val| {
            if (@abs(val) > max_abs) max_abs = @abs(val);
        }
        
        const scale = max_abs / 127.0;
        
        for (momentum, 0..) |val, i| {
            const qval = @round(val / scale);
            quantized[i] = @as(i8, @intFromFloat(@max(-127.0, @min(127.0, qval))));
        }
        
        return quantized;
    }
    
    pub fn dequantizeMomentum(self: *OptimizerQuant, quantized: []const i8, scale: f32, output: []f32) !void {
        _ = self;
        
        for (quantized, 0..) |qval, i| {
            output[i] = @as(f32, @floatFromInt(qval)) * scale;
        }
    }
    
    pub fn blockwiseQuantize(self: *OptimizerQuant, tensor: []f32, block_size: usize, bits: u8) !BlockwiseQuantized {
        const num_blocks = (tensor.len + block_size - 1) / block_size;
        
        const scales = try self.allocator.alloc(f32, num_blocks);
        const zero_points = try self.allocator.alloc(f32, num_blocks);
        
        const quantized_data = try self.allocator.alloc(u8, tensor.len * bits / 8 + num_blocks);
        
        const num_levels = @as(u32, 1) << @as(u5, @intCast(bits));
        const qmax = @as(f32, @floatFromInt(num_levels - 1));
        
        for (0..num_blocks) |block_idx| {
            const start = block_idx * block_size;
            const end = @min(start + block_size, tensor.len);
            
            var min_val: f32 = tensor[start];
            var max_val: f32 = tensor[start];
            
            for (tensor[start..end]) |val| {
                if (val < min_val) min_val = val;
                if (val > max_val) max_val = val;
            }
            
            scales[block_idx] = (max_val - min_val) / qmax;
            zero_points[block_idx] = -min_val / scales[block_idx];
            
            if (bits == 4) {
                for (tensor[start..end], 0..) |val, i| {
                    const qval = @as(u8, @intFromFloat(@round(val / scales[block_idx] + zero_points[block_idx])));
                    const clamped = @min(qval, @as(u8, @intCast(num_levels - 1)));
                    const out_idx = start + i;
                    if (out_idx % 2 == 0) {
                        quantized_data[out_idx / 2] = clamped & 0x0F;
                    } else {
                        quantized_data[out_idx / 2] |= (clamped & 0x0F) << 4;
                    }
                }
            } else if (bits == 8) {
                for (tensor[start..end], 0..) |val, i| {
                    const qval = @as(u8, @intFromFloat(@round(val / scales[block_idx] + zero_points[block_idx])));
                    const clamped = @min(qval, @as(u8, @intCast(num_levels - 1)));
                    quantized_data[start + i] = clamped;
                }
            }
        }
        
        return BlockwiseQuantized{
            .data = quantized_data,
            .scales = scales,
            .zero_points = zero_points,
            .block_size = block_size,
            .bits = bits,
            .original_len = tensor.len,
        };
    }
    
    pub fn blockwiseDequantize(self: *OptimizerQuant, bq: BlockwiseQuantized) ![]f32 {
        const output = try self.allocator.alloc(f32, bq.original_len);
        
        const num_levels = @as(u32, 1) << @as(u5, @intCast(bq.bits));
        const qmax = @as(f32, @floatFromInt(num_levels - 1));
        
        const num_blocks = (bq.original_len + bq.block_size - 1) / bq.block_size;
        
        for (0..num_blocks) |block_idx| {
            const start = block_idx * bq.block_size;
            const end = @min(start + bq.block_size, bq.original_len);
            
            const scale = bq.scales[block_idx];
            const zero_point = bq.zero_points[block_idx];
            
            if (bq.bits == 4) {
                for (start..end) |i| {
                    const byte_idx = i / 2;
                    const nibble_idx = i % 2;
                    const qval = if (nibble_idx == 0) bq.data[byte_idx] & 0x0F else (bq.data[byte_idx] >> 4) & 0x0F;
                    const clamped = @as(f32, @floatFromInt(@min(qval, @as(u8, @intCast(num_levels - 1)))));
                    output[i] = (clamped - zero_point) * scale;
                }
            } else if (bq.bits == 8) {
                for (start..end) |i| {
                    const qval = bq.data[i];
                    const clamped = @as(f32, @floatFromInt(@min(qval, @as(u8, @intCast(num_levels - 1)))));
                    output[i] = (clamped - zero_point) * scale;
                }
            }
        }
        
        _ = qmax;
        return output;
    }
    
    pub fn dynamicQuantize(self: *OptimizerQuant, tensor: []f32, bits: u8) !DynamicQuantized {
        const num_levels = @as(u32, 1) << @as(u5, @intCast(bits));
        
        var min_val: f32 = tensor[0];
        var max_val: f32 = tensor[0];
        
        for (tensor) |val| {
            if (val < min_val) min_val = val;
            if (val > max_val) max_val = val;
        }
        
        const scale = (max_val - min_val) / @as(f32, @floatFromInt(num_levels - 1));
        const zero_point = -min_val / scale;
        
        const quantized = try self.allocator.alloc(u8, tensor.len);
        
        for (tensor, 0..) |val, i| {
            const qval = @round(val / scale + zero_point);
            quantized[i] = @as(u8, @intFromFloat(@max(0.0, @min(@as(f32, @floatFromInt(num_levels - 1)), qval))));
        }
        
        return DynamicQuantized{
            .data = quantized,
            .scale = scale,
            .zero_point = zero_point,
            .bits = bits,
            .original_len = tensor.len,
        };
    }
    
    pub fn dynamicDequantize(self: *OptimizerQuant, dq: DynamicQuantized) ![]f32 {
        const output = try self.allocator.alloc(f32, dq.original_len);
        
        for (dq.data, 0..) |qval, i| {
            const val = @as(f32, @floatFromInt(qval));
            output[i] = (val - dq.zero_point) * dq.scale;
        }
        
        return output;
    }
    
    pub fn quantizeGradient(self: *OptimizerQuant, gradient: []f32) ![]i8 {
        return try self.quantizeMomentum(gradient);
    }
    
    pub fn getScaleForBlock(self: *OptimizerQuant, block_idx: usize) f32 {
        if (block_idx >= self.scale_factors.items.len) return 1.0;
        return self.scale_factors.items[block_idx];
    }
    
    pub const BlockwiseQuantized = struct {
        data: []u8,
        scales: []f32,
        zero_points: []f32,
        block_size: usize,
        bits: u8,
        original_len: usize,
        
        pub fn deinit(self: *BlockwiseQuantized, allocator: std.mem.Allocator) void {
            allocator.free(self.data);
            allocator.free(self.scales);
            allocator.free(self.zero_points);
        }
        
        pub fn getMemoryUsage(self: BlockwiseQuantized) usize {
            const data_bytes = self.data.len;
            const metadata_bytes = self.scales.len * @sizeOf(f32) + self.zero_points.len * @sizeOf(f32);
            return data_bytes + metadata_bytes;
        }
    };
    
    pub const DynamicQuantized = struct {
        data: []u8,
        scale: f32,
        zero_point: f32,
        bits: u8,
        original_len: usize,
        
        pub fn deinit(self: *DynamicQuantized, allocator: std.mem.Allocator) void {
            allocator.free(self.data);
        }
        
        pub fn getCompressionRatio(self: DynamicQuantized) f32 {
            const original_bytes = self.original_len * @sizeOf(f32);
            const compressed_bytes = self.data.len + @sizeOf(f32) * 2 + @sizeOf(u8) + @sizeOf(usize);
            return @as(f32, @floatFromInt(original_bytes)) / @as(f32, @floatFromInt(compressed_bytes));
        }
    };
};
