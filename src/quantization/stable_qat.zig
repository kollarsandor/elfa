const std = @import("std");
const Config = @import("../main.zig").Config;
const model = @import("../model.zig");

pub const StableQAT = struct {
    allocator: std.mem.Allocator,
    config: *const Config,
    scale_factors: std.ArrayList(f32),
    zero_points: std.ArrayList(f32),
    surrogate_gradient_fn: SurrogateGradientFn,
    
    const SurrogateGradientFn = enum {
        straight_through,
        sigmoid_tempered,
        piecewise_linear,
    };
    
    pub fn init(allocator: std.mem.Allocator, config: *const Config) !StableQAT {
        const scale_factors = std.ArrayList(f32).init(allocator);
        const zero_points = std.ArrayList(f32).init(allocator);
        
        return StableQAT{
            .allocator = allocator,
            .config = config,
            .scale_factors = scale_factors,
            .zero_points = zero_points,
            .surrogate_gradient_fn = .straight_through,
        };
    }
    
    pub fn deinit(self: *StableQAT) void {
        self.scale_factors.deinit();
        self.zero_points.deinit();
    }
    
    pub fn applyQuantization(self: *StableQAT, params: model.Parameters) !void {
        try self.quantizeTensor(params.embedding_weights, self.config.weight_bits);
        
        for (params.layers) |layer| {
            try self.quantizeTensor(layer.q_weights, self.config.weight_bits);
            try self.quantizeTensor(layer.k_weights, self.config.weight_bits);
            try self.quantizeTensor(layer.v_weights, self.config.weight_bits);
            try self.quantizeTensor(layer.o_weights, self.config.weight_bits);
            try self.quantizeTensor(layer.ffn_up_weights, self.config.weight_bits);
            try self.quantizeTensor(layer.ffn_down_weights, self.config.weight_bits);
            try self.quantizeTensor(layer.ffn_gate_weights, self.config.weight_bits);
        }
        
        try self.quantizeTensor(params.output_weights, self.config.weight_bits);
    }
    
    fn quantizeTensor(self: *StableQAT, tensor: []f32, bits: u8) !void {
        if (bits >= 16) return;
        
        const num_levels = @as(u32, 1) << @as(u5, @intCast(bits));
        const qmax = @as(f32, @floatFromInt(num_levels - 1));
        
        var min_val: f32 = tensor[0];
        var max_val: f32 = tensor[0];
        
        for (tensor) |val| {
            if (val < min_val) min_val = val;
            if (val > max_val) max_val = val;
        }
        
        const scale = (max_val - min_val) / qmax;
        const zero_point = -min_val / scale;
        
        try self.scale_factors.append(scale);
        try self.zero_points.append(zero_point);
        
        for (tensor) |*val| {
            const quantized = @round(val.* / scale + zero_point);
            const clamped = @max(0.0, @min(qmax, quantized));
            val.* = (clamped - zero_point) * scale;
        }
    }
    
    pub fn computeSurrogateGradient(self: *StableQAT, input: []const f32, grad_output: []const f32) ![]f32 {
        const grad_input = try self.allocator.alloc(f32, input.len);
        
        switch (self.surrogate_gradient_fn) {
            .straight_through => {
                @memcpy(grad_input, grad_output);
            },
            .sigmoid_tempered => {
                const temperature: f32 = 0.1;
                for (input, grad_output, 0..) |x, go, i| {
                    const sigmoid = 1.0 / (1.0 + @exp(-x / temperature));
                    grad_input[i] = go * sigmoid * (1.0 - sigmoid) / temperature;
                }
            },
            .piecewise_linear => {
                const threshold: f32 = 1.0;
                for (input, grad_output, 0..) |x, go, i| {
                    if (@abs(x) <= threshold) {
                        grad_input[i] = go;
                    } else {
                        grad_input[i] = go * 0.1;
                    }
                }
            },
        }
        
        return grad_input;
    }
    
    pub fn fakeQuantize(self: *StableQAT, tensor: []f32, bits: u8) !void {
        if (bits >= 16) return;
        
        const num_levels = @as(u32, 1) << @as(u5, @intCast(bits));
        const qmax = @as(f32, @floatFromInt(num_levels - 1));
        
        var min_val: f32 = tensor[0];
        var max_val: f32 = tensor[0];
        
        for (tensor) |val| {
            if (val < min_val) min_val = val;
            if (val > max_val) max_val = val;
        }
        
        const scale = (max_val - min_val) / qmax;
        const zero_point = -min_val / scale;
        
        for (tensor) |*val| {
            const quantized = @round(val.* / scale + zero_point);
            const clamped = @max(0.0, @min(qmax, quantized));
            val.* = (clamped - zero_point) * scale;
        }
    }
    
    pub fn quantizeToInt(self: *StableQAT, tensor: []const f32, bits: u8) ![]u8 {
        const num_levels = @as(u32, 1) << @as(u5, @intCast(bits));
        const qmax = @as(f32, @floatFromInt(num_levels - 1));
        
        var min_val: f32 = tensor[0];
        var max_val: f32 = tensor[0];
        
        for (tensor) |val| {
            if (val < min_val) min_val = val;
            if (val > max_val) max_val = val;
        }
        
        const scale = (max_val - min_val) / qmax;
        const zero_point = -min_val / scale;
        
        const quantized = try self.allocator.alloc(u8, tensor.len * bits / 8 + 1);
        
        if (bits == 4) {
            for (tensor, 0..) |val, i| {
                const qval = @as(u8, @intFromFloat(@round(val / scale + zero_point)));
                if (i % 2 == 0) {
                    quantized[i / 2] = qval & 0x0F;
                } else {
                    quantized[i / 2] |= (qval & 0x0F) << 4;
                }
            }
        } else if (bits == 8) {
            for (tensor, 0..) |val, i| {
                quantized[i] = @as(u8, @intFromFloat(@round(val / scale + zero_point)));
            }
        }
        
        return quantized;
    }
    
    pub fn dequantizeFromInt(self: *StableQAT, quantized: []const u8, bits: u8, scale: f32, zero_point: f32, output: []f32) !void {
        const num_levels = @as(u32, 1) << @as(u5, @intCast(bits));
        const qmax = @as(f32, @floatFromInt(num_levels - 1));
        
        if (bits == 4) {
            for (0..output.len) |i| {
                const byte_idx = i / 2;
                const nibble_idx = i % 2;
                const qval = if (nibble_idx == 0) quantized[byte_idx] & 0x0F else (quantized[byte_idx] >> 4) & 0x0F;
                const clamped = @as(f32, @floatFromInt(qval));
                output[i] = (clamped - zero_point) * scale;
            }
        } else if (bits == 8) {
            for (quantized, 0..) |qval, i| {
                const clamped = @as(f32, @floatFromInt(qval));
                output[i] = (clamped - zero_point) * scale;
            }
        }
        
        _ = qmax;
    }
    
    pub fn setSurrogateGradientFn(self: *StableQAT, fn_type: SurrogateGradientFn) void {
        self.surrogate_gradient_fn = fn_type;
    }
    
    pub fn calibrate(self: *StableQAT, calibration_data: []const []const f32) !void {
        for (calibration_data) |data| {
            var min_val: f32 = data[0];
            var max_val: f32 = data[0];
            
            for (data) |val| {
                if (val < min_val) min_val = val;
                if (val > max_val) max_val = val;
            }
            
            try self.scale_factors.append((max_val - min_val) / 15.0);
            try self.zero_points.append(-min_val / ((max_val - min_val) / 15.0));
        }
    }
    
    pub fn getScaleFactor(self: *StableQAT, idx: usize) f32 {
        if (idx >= self.scale_factors.items.len) return 1.0;
        return self.scale_factors.items[idx];
    }
    
    pub fn getZeroPoint(self: *StableQAT, idx: usize) f32 {
        if (idx >= self.zero_points.items.len) return 0.0;
        return self.zero_points.items[idx];
    }
};
