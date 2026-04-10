const std = @import("std");
const tensor_mod = @import("../tensor/tensor.zig");
const dtype_mod = @import("../tensor/dtype.zig");
const config_mod = @import("../util/config.zig");
const kernels = @import("../kernels/prism_kernels.zig");

pub const Tensor = tensor_mod.Tensor;
pub const Shape = tensor_mod.Shape;
comptime {
    _ = kernels;
}
pub const DType = dtype_mod.DType;

fn sigmoid(x: f32) f32 {
    if (x >= 0.0) {
        const z = @exp(-x);
        return 1.0 / (1.0 + z);
    }
    const z = @exp(x);
    return z / (1.0 + z);
}

fn geluApprox(x: f32) f32 {
    const c: f32 = 0.7978845608028654;
    const k: f32 = 0.044715;
    const x2 = x * x;
    const inner = c * (x + k * x * x2);
    return 0.5 * x * (1.0 + std.math.tanh(inner));
}

fn geluApproxDerivative(x: f32) f32 {
    const c: f32 = 0.7978845608028654;
    const k: f32 = 0.044715;
    const x2 = x * x;
    const inner = c * (x + k * x * x2);
    const t = std.math.tanh(inner);
    const sech2 = 1.0 - t * t;
    const inner_prime = c * (1.0 + 3.0 * k * x2);
    return 0.5 * (1.0 + t) + 0.5 * x * sech2 * inner_prime;
}

fn zeroSlice(slice: []f32) void {
    for (slice) |*v| v.* = 0.0;
}

fn normalizeInPlace(vec: []f32) f32 {
    var norm_sq: f32 = 0.0;
    for (vec) |v| norm_sq += v * v;
    if (norm_sq <= 1.0e-12) return 1.0;
    const norm = @sqrt(norm_sq);
    const inv_norm = 1.0 / norm;
    for (vec) |*v| v.* *= inv_norm;
    return norm;
}

fn backpropNormalize(grad_raw: []f32, normalized: []const f32, raw: []const f32, grad_normalized: []const f32, norm: f32) void {
    if (norm <= 1.0e-6) {
        for (0..grad_raw.len) |i| grad_raw[i] += grad_normalized[i];
        return;
    }
    var dot_val: f32 = 0.0;
    for (0..normalized.len) |i| dot_val += normalized[i] * grad_normalized[i];
    const inv_norm = 1.0 / norm;
    for (0..grad_raw.len) |i| grad_raw[i] += (grad_normalized[i] - normalized[i] * dot_val) * inv_norm;
    _ = raw;
}

fn projectTokenInto(dst: []f32, weight: *Tensor, token_ptr: anytype, token_offset: usize, hidden_dim: usize, head_dim: usize) !void {
    const weight_ptr = weight.typedPtr(dtype_mod.BF16) orelse return error.InvalidDType;
    for (0..head_dim) |j| {
        var sum: f32 = 0.0;
        for (0..hidden_dim) |i| sum += token_ptr[token_offset + i].toFloat32() * weight_ptr[i * head_dim + j].toFloat32();
        dst[j] = sum;
    }
}

fn accumulateMatVecTransposeToInput(grad_input: []f32, weight: *Tensor, grad_output: []const f32, hidden_dim: usize, head_dim: usize) !void {
    const weight_ptr = weight.typedPtr(dtype_mod.BF16) orelse return error.InvalidDType;
    for (0..hidden_dim) |i| {
        var sum: f32 = 0.0;
        for (0..head_dim) |j| sum += weight_ptr[i * head_dim + j].toFloat32() * grad_output[j];
        grad_input[i] += sum;
    }
}

fn storeGrad(param: *Tensor, allocator: std.mem.Allocator, grad: *Tensor) !void {
    if (param.grad) |existing| existing.deinit();
    param.grad = try grad.to(allocator, param.device, param.device_id);
}

fn cloneTensorToCpu(allocator: std.mem.Allocator, tensor: *Tensor) !*Tensor {
    return try tensor.to(allocator, .cpu, 0);
}

fn cloneOptionalTensorToCpu(allocator: std.mem.Allocator, tensor: ?*Tensor) !?*Tensor {
    if (tensor) |t| return try cloneTensorToCpu(allocator, t);
    return null;
}

fn cloneTensorArrayToCpu(allocator: std.mem.Allocator, tensors: []*Tensor) ![]*Tensor {
    const cloned = try allocator.alloc(*Tensor, tensors.len);
    errdefer allocator.free(cloned);
    var initialized: usize = 0;
    errdefer {
        for (cloned[0..initialized]) |tensor| tensor.deinit();
    }
    for (tensors, 0..) |tensor, i| {
        cloned[i] = try cloneTensorToCpu(allocator, tensor);
        initialized += 1;
    }
    return cloned;
}

fn freeTensorArray(allocator: std.mem.Allocator, tensors: []*Tensor) void {
    for (tensors) |tensor| tensor.deinit();
    allocator.free(tensors);
}

fn stateElementOffset(num_groups: usize, head_dim: usize, batch_idx: usize, group_idx: usize, row: usize, col: usize) usize {
    return (((batch_idx * num_groups) + group_idx) * head_dim + row) * head_dim + col;
}

fn copyMatrixToSlice(dst: []f32, src: []const dtype_mod.BF16, num_groups: usize, head_dim: usize, batch_idx: usize, group_idx: usize) void {
    for (0..head_dim) |row| {
        for (0..head_dim) |col| {
            dst[row * head_dim + col] = src[stateElementOffset(num_groups, head_dim, batch_idx, group_idx, row, col)].toFloat32();
        }
    }
}

fn copySliceToMatrix(dst: []dtype_mod.BF16, src: []const f32, num_groups: usize, head_dim: usize, batch_idx: usize, group_idx: usize) void {
    for (0..head_dim) |row| {
        for (0..head_dim) |col| {
            dst[stateElementOffset(num_groups, head_dim, batch_idx, group_idx, row, col)] = dtype_mod.BF16.fromFloat32(src[row * head_dim + col]);
        }
    }
}

pub const PrismLayer = struct {
    config: config_mod.PrismConfig,
    hidden_dim: usize,
    num_iterations: usize,
    head_dim: usize,
    num_groups: usize,
    shortconv: *ShortConv,
    w_beta: []*Tensor,
    w_k: []*Tensor,
    w_p: []*Tensor,
    alpha: f32,
    allocator: std.mem.Allocator,
    device: tensor_mod.Device,
    device_id: i32,

    pub fn init(allocator: std.mem.Allocator, config: config_mod.PrismConfig, hidden_dim: usize, head_dim: usize, device: tensor_mod.Device, device_id: i32, rng: *std.Random) !*PrismLayer {
        if (hidden_dim == 0) return error.InvalidHiddenDimension;
        if (head_dim == 0 or head_dim > hidden_dim) return error.InvalidHeadDimension;
        if (hidden_dim % head_dim != 0) return error.InvalidHeadDimension;
        if (config.num_iterations == 0) return error.InvalidNumIterations;
        if (config.shortconv_window == 0) return error.InvalidShortConvWindow;
        const self = try allocator.create(PrismLayer);
        errdefer allocator.destroy(self);
        const num_iterations = config.num_iterations;
        const scale = @sqrt(2.0 / @as(f64, @floatFromInt(hidden_dim)));
        const shortconv = try ShortConv.init(allocator, hidden_dim, config.shortconv_window, device, device_id, rng);
        errdefer shortconv.deinit();
        var w_beta = try allocator.alloc(*Tensor, num_iterations);
        errdefer allocator.free(w_beta);
        var w_k = try allocator.alloc(*Tensor, num_iterations);
        errdefer allocator.free(w_k);
        var w_p = try allocator.alloc(*Tensor, num_iterations);
        errdefer allocator.free(w_p);
        var beta_initialized: usize = 0;
        var k_initialized: usize = 0;
        var p_initialized: usize = 0;
        errdefer {
            var i: usize = 0;
            while (i < beta_initialized) : (i += 1) w_beta[i].deinit();
        }
        errdefer {
            var i: usize = 0;
            while (i < k_initialized) : (i += 1) w_k[i].deinit();
        }
        errdefer {
            var i: usize = 0;
            while (i < p_initialized) : (i += 1) w_p[i].deinit();
        }
        for (0..num_iterations) |l| {
            const beta_shape = Shape.init(&[_]usize{ hidden_dim, head_dim });
            w_beta[l] = try Tensor.randNormal(allocator, beta_shape, .bf16, device, device_id, rng, 0.0, @floatCast(scale));
            beta_initialized += 1;
            const k_shape = Shape.init(&[_]usize{ hidden_dim, head_dim });
            w_k[l] = try Tensor.randNormal(allocator, k_shape, .bf16, device, device_id, rng, 0.0, @floatCast(scale));
            k_initialized += 1;
            const p_shape = Shape.init(&[_]usize{ hidden_dim, head_dim });
            w_p[l] = try Tensor.randNormal(allocator, p_shape, .bf16, device, device_id, rng, 0.0, @floatCast(scale));
            p_initialized += 1;
        }
        self.* = .{
            .config = config,
            .hidden_dim = hidden_dim,
            .num_iterations = num_iterations,
            .head_dim = head_dim,
            .num_groups = hidden_dim / head_dim,
            .shortconv = shortconv,
            .w_beta = w_beta,
            .w_k = w_k,
            .w_p = w_p,
            .alpha = config.forget_factor,
            .allocator = allocator,
            .device = device,
            .device_id = device_id,
        };
        return self;
    }

    pub fn deinit(self: *PrismLayer) void {
        self.shortconv.deinit();
        for (self.w_beta) |w| w.deinit();
        for (self.w_k) |w| w.deinit();
        for (self.w_p) |w| w.deinit();
        self.allocator.free(self.w_beta);
        self.allocator.free(self.w_k);
        self.allocator.free(self.w_p);
        self.allocator.destroy(self);
    }

    fn validateForwardInputs(self: *PrismLayer, input: *Tensor, v: *Tensor, state: ?*PrismState) !void {
        if (input.device != self.device or input.device_id != self.device_id) return error.DeviceMismatch;
        if (v.device != self.device or v.device_id != self.device_id) return error.DeviceMismatch;
        if (input.dtype != .bf16 or v.dtype != .bf16) return error.DTypeMismatch;
        if (input.shape.ndim != 3) return error.InvalidInputShape;
        if (v.shape.ndim != 3) return error.InvalidValueShape;
        if (input.shape.dim(2) != self.hidden_dim) return error.InvalidInputShape;
        if (v.shape.dim(0) != input.shape.dim(0) or v.shape.dim(1) != input.shape.dim(1) or v.shape.dim(2) != self.hidden_dim) return error.InvalidValueShape;
        if (state) |s| {
            if (s.device != self.device or s.device_id != self.device_id) return error.DeviceMismatch;
            if (s.state.dtype != .bf16) return error.DTypeMismatch;
            if (s.state.shape.ndim != 4) return error.InvalidStateShape;
            if (s.batch_size != input.shape.dim(0)) return error.InvalidStateShape;
            if (s.num_groups != self.num_groups) return error.InvalidStateShape;
            if (s.state_dim != self.head_dim or s.value_dim != self.head_dim) return error.InvalidStateShape;
            if (s.state.shape.dim(0) != input.shape.dim(0) or s.state.shape.dim(1) != self.num_groups or s.state.shape.dim(2) != self.head_dim or s.state.shape.dim(3) != self.head_dim) return error.InvalidStateShape;
        }
    }

    fn validateBackwardInputs(self: *PrismLayer, grad_output: *Tensor, input: *Tensor, v: *Tensor, state: *PrismState) !void {
        try self.validateForwardInputs(input, v, state);
        if (grad_output.device != self.device or grad_output.device_id != self.device_id) return error.DeviceMismatch;
        if (grad_output.dtype != .bf16) return error.DTypeMismatch;
        if (grad_output.shape.ndim != 3) return error.InvalidGradientShape;
        if (!grad_output.shape.equalTo(v.shape)) return error.InvalidGradientShape;
    }

    pub fn forward(self: *PrismLayer, input: *Tensor, v: *Tensor, state: ?*PrismState) !struct { output: *Tensor, new_state: *PrismState } {
        try self.validateForwardInputs(input, v, state);
        if (self.device != .cpu) return self.forwardViaCpu(input, v, state);
        return self.forwardCpuImpl(input, v, state);
    }

    fn forwardViaCpu(self: *PrismLayer, input: *Tensor, v: *Tensor, state: ?*PrismState) !struct { output: *Tensor, new_state: *PrismState } {
        var input_cpu = try cloneTensorToCpu(self.allocator, input);
        defer input_cpu.deinit();
        var v_cpu = try cloneTensorToCpu(self.allocator, v);
        defer v_cpu.deinit();
        var state_cpu: ?*PrismState = null;
        defer if (state_cpu) |s| s.deinit();
        if (state) |s| state_cpu = try s.to(self.allocator, .cpu, 0);
        const shortconv_weight_cpu = try cloneTensorToCpu(self.allocator, self.shortconv.weight);
        defer shortconv_weight_cpu.deinit();
        const shortconv_bias_cpu = try cloneOptionalTensorToCpu(self.allocator, self.shortconv.bias);
        defer if (shortconv_bias_cpu) |b| b.deinit();
        const w_beta_cpu = try cloneTensorArrayToCpu(self.allocator, self.w_beta);
        defer freeTensorArray(self.allocator, w_beta_cpu);
        const w_k_cpu = try cloneTensorArrayToCpu(self.allocator, self.w_k);
        defer freeTensorArray(self.allocator, w_k_cpu);
        const w_p_cpu = try cloneTensorArrayToCpu(self.allocator, self.w_p);
        defer freeTensorArray(self.allocator, w_p_cpu);
        var cpu_shortconv = ShortConv{
            .window_size = self.shortconv.window_size,
            .hidden_dim = self.shortconv.hidden_dim,
            .weight = shortconv_weight_cpu,
            .bias = shortconv_bias_cpu,
            .allocator = self.allocator,
            .device = .cpu,
            .device_id = 0,
        };
        var cpu_layer = self.*;
        cpu_layer.shortconv = &cpu_shortconv;
        cpu_layer.w_beta = w_beta_cpu;
        cpu_layer.w_k = w_k_cpu;
        cpu_layer.w_p = w_p_cpu;
        cpu_layer.device = .cpu;
        cpu_layer.device_id = 0;
        var cpu_result = try cpu_layer.forwardCpuImpl(input_cpu, v_cpu, state_cpu);
        defer cpu_result.output.deinit();
        defer cpu_result.new_state.deinit();
        const output = try cpu_result.output.to(self.allocator, self.device, self.device_id);
        errdefer output.deinit();
        const new_state = try cpu_result.new_state.to(self.allocator, self.device, self.device_id);
        errdefer new_state.deinit();
        return .{ .output = output, .new_state = new_state };
    }

    fn forwardCpuImpl(self: *PrismLayer, input: *Tensor, v: *Tensor, state: ?*PrismState) !struct { output: *Tensor, new_state: *PrismState } {
        const batch_size = input.shape.dim(0);
        const seq_len = input.shape.dim(1);
        const u = try self.shortconv.forward(input);
        defer u.deinit();
        var new_state = if (state) |s| try s.clone() else try PrismState.initWithGroups(self.allocator, batch_size, self.num_groups, self.head_dim, self.head_dim, self.device, self.device_id);
        errdefer new_state.deinit();
        const output_shape = Shape.init(&[_]usize{ batch_size, seq_len, self.hidden_dim });
        var output = try Tensor.zeros(self.allocator, output_shape, .bf16, self.device, self.device_id);
        errdefer output.deinit();
        try self.prismForwardCpu(u, v, new_state, output);
        return .{ .output = output, .new_state = new_state };
    }

    fn prismForwardCpu(self: *PrismLayer, u: *Tensor, v: *Tensor, new_state: *PrismState, output: *Tensor) !void {
        const batch_size = u.shape.dim(0);
        const seq_len = u.shape.dim(1);
        const u_ptr = u.typedPtr(dtype_mod.BF16) orelse return error.InvalidDType;
        const v_ptr = v.typedPtr(dtype_mod.BF16) orelse return error.InvalidDType;
        const out_ptr = output.typedPtr(dtype_mod.BF16) orelse return error.InvalidDType;
        const new_state_ptr = new_state.state.typedPtr(dtype_mod.BF16) orelse return error.InvalidDType;
        const total_output_elems = batch_size * seq_len * self.hidden_dim;
        for (0..total_output_elems) |i| out_ptr[i] = u_ptr[i];
        var current_state = try self.allocator.alloc(f32, self.num_groups * self.head_dim * self.head_dim);
        defer self.allocator.free(current_state);
        var next_state = try self.allocator.alloc(f32, self.num_groups * self.head_dim * self.head_dim);
        defer self.allocator.free(next_state);
        var beta_proj = try self.allocator.alloc(f32, self.head_dim);
        defer self.allocator.free(beta_proj);
        var key_raw = try self.allocator.alloc(f32, self.head_dim);
        defer self.allocator.free(key_raw);
        var key_vec = try self.allocator.alloc(f32, self.head_dim);
        defer self.allocator.free(key_vec);
        var p_raw = try self.allocator.alloc(f32, self.head_dim);
        defer self.allocator.free(p_raw);
        var p_vec = try self.allocator.alloc(f32, self.head_dim);
        defer self.allocator.free(p_vec);
        var delta = try self.allocator.alloc(f32, self.head_dim);
        defer self.allocator.free(delta);
        var residual = try self.allocator.alloc(f32, self.num_groups * self.head_dim);
        defer self.allocator.free(residual);
        var refined = try self.allocator.alloc(f32, self.num_groups * self.head_dim);
        defer self.allocator.free(refined);
        var state_times_key = try self.allocator.alloc(f32, self.head_dim);
        defer self.allocator.free(state_times_key);
        const state_matrix_size = self.head_dim * self.head_dim;
        for (0..batch_size) |b| {
            const base_state_ptr = new_state_ptr;
            for (0..self.num_groups) |g| {
                for (0..self.head_dim) |row| {
                    for (0..self.head_dim) |col| {
                        current_state[g * state_matrix_size + row * self.head_dim + col] = base_state_ptr[stateElementOffset(self.num_groups, self.head_dim, b, g, row, col)].toFloat32();
                    }
                }
            }
            for (0..seq_len) |t| {
                const token_offset = (b * seq_len + t) * self.hidden_dim;
                for (0..self.hidden_dim) |d| out_ptr[token_offset + d] = u_ptr[token_offset + d];
                try projectTokenInto(beta_proj, self.w_beta[0], u_ptr, token_offset, self.hidden_dim, self.head_dim);
                _ = beta_proj;
                for (0..self.num_groups) |g| {
                    const group_offset = g * self.head_dim;
                    for (0..self.head_dim) |d| {
                        residual[group_offset + d] = v_ptr[token_offset + group_offset + d].toFloat32() - u_ptr[token_offset + group_offset + d].toFloat32();
                        refined[group_offset + d] = 0.0;
                    }
                }
                for (0..self.num_iterations) |l| {
                    try projectTokenInto(beta_proj, self.w_beta[l], u_ptr, token_offset, self.hidden_dim, self.head_dim);
                    var beta_sum: f32 = 0.0;
                    for (beta_proj) |v_beta| beta_sum += v_beta;
                    const beta_scalar = sigmoid(beta_sum / @as(f32, @floatFromInt(self.head_dim)));
                    try projectTokenInto(key_raw, self.w_k[l], u_ptr, token_offset, self.hidden_dim, self.head_dim);
                    for (0..self.head_dim) |d| key_vec[d] = key_raw[d];
                    _ = normalizeInPlace(key_vec);
                    try projectTokenInto(p_raw, self.w_p[l], u_ptr, token_offset, self.hidden_dim, self.head_dim);
                    for (0..self.head_dim) |d| p_vec[d] = sigmoid(p_raw[d]);
                    for (0..self.num_groups) |g| {
                        const group_offset = g * self.head_dim;
                        const state_offset = g * state_matrix_size;
                        for (0..self.head_dim) |d| delta[d] = geluApprox(p_vec[d] * residual[group_offset + d]);
                        if (l == 0) {
                            for (0..self.head_dim) |row| {
                                var sum: f32 = 0.0;
                                for (0..self.head_dim) |col| sum += current_state[state_offset + row * self.head_dim + col] * key_vec[col];
                                state_times_key[row] = sum;
                            }
                            for (0..self.head_dim) |row| {
                                for (0..self.head_dim) |col| {
                                    next_state[state_offset + row * self.head_dim + col] = self.alpha * current_state[state_offset + row * self.head_dim + col] - self.alpha * beta_scalar * state_times_key[row] * key_vec[col] + beta_scalar * delta[row] * key_vec[col];
                                }
                            }
                        } else {
                            for (0..self.head_dim) |row| {
                                for (0..self.head_dim) |col| {
                                    next_state[state_offset + row * self.head_dim + col] = current_state[state_offset + row * self.head_dim + col] + beta_scalar * delta[row] * key_vec[col];
                                }
                            }
                        }
                        for (0..self.head_dim) |d| {
                            residual[group_offset + d] -= delta[d];
                            refined[group_offset + d] += delta[d];
                        }
                    }
                    @memcpy(current_state, next_state);
                }
                for (0..self.hidden_dim) |d| out_ptr[token_offset + d] = dtype_mod.BF16.fromFloat32(u_ptr[token_offset + d].toFloat32() + refined[d]);
            }
            for (0..self.num_groups) |g| {
                const state_offset = g * state_matrix_size;
                for (0..self.head_dim) |row| {
                    for (0..self.head_dim) |col| {
                        new_state_ptr[stateElementOffset(self.num_groups, self.head_dim, b, g, row, col)] = dtype_mod.BF16.fromFloat32(current_state[state_offset + row * self.head_dim + col]);
                    }
                }
            }
        }
    }

    pub fn backward(self: *PrismLayer, grad_output: *Tensor, input: *Tensor, v: *Tensor, state: *PrismState) !struct { grad_input: *Tensor, grad_v: *Tensor, grad_state: *PrismState } {
        try self.validateBackwardInputs(grad_output, input, v, state);
        if (self.device != .cpu) return self.backwardViaCpu(grad_output, input, v, state);
        return self.backwardCpu(grad_output, input, v, state);
    }

    fn backwardViaCpu(self: *PrismLayer, grad_output: *Tensor, input: *Tensor, v: *Tensor, state: *PrismState) !struct { grad_input: *Tensor, grad_v: *Tensor, grad_state: *PrismState } {
        var grad_output_cpu = try cloneTensorToCpu(self.allocator, grad_output);
        defer grad_output_cpu.deinit();
        var input_cpu = try cloneTensorToCpu(self.allocator, input);
        defer input_cpu.deinit();
        var v_cpu = try cloneTensorToCpu(self.allocator, v);
        defer v_cpu.deinit();
        var state_cpu = try state.to(self.allocator, .cpu, 0);
        defer state_cpu.deinit();
        const shortconv_weight_cpu = try cloneTensorToCpu(self.allocator, self.shortconv.weight);
        defer shortconv_weight_cpu.deinit();
        const shortconv_bias_cpu = try cloneOptionalTensorToCpu(self.allocator, self.shortconv.bias);
        defer if (shortconv_bias_cpu) |b| b.deinit();
        const w_beta_cpu = try cloneTensorArrayToCpu(self.allocator, self.w_beta);
        defer freeTensorArray(self.allocator, w_beta_cpu);
        const w_k_cpu = try cloneTensorArrayToCpu(self.allocator, self.w_k);
        defer freeTensorArray(self.allocator, w_k_cpu);
        const w_p_cpu = try cloneTensorArrayToCpu(self.allocator, self.w_p);
        defer freeTensorArray(self.allocator, w_p_cpu);
        var cpu_shortconv = ShortConv{
            .window_size = self.shortconv.window_size,
            .hidden_dim = self.shortconv.hidden_dim,
            .weight = shortconv_weight_cpu,
            .bias = shortconv_bias_cpu,
            .allocator = self.allocator,
            .device = .cpu,
            .device_id = 0,
        };
        var cpu_layer = self.*;
        cpu_layer.shortconv = &cpu_shortconv;
        cpu_layer.w_beta = w_beta_cpu;
        cpu_layer.w_k = w_k_cpu;
        cpu_layer.w_p = w_p_cpu;
        cpu_layer.device = .cpu;
        cpu_layer.device_id = 0;
        var cpu_result = try cpu_layer.backwardCpu(grad_output_cpu, input_cpu, v_cpu, state_cpu);
        defer cpu_result.grad_input.deinit();
        defer cpu_result.grad_v.deinit();
        defer cpu_result.grad_state.deinit();
        try storeGrad(self.shortconv.weight, self.allocator, cpu_shortconv.weight.grad orelse return error.GradientNotAvailable);
        if (self.shortconv.bias) |bias| {
            const cpu_bias = cpu_shortconv.bias orelse return error.GradientNotAvailable;
            const bias_grad = cpu_bias.grad orelse return error.GradientNotAvailable;
            try storeGrad(bias, self.allocator, bias_grad);
        }
        for (0..self.num_iterations) |l| {
            try storeGrad(self.w_beta[l], self.allocator, cpu_layer.w_beta[l].grad orelse return error.GradientNotAvailable);
            try storeGrad(self.w_k[l], self.allocator, cpu_layer.w_k[l].grad orelse return error.GradientNotAvailable);
            try storeGrad(self.w_p[l], self.allocator, cpu_layer.w_p[l].grad orelse return error.GradientNotAvailable);
        }
        const grad_input = try cpu_result.grad_input.to(self.allocator, self.device, self.device_id);
        errdefer grad_input.deinit();
        const grad_v_dev = try cpu_result.grad_v.to(self.allocator, self.device, self.device_id);
        errdefer grad_v_dev.deinit();
        const grad_state_dev = try cpu_result.grad_state.to(self.allocator, self.device, self.device_id);
        errdefer grad_state_dev.deinit();
        return .{ .grad_input = grad_input, .grad_v = grad_v_dev, .grad_state = grad_state_dev };
    }

    fn backwardCpu(self: *PrismLayer, grad_output: *Tensor, input: *Tensor, v: *Tensor, state: *PrismState) !struct { grad_input: *Tensor, grad_v: *Tensor, grad_state: *PrismState } {
        const batch_size = input.shape.dim(0);
        const seq_len = input.shape.dim(1);
        const state_matrix_size = self.head_dim * self.head_dim;
        const token_state_span = self.num_groups * state_matrix_size;
        const u = try self.shortconv.forward(input);
        defer u.deinit();
        const grad_output_ptr = grad_output.typedPtr(dtype_mod.BF16) orelse return error.InvalidDType;
        const u_ptr = u.typedPtr(dtype_mod.BF16) orelse return error.InvalidDType;
        const v_ptr = v.typedPtr(dtype_mod.BF16) orelse return error.InvalidDType;
        const state_ptr = state.state.typedPtr(dtype_mod.BF16) orelse return error.InvalidDType;
        var token_start_states = try self.allocator.alloc(f32, batch_size * seq_len * token_state_span);
        defer self.allocator.free(token_start_states);
        var current_state = try self.allocator.alloc(f32, token_state_span);
        defer self.allocator.free(current_state);
        var next_state = try self.allocator.alloc(f32, token_state_span);
        defer self.allocator.free(next_state);
        var beta_proj = try self.allocator.alloc(f32, self.head_dim);
        defer self.allocator.free(beta_proj);
        var key_raw = try self.allocator.alloc(f32, self.head_dim);
        defer self.allocator.free(key_raw);
        var key_vec = try self.allocator.alloc(f32, self.head_dim);
        defer self.allocator.free(key_vec);
        var p_raw = try self.allocator.alloc(f32, self.head_dim);
        defer self.allocator.free(p_raw);
        var p_vec = try self.allocator.alloc(f32, self.head_dim);
        defer self.allocator.free(p_vec);
        var delta = try self.allocator.alloc(f32, self.head_dim);
        defer self.allocator.free(delta);
        var residual = try self.allocator.alloc(f32, self.num_groups * self.head_dim);
        defer self.allocator.free(residual);
        var state_times_key = try self.allocator.alloc(f32, self.head_dim);
        defer self.allocator.free(state_times_key);
        for (0..batch_size) |b| {
            for (0..self.num_groups) |g| {
                for (0..self.head_dim) |row| {
                    for (0..self.head_dim) |col| {
                        current_state[g * state_matrix_size + row * self.head_dim + col] = state_ptr[stateElementOffset(self.num_groups, self.head_dim, b, g, row, col)].toFloat32();
                    }
                }
            }
            for (0..seq_len) |t| {
                const token_state_offset = ((b * seq_len) + t) * token_state_span;
                @memcpy(token_start_states[token_state_offset .. token_state_offset + token_state_span], current_state);
                const token_offset = (b * seq_len + t) * self.hidden_dim;
                for (0..self.num_groups) |g| {
                    const group_offset = g * self.head_dim;
                    for (0..self.head_dim) |d| residual[group_offset + d] = v_ptr[token_offset + group_offset + d].toFloat32() - u_ptr[token_offset + group_offset + d].toFloat32();
                }
                for (0..self.num_iterations) |l| {
                    try projectTokenInto(beta_proj, self.w_beta[l], u_ptr, token_offset, self.hidden_dim, self.head_dim);
                    var beta_sum: f32 = 0.0;
                    for (beta_proj) |v_beta| beta_sum += v_beta;
                    const beta_scalar = sigmoid(beta_sum / @as(f32, @floatFromInt(self.head_dim)));
                    try projectTokenInto(key_raw, self.w_k[l], u_ptr, token_offset, self.hidden_dim, self.head_dim);
                    for (0..self.head_dim) |d| key_vec[d] = key_raw[d];
                    _ = normalizeInPlace(key_vec);
                    try projectTokenInto(p_raw, self.w_p[l], u_ptr, token_offset, self.hidden_dim, self.head_dim);
                    for (0..self.head_dim) |d| p_vec[d] = sigmoid(p_raw[d]);
                    for (0..self.num_groups) |g| {
                        const group_offset = g * self.head_dim;
                        const group_state_offset = g * state_matrix_size;
                        for (0..self.head_dim) |d| delta[d] = geluApprox(p_vec[d] * residual[group_offset + d]);
                        if (l == 0) {
                            for (0..self.head_dim) |row| {
                                var sum: f32 = 0.0;
                                for (0..self.head_dim) |col| sum += current_state[group_state_offset + row * self.head_dim + col] * key_vec[col];
                                state_times_key[row] = sum;
                            }
                            for (0..self.head_dim) |row| {
                                for (0..self.head_dim) |col| {
                                    next_state[group_state_offset + row * self.head_dim + col] = self.alpha * current_state[group_state_offset + row * self.head_dim + col] - self.alpha * beta_scalar * state_times_key[row] * key_vec[col] + beta_scalar * delta[row] * key_vec[col];
                                }
                            }
                        } else {
                            for (0..self.head_dim) |row| {
                                for (0..self.head_dim) |col| {
                                    next_state[group_state_offset + row * self.head_dim + col] = current_state[group_state_offset + row * self.head_dim + col] + beta_scalar * delta[row] * key_vec[col];
                                }
                            }
                        }
                        for (0..self.head_dim) |d| residual[group_offset + d] -= delta[d];
                    }
                    @memcpy(current_state, next_state);
                }
            }
        }
        var grad_input = try Tensor.zeros(self.allocator, input.shape, .bf16, self.device, self.device_id);
        errdefer grad_input.deinit();
        var grad_v = try Tensor.zeros(self.allocator, v.shape, .bf16, self.device, self.device_id);
        errdefer grad_v.deinit();
        var grad_state = try PrismState.initWithGroups(self.allocator, state.batch_size, self.num_groups, self.head_dim, self.head_dim, self.device, self.device_id);
        errdefer grad_state.deinit();
        const grad_input_ptr = grad_input.typedPtr(dtype_mod.BF16) orelse return error.InvalidDType;
        const grad_v_ptr = grad_v.typedPtr(dtype_mod.BF16) orelse return error.InvalidDType;
        const grad_state_ptr = grad_state.state.typedPtr(dtype_mod.BF16) orelse return error.InvalidDType;
        for (0..grad_v.shape.numel()) |i| grad_v_ptr[i] = dtype_mod.BF16.fromFloat32(0.0);
        for (0..grad_state.state.shape.numel()) |i| grad_state_ptr[i] = dtype_mod.BF16.fromFloat32(0.0);
        var grad_u_full = try self.allocator.alloc(f32, batch_size * seq_len * self.hidden_dim);
        defer self.allocator.free(grad_u_full);
        zeroSlice(grad_u_full);
        var grad_shortconv_weight = try Tensor.zeros(self.allocator, self.shortconv.weight.shape, .bf16, self.device, self.device_id);
        defer grad_shortconv_weight.deinit();
        var grad_shortconv_bias: ?*Tensor = null;
        if (self.shortconv.bias) |bias| {
            grad_shortconv_bias = try Tensor.zeros(self.allocator, bias.shape, .bf16, self.device, self.device_id);
        }
        defer if (grad_shortconv_bias) |b| b.deinit();
        var grad_w_beta = try self.allocator.alloc(*Tensor, self.num_iterations);
        defer self.allocator.free(grad_w_beta);
        var grad_w_k = try self.allocator.alloc(*Tensor, self.num_iterations);
        defer self.allocator.free(grad_w_k);
        var grad_w_p = try self.allocator.alloc(*Tensor, self.num_iterations);
        defer self.allocator.free(grad_w_p);
        for (0..self.num_iterations) |l| {
            grad_w_beta[l] = try Tensor.zeros(self.allocator, self.w_beta[l].shape, .bf16, self.device, self.device_id);
            grad_w_k[l] = try Tensor.zeros(self.allocator, self.w_k[l].shape, .bf16, self.device, self.device_id);
            grad_w_p[l] = try Tensor.zeros(self.allocator, self.w_p[l].shape, .bf16, self.device, self.device_id);
        }
        defer {
            for (0..self.num_iterations) |l| {
                grad_w_beta[l].deinit();
                grad_w_k[l].deinit();
                grad_w_p[l].deinit();
            }
        }
        const grad_sc_ptr = grad_shortconv_weight.typedPtr(dtype_mod.BF16) orelse return error.InvalidDType;
        const grad_bias_ptr = if (grad_shortconv_bias) |b| b.typedPtr(dtype_mod.BF16) orelse return error.InvalidDType else null;
        const input_ptr = input.typedPtr(dtype_mod.BF16) orelse return error.InvalidDType;
        const weight_ptr = self.shortconv.weight.typedPtr(dtype_mod.BF16) orelse return error.InvalidDType;
        var iter_state_cache = try self.allocator.alloc(f32, (self.num_iterations + 1) * token_state_span);
        defer self.allocator.free(iter_state_cache);
        var residual_before_cache = try self.allocator.alloc(f32, self.num_iterations * self.num_groups * self.head_dim);
        defer self.allocator.free(residual_before_cache);
        var delta_cache = try self.allocator.alloc(f32, self.num_iterations * self.num_groups * self.head_dim);
        defer self.allocator.free(delta_cache);
        var beta_scalar_cache = try self.allocator.alloc(f32, self.num_iterations);
        defer self.allocator.free(beta_scalar_cache);
        var key_raw_cache = try self.allocator.alloc(f32, self.num_iterations * self.head_dim);
        defer self.allocator.free(key_raw_cache);
        var key_cache = try self.allocator.alloc(f32, self.num_iterations * self.head_dim);
        defer self.allocator.free(key_cache);
        var key_norm_cache = try self.allocator.alloc(f32, self.num_iterations);
        defer self.allocator.free(key_norm_cache);
        var p_cache = try self.allocator.alloc(f32, self.num_iterations * self.head_dim);
        defer self.allocator.free(p_cache);
        var grad_state_after = try self.allocator.alloc(f32, token_state_span);
        defer self.allocator.free(grad_state_after);
        var grad_state_before = try self.allocator.alloc(f32, token_state_span);
        defer self.allocator.free(grad_state_before);
        var grad_residual_after = try self.allocator.alloc(f32, self.num_groups * self.head_dim);
        defer self.allocator.free(grad_residual_after);
        var grad_residual_before = try self.allocator.alloc(f32, self.num_groups * self.head_dim);
        defer self.allocator.free(grad_residual_before);
        var grad_u_token = try self.allocator.alloc(f32, self.hidden_dim);
        defer self.allocator.free(grad_u_token);
        var grad_v_token = try self.allocator.alloc(f32, self.hidden_dim);
        defer self.allocator.free(grad_v_token);
        var grad_beta_proj = try self.allocator.alloc(f32, self.head_dim);
        defer self.allocator.free(grad_beta_proj);
        var grad_key = try self.allocator.alloc(f32, self.head_dim);
        defer self.allocator.free(grad_key);
        var grad_key_raw = try self.allocator.alloc(f32, self.head_dim);
        defer self.allocator.free(grad_key_raw);
        var grad_p_raw = try self.allocator.alloc(f32, self.head_dim);
        defer self.allocator.free(grad_p_raw);
        var grad_delta = try self.allocator.alloc(f32, self.head_dim);
        defer self.allocator.free(grad_delta);
        var state_times_key_local = try self.allocator.alloc(f32, self.head_dim);
        defer self.allocator.free(state_times_key_local);
        var temp_gk = try self.allocator.alloc(f32, self.head_dim);
        defer self.allocator.free(temp_gk);
        var temp_gt = try self.allocator.alloc(f32, self.head_dim);
        defer self.allocator.free(temp_gt);
        for (0..batch_size) |b| {
            zeroSlice(grad_state_after);
            var t_rev: usize = seq_len;
            while (t_rev > 0) {
                t_rev -= 1;
                const token_offset = (b * seq_len + t_rev) * self.hidden_dim;
                const token_state_offset = ((b * seq_len) + t_rev) * token_state_span;
                @memcpy(iter_state_cache[0..token_state_span], token_start_states[token_state_offset .. token_state_offset + token_state_span]);
                for (0..self.num_iterations) |l| {
                    try projectTokenInto(beta_proj, self.w_beta[l], u_ptr, token_offset, self.hidden_dim, self.head_dim);
                    var beta_sum: f32 = 0.0;
                    for (0..self.head_dim) |d| beta_sum += beta_proj[d];
                    beta_scalar_cache[l] = sigmoid(beta_sum / @as(f32, @floatFromInt(self.head_dim)));
                    try projectTokenInto(key_raw_cache[l * self.head_dim ..][0..self.head_dim], self.w_k[l], u_ptr, token_offset, self.hidden_dim, self.head_dim);
                    for (0..self.head_dim) |d| key_cache[l * self.head_dim + d] = key_raw_cache[l * self.head_dim + d];
                    key_norm_cache[l] = normalizeInPlace(key_cache[l * self.head_dim ..][0..self.head_dim]);
                    try projectTokenInto(p_raw, self.w_p[l], u_ptr, token_offset, self.hidden_dim, self.head_dim);
                    for (0..self.head_dim) |d| p_cache[l * self.head_dim + d] = sigmoid(p_raw[d]);
                    for (0..self.num_groups) |g| {
                        const group_offset = g * self.head_dim;
                        const group_state_offset = g * state_matrix_size;
                        const state_before = iter_state_cache[l * token_state_span + group_state_offset ..][0..state_matrix_size];
                        const state_after = iter_state_cache[(l + 1) * token_state_span + group_state_offset ..][0..state_matrix_size];
                        for (0..self.head_dim) |d| {
                            const cache_idx = (l * self.num_groups + g) * self.head_dim + d;
                            if (l == 0) {
                                residual_before_cache[cache_idx] = v_ptr[token_offset + group_offset + d].toFloat32() - u_ptr[token_offset + group_offset + d].toFloat32();
                            } else {
                                residual_before_cache[cache_idx] = residual_before_cache[((l - 1) * self.num_groups + g) * self.head_dim + d] - delta_cache[((l - 1) * self.num_groups + g) * self.head_dim + d];
                            }
                            delta_cache[cache_idx] = geluApprox(p_cache[l * self.head_dim + d] * residual_before_cache[cache_idx]);
                        }
                        if (l == 0) {
                            for (0..self.head_dim) |row| {
                                var sum: f32 = 0.0;
                                for (0..self.head_dim) |col| sum += state_before[row * self.head_dim + col] * key_cache[l * self.head_dim + col];
                                state_times_key_local[row] = sum;
                            }
                            for (0..self.head_dim) |row| {
                                for (0..self.head_dim) |col| {
                                    state_after[row * self.head_dim + col] = self.alpha * state_before[row * self.head_dim + col] - self.alpha * beta_scalar_cache[l] * state_times_key_local[row] * key_cache[l * self.head_dim + col] + beta_scalar_cache[l] * delta_cache[(l * self.num_groups + g) * self.head_dim + row] * key_cache[l * self.head_dim + col];
                                }
                            }
                        } else {
                            for (0..self.head_dim) |row| {
                                for (0..self.head_dim) |col| {
                                    state_after[row * self.head_dim + col] = state_before[row * self.head_dim + col] + beta_scalar_cache[l] * delta_cache[(l * self.num_groups + g) * self.head_dim + row] * key_cache[l * self.head_dim + col];
                                }
                            }
                        }
                    }
                }
                for (0..self.hidden_dim) |d| {
                    grad_u_token[d] = grad_output_ptr[token_offset + d].toFloat32();
                    grad_v_token[d] = 0.0;
                }
                zeroSlice(grad_residual_after);
                var l_rev: usize = self.num_iterations;
                while (l_rev > 0) {
                    l_rev -= 1;
                    zeroSlice(grad_beta_proj);
                    zeroSlice(grad_key);
                    zeroSlice(grad_p_raw);
                    var grad_beta_scalar_total: f32 = 0.0;
                    for (0..self.num_groups) |g| {
                        const group_offset = g * self.head_dim;
                        const group_state_offset = g * state_matrix_size;
                        const state_before = iter_state_cache[l_rev * token_state_span + group_state_offset ..][0..state_matrix_size];
                        const state_after = iter_state_cache[(l_rev + 1) * token_state_span + group_state_offset ..][0..state_matrix_size];
                        const grad_state_after_group = grad_state_after[group_state_offset..][0..state_matrix_size];
                        const grad_state_before_group = grad_state_before[group_state_offset..][0..state_matrix_size];
                        for (0..state_matrix_size) |idx| grad_state_before_group[idx] = 0.0;
                        const key = key_cache[l_rev * self.head_dim ..][0..self.head_dim];
                        const key_raw_vec = key_raw_cache[l_rev * self.head_dim ..][0..self.head_dim];
                        const p_vec_iter = p_cache[l_rev * self.head_dim ..][0..self.head_dim];
                        const beta_scalar = beta_scalar_cache[l_rev];
                        for (0..self.head_dim) |row| {
                            var sum: f32 = 0.0;
                            for (0..self.head_dim) |col| sum += grad_state_after_group[row * self.head_dim + col] * key[col];
                            temp_gk[row] = sum;
                        }
                        for (0..self.head_dim) |col| {
                            var sum: f32 = 0.0;
                            for (0..self.head_dim) |row| sum += grad_state_after_group[row * self.head_dim + col] * delta_cache[(l_rev * self.num_groups + g) * self.head_dim + row];
                            temp_gt[col] = sum;
                        }
                        for (0..self.head_dim) |d| grad_delta[d] = grad_output_ptr[token_offset + group_offset + d].toFloat32() - grad_residual_after[group_offset + d];
                        if (l_rev == 0) {
                            for (0..self.head_dim) |row| {
                                var sum: f32 = 0.0;
                                for (0..self.head_dim) |col| sum += state_before[row * self.head_dim + col] * key[col];
                                state_times_key_local[row] = sum;
                            }
                            for (0..state_matrix_size) |idx| grad_state_before_group[idx] += self.alpha * grad_state_after_group[idx];
                            for (0..self.head_dim) |row| {
                                grad_delta[row] += beta_scalar * temp_gk[row];
                                grad_beta_scalar_total += delta_cache[(l_rev * self.num_groups + g) * self.head_dim + row] * temp_gk[row];
                                const grad_a = -self.alpha * beta_scalar * temp_gk[row];
                                grad_beta_scalar_total += -self.alpha * state_times_key_local[row] * temp_gk[row];
                                for (0..self.head_dim) |col| {
                                    grad_state_before_group[row * self.head_dim + col] += grad_a * key[col];
                                    grad_key[col] += grad_a * state_before[row * self.head_dim + col];
                                }
                            }
                            for (0..self.head_dim) |col| grad_key[col] += beta_scalar * temp_gt[col];
                            for (0..self.head_dim) |col| {
                                var sum: f32 = 0.0;
                                for (0..self.head_dim) |row| sum += grad_state_after_group[row * self.head_dim + col] * state_times_key_local[row];
                                grad_key[col] += -self.alpha * beta_scalar * sum;
                            }
                        } else {
                            for (0..state_matrix_size) |idx| grad_state_before_group[idx] += grad_state_after_group[idx];
                            for (0..self.head_dim) |row| {
                                grad_delta[row] += beta_scalar * temp_gk[row];
                                grad_beta_scalar_total += delta_cache[(l_rev * self.num_groups + g) * self.head_dim + row] * temp_gk[row];
                            }
                            for (0..self.head_dim) |col| grad_key[col] += beta_scalar * temp_gt[col];
                        }
                        for (0..self.head_dim) |d| {
                            const cache_idx = (l_rev * self.num_groups + g) * self.head_dim + d;
                            const z = p_vec_iter[d] * residual_before_cache[cache_idx];
                            const grad_z = grad_delta[d] * geluApproxDerivative(z);
                            grad_p_raw[d] += grad_z * residual_before_cache[cache_idx] * p_vec_iter[d] * (1.0 - p_vec_iter[d]);
                            grad_residual_before[group_offset + d] = grad_residual_after[group_offset + d] + grad_z * p_vec_iter[d];
                        }
                    }
                    const beta_factor = beta_scalar_cache[l_rev] * (1.0 - beta_scalar_cache[l_rev]) / @as(f32, @floatFromInt(self.head_dim));
                    for (0..self.head_dim) |d| grad_beta_proj[d] = grad_beta_scalar_total * beta_factor;
                    zeroSlice(grad_key_raw);
                    backpropNormalize(grad_key_raw, key_cache[l_rev * self.head_dim ..][0..self.head_dim], key_raw_cache[l_rev * self.head_dim ..][0..self.head_dim], grad_key, key_norm_cache[l_rev]);
                    const grad_w_beta_ptr = grad_w_beta[l_rev].typedPtr(dtype_mod.BF16) orelse return error.InvalidDType;
                    const grad_w_k_ptr = grad_w_k[l_rev].typedPtr(dtype_mod.BF16) orelse return error.InvalidDType;
                    const grad_w_p_ptr = grad_w_p[l_rev].typedPtr(dtype_mod.BF16) orelse return error.InvalidDType;
                    for (0..self.hidden_dim) |i| {
                        const u_val = u_ptr[token_offset + i].toFloat32();
                        for (0..self.head_dim) |d| {
                            const idx = i * self.head_dim + d;
                            grad_w_beta_ptr[idx] = dtype_mod.BF16.fromFloat32(grad_w_beta_ptr[idx].toFloat32() + u_val * grad_beta_proj[d]);
                            grad_w_k_ptr[idx] = dtype_mod.BF16.fromFloat32(grad_w_k_ptr[idx].toFloat32() + u_val * grad_key_raw[d]);
                            grad_w_p_ptr[idx] = dtype_mod.BF16.fromFloat32(grad_w_p_ptr[idx].toFloat32() + u_val * grad_p_raw[d]);
                        }
                    }
                    try accumulateMatVecTransposeToInput(grad_u_token, self.w_beta[l_rev], grad_beta_proj, self.hidden_dim, self.head_dim);
                    try accumulateMatVecTransposeToInput(grad_u_token, self.w_k[l_rev], grad_key_raw, self.hidden_dim, self.head_dim);
                    try accumulateMatVecTransposeToInput(grad_u_token, self.w_p[l_rev], grad_p_raw, self.hidden_dim, self.head_dim);
                    @memcpy(grad_residual_after, grad_residual_before);
                    @memcpy(grad_state_after, grad_state_before);
                }
                for (0..self.num_groups) |g| {
                    const group_offset = g * self.head_dim;
                    const group_state_offset = g * state_matrix_size;
                    for (0..self.head_dim) |d| {
                        grad_u_token[group_offset + d] -= grad_residual_after[group_offset + d];
                        grad_v_token[group_offset + d] += grad_residual_after[group_offset + d];
                    }
                    if (t_rev == 0) {
                        for (0..self.head_dim) |row| {
                            for (0..self.head_dim) |col| {
                                grad_state_ptr[stateElementOffset(self.num_groups, self.head_dim, b, g, row, col)] = dtype_mod.BF16.fromFloat32(grad_state_after[group_state_offset + row * self.head_dim + col]);
                            }
                        }
                    }
                }
                for (0..self.hidden_dim) |d| {
                    grad_u_full[token_offset + d] = grad_u_token[d];
                    grad_v_ptr[token_offset + d] = dtype_mod.BF16.fromFloat32(grad_v_token[d]);
                    if (grad_bias_ptr) |bp| bp[d] = dtype_mod.BF16.fromFloat32(bp[d].toFloat32() + grad_u_token[d]);
                }
            }
        }
        for (0..batch_size) |b| {
            for (0..seq_len) |t| {
                for (0..self.hidden_dim) |d| {
                    var sum: f32 = 0.0;
                    for (0..self.shortconv.window_size) |w| {
                        const out_t = t + w;
                        if (out_t < seq_len) sum += grad_u_full[(b * seq_len + out_t) * self.hidden_dim + d] * weight_ptr[d * self.shortconv.window_size + w].toFloat32();
                    }
                    grad_input_ptr[(b * seq_len + t) * self.hidden_dim + d] = dtype_mod.BF16.fromFloat32(sum);
                }
            }
        }
        for (0..batch_size) |b| {
            for (0..seq_len) |t| {
                for (0..self.hidden_dim) |d| {
                    for (0..self.shortconv.window_size) |w| {
                        const lookback = @as(isize, @intCast(t)) - @as(isize, @intCast(w));
                        if (lookback >= 0) {
                            const src_t = @as(usize, @intCast(lookback));
                            const grad_u_val = grad_u_full[(b * seq_len + t) * self.hidden_dim + d];
                            const in_val = input_ptr[(b * seq_len + src_t) * self.hidden_dim + d].toFloat32();
                            const idx = d * self.shortconv.window_size + w;
                            grad_sc_ptr[idx] = dtype_mod.BF16.fromFloat32(grad_sc_ptr[idx].toFloat32() + grad_u_val * in_val);
                        }
                    }
                }
            }
        }
        try storeGrad(self.shortconv.weight, self.allocator, grad_shortconv_weight);
        if (self.shortconv.bias) |bias| {
            const grad_bias = grad_shortconv_bias orelse return error.GradientNotAvailable;
            try storeGrad(bias, self.allocator, grad_bias);
        }
        for (0..self.num_iterations) |l| {
            try storeGrad(self.w_beta[l], self.allocator, grad_w_beta[l]);
            try storeGrad(self.w_k[l], self.allocator, grad_w_k[l]);
            try storeGrad(self.w_p[l], self.allocator, grad_w_p[l]);
        }
        return .{ .grad_input = grad_input, .grad_v = grad_v, .grad_state = grad_state };
    }
};

pub const ShortConv = struct {
    window_size: usize,
    hidden_dim: usize,
    weight: *Tensor,
    bias: ?*Tensor,
    allocator: std.mem.Allocator,
    device: tensor_mod.Device,
    device_id: i32,

    pub fn init(allocator: std.mem.Allocator, hidden_dim: usize, window_size: usize, device: tensor_mod.Device, device_id: i32, rng: *std.Random) !*ShortConv {
        if (hidden_dim == 0) return error.InvalidHiddenDimension;
        if (window_size == 0) return error.InvalidWindowSize;
        const self = try allocator.create(ShortConv);
        errdefer allocator.destroy(self);
        const weight_shape = Shape.init(&[_]usize{ hidden_dim, window_size });
        const weight = try Tensor.randNormal(allocator, weight_shape, .bf16, device, device_id, rng, 0.0, @sqrt(1.0 / @as(f64, @floatFromInt(window_size))));
        errdefer weight.deinit();
        self.* = .{ .window_size = window_size, .hidden_dim = hidden_dim, .weight = weight, .bias = null, .allocator = allocator, .device = device, .device_id = device_id };
        return self;
    }

    pub fn deinit(self: *ShortConv) void {
        self.weight.deinit();
        if (self.bias) |b| b.deinit();
        self.allocator.destroy(self);
    }

    pub fn forward(self: *ShortConv, input: *Tensor) !*Tensor {
        if (input.device != self.device or input.device_id != self.device_id) return error.DeviceMismatch;
        if (input.dtype != .bf16 or self.weight.dtype != .bf16) return error.DTypeMismatch;
        if (input.shape.ndim != 3) return error.InvalidInputShape;
        if (input.shape.dim(2) != self.hidden_dim) return error.InvalidInputShape;
        if (self.weight.device != self.device or self.weight.device_id != self.device_id) return error.DeviceMismatch;
        if (self.weight.shape.ndim != 2 or self.weight.shape.dim(0) != self.hidden_dim or self.weight.shape.dim(1) != self.window_size) return error.InvalidWeightShape;
        if (self.bias) |b| {
            if (b.device != self.device or b.device_id != self.device_id) return error.DeviceMismatch;
            if (b.dtype != .bf16) return error.DTypeMismatch;
            if (b.shape.ndim != 1 or b.shape.dim(0) != self.hidden_dim) return error.InvalidBiasShape;
        }
        if (self.device != .cpu) return self.forwardViaCpu(input);
        return self.forwardCpuOwned(input);
    }

    fn forwardViaCpu(self: *ShortConv, input: *Tensor) !*Tensor {
        var input_cpu = try cloneTensorToCpu(self.allocator, input);
        defer input_cpu.deinit();
        const weight_cpu = try cloneTensorToCpu(self.allocator, self.weight);
        defer weight_cpu.deinit();
        const bias_cpu = try cloneOptionalTensorToCpu(self.allocator, self.bias);
        defer if (bias_cpu) |b| b.deinit();
        var cpu_conv = ShortConv{
            .window_size = self.window_size,
            .hidden_dim = self.hidden_dim,
            .weight = weight_cpu,
            .bias = bias_cpu,
            .allocator = self.allocator,
            .device = .cpu,
            .device_id = 0,
        };
        var output_cpu = try cpu_conv.forwardCpuOwned(input_cpu);
        defer output_cpu.deinit();
        return try output_cpu.to(self.allocator, self.device, self.device_id);
    }

    fn forwardCpuOwned(self: *ShortConv, input: *Tensor) !*Tensor {
        const batch_size = input.shape.dim(0);
        const seq_len = input.shape.dim(1);
        const output_shape = Shape.init(&[_]usize{ batch_size, seq_len, self.hidden_dim });
        var output = try Tensor.zeros(self.allocator, output_shape, .bf16, self.device, self.device_id);
        errdefer output.deinit();
        try self.forwardCpu(input, output);
        return output;
    }

    fn forwardCpu(self: *ShortConv, input: *Tensor, output: *Tensor) !void {
        const batch_size = input.shape.dim(0);
        const seq_len = input.shape.dim(1);
        const input_ptr = input.typedPtr(dtype_mod.BF16) orelse return error.InvalidDType;
        const weight_ptr = self.weight.typedPtr(dtype_mod.BF16) orelse return error.InvalidDType;
        const output_ptr = output.typedPtr(dtype_mod.BF16) orelse return error.InvalidDType;
        const bias_ptr = if (self.bias) |b| blk: {
            break :blk b.typedPtr(dtype_mod.BF16) orelse return error.InvalidDType;
        } else null;
        for (0..batch_size) |b| {
            for (0..seq_len) |t| {
                for (0..self.hidden_dim) |d| {
                    var sum: f32 = 0.0;
                    for (0..self.window_size) |w| {
                        const lookback = @as(isize, @intCast(t)) - @as(isize, @intCast(w));
                        if (lookback >= 0) {
                            const t_lookback = @as(usize, @intCast(lookback));
                            sum += input_ptr[(b * seq_len + t_lookback) * self.hidden_dim + d].toFloat32() * weight_ptr[d * self.window_size + w].toFloat32();
                        }
                    }
                    if (bias_ptr) |bp| sum += bp[d].toFloat32();
                    output_ptr[(b * seq_len + t) * self.hidden_dim + d] = dtype_mod.BF16.fromFloat32(sum);
                }
            }
        }
    }
};

pub const PrismState = struct {
    state: *Tensor,
    batch_size: usize,
    num_groups: usize,
    state_dim: usize,
    value_dim: usize,
    allocator: std.mem.Allocator,
    device: tensor_mod.Device,
    device_id: i32,

    pub fn init(allocator: std.mem.Allocator, num_states: usize, state_dim: usize, value_dim: usize, device: tensor_mod.Device, device_id: i32) !*PrismState {
        return initWithGroups(allocator, num_states, 1, state_dim, value_dim, device, device_id);
    }

    pub fn initWithGroups(allocator: std.mem.Allocator, batch_size: usize, num_groups: usize, state_dim: usize, value_dim: usize, device: tensor_mod.Device, device_id: i32) !*PrismState {
        if (batch_size == 0) return error.InvalidNumStates;
        if (num_groups == 0) return error.InvalidNumGroups;
        if (state_dim == 0) return error.InvalidStateDimension;
        if (value_dim == 0) return error.InvalidValueDimension;
        const self = try allocator.create(PrismState);
        errdefer allocator.destroy(self);
        const state_shape = Shape.init(&[_]usize{ batch_size, num_groups, state_dim, value_dim });
        const state_tensor = try Tensor.zeros(allocator, state_shape, .bf16, device, device_id);
        errdefer state_tensor.deinit();
        self.* = .{
            .state = state_tensor,
            .batch_size = batch_size,
            .num_groups = num_groups,
            .state_dim = state_dim,
            .value_dim = value_dim,
            .allocator = allocator,
            .device = device,
            .device_id = device_id,
        };
        return self;
    }

    pub fn deinit(self: *PrismState) void {
        self.state.deinit();
        self.allocator.destroy(self);
    }

    pub fn reset(self: *PrismState) !void {
        try self.state.zero_();
    }

    pub fn clone(self: *PrismState) !*PrismState {
        const cloned_tensor = try self.state.to(self.allocator, self.device, self.device_id);
        errdefer cloned_tensor.deinit();
        const cloned = try self.allocator.create(PrismState);
        errdefer self.allocator.destroy(cloned);
        cloned.* = .{
            .state = cloned_tensor,
            .batch_size = self.batch_size,
            .num_groups = self.num_groups,
            .state_dim = self.state_dim,
            .value_dim = self.value_dim,
            .allocator = self.allocator,
            .device = self.device,
            .device_id = self.device_id,
        };
        return cloned;
    }

    pub fn to(self: *PrismState, allocator: std.mem.Allocator, device: tensor_mod.Device, device_id: i32) !*PrismState {
        const cloned_tensor = try self.state.to(allocator, device, device_id);
        errdefer cloned_tensor.deinit();
        const cloned = try allocator.create(PrismState);
        errdefer allocator.destroy(cloned);
        cloned.* = .{
            .state = cloned_tensor,
            .batch_size = self.batch_size,
            .num_groups = self.num_groups,
            .state_dim = self.state_dim,
            .value_dim = self.value_dim,
            .allocator = allocator,
            .device = device,
            .device_id = device_id,
        };
        return cloned;
    }
};

test "ShortConv forward" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    var prng = std.Random.DefaultPrng.init(42);
    var rng = prng.random();
    var conv = try ShortConv.init(gpa.allocator(), 64, 16, .cpu, 0, &rng);
    defer conv.deinit();
    const input_shape = Shape.init(&[_]usize{ 2, 32, 64 });
    var input = try Tensor.randNormal(gpa.allocator(), input_shape, .bf16, .cpu, 0, &rng, 0.0, 1.0);
    defer input.deinit();
    var output = try conv.forward(input);
    defer output.deinit();
    try std.testing.expectEqual(@as(usize, 2), output.shape.dim(0));
    try std.testing.expectEqual(@as(usize, 32), output.shape.dim(1));
    try std.testing.expectEqual(@as(usize, 64), output.shape.dim(2));
}

test "PRISM state init" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    var state = try PrismState.init(gpa.allocator(), 1, 64, 64, .cpu, 0);
    defer state.deinit();
    try std.testing.expectEqual(@as(usize, 1), state.batch_size);
    try std.testing.expectEqual(@as(usize, 1), state.num_groups);
    try std.testing.expectEqual(@as(usize, 64), state.state_dim);
}
