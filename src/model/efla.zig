const std = @import("std");
const tensor_mod = @import("../tensor/tensor.zig");
const dtype_mod = @import("../tensor/dtype.zig");
const config_mod = @import("../util/config.zig");
const kernels = @import("../kernels/efla_kernels.zig");

pub const Tensor = tensor_mod.Tensor;
pub const Shape = tensor_mod.Shape;
comptime {
    _ = kernels;
}

fn zeroSlice(s: []f32) void {
    for (s) |*v| v.* = 0.0;
}

fn storeGrad(param: *Tensor, allocator: std.mem.Allocator, grad: *Tensor) !void {
    const new_grad = try grad.to(allocator, param.device, param.device_id);
    if (param.grad) |existing| existing.deinit();
    param.grad = new_grad;
}

fn cloneTensorToCpu(allocator: std.mem.Allocator, tensor: *Tensor) !*Tensor {
    return try tensor.to(allocator, .cpu, 0);
}

fn stableCoefficient(beta: f32, lambda: f32) f32 {
    if (lambda < 1e-6) {
        const beta_sq = beta * beta;
        const beta_cu = beta_sq * beta;
        return beta - 0.5 * beta_sq * lambda + (beta_cu * lambda * lambda) / 6.0;
    }
    return (1.0 - @exp(-beta * lambda)) / lambda;
}

fn stableCoefficientDerivatives(beta: f32, lambda: f32) struct { dc_dbeta: f32, dc_dlambda: f32 } {
    if (lambda < 1e-6) {
        const beta_sq = beta * beta;
        return .{
            .dc_dbeta = 1.0 - beta * lambda + 0.5 * beta_sq * lambda * lambda,
            .dc_dlambda = -0.5 * beta_sq + (beta_sq * beta * lambda) / 3.0,
        };
    }
    const exp_term = @exp(-beta * lambda);
    return .{
        .dc_dbeta = exp_term,
        .dc_dlambda = (beta * lambda * exp_term - (1.0 - exp_term)) / (lambda * lambda),
    };
}

pub const EflaState = struct {
    state: *Tensor,
    batch_size: usize,
    num_heads: usize,
    state_dim: usize,
    value_dim: usize,
    allocator: std.mem.Allocator,
    device: tensor_mod.Device,
    device_id: i32,

    pub fn init(allocator: std.mem.Allocator, num_heads: usize, state_dim: usize, value_dim: usize, device: tensor_mod.Device, device_id: i32) !*EflaState {
        return initWithBatch(allocator, 1, num_heads, state_dim, value_dim, device, device_id);
    }

    pub fn initWithBatch(allocator: std.mem.Allocator, batch_size: usize, num_heads: usize, state_dim: usize, value_dim: usize, device: tensor_mod.Device, device_id: i32) !*EflaState {
        if (batch_size == 0 or num_heads == 0 or state_dim == 0 or value_dim == 0) return error.InvalidConfiguration;
        const self = try allocator.create(EflaState);
        errdefer allocator.destroy(self);
        const state_shape = Shape.init(&[_]usize{ batch_size, num_heads, state_dim, value_dim });
        const state_tensor = try Tensor.zeros(allocator, state_shape, .bf16, device, device_id);
        errdefer state_tensor.deinit();
        self.* = .{
            .state = state_tensor,
            .batch_size = batch_size,
            .num_heads = num_heads,
            .state_dim = state_dim,
            .value_dim = value_dim,
            .allocator = allocator,
            .device = device,
            .device_id = device_id,
        };
        return self;
    }

    pub fn deinit(self: *EflaState) void {
        self.state.deinit();
        self.allocator.destroy(self);
    }

    pub fn reset(self: *EflaState) !void {
        try self.state.zero_();
    }

    pub fn clone(self: *EflaState) !*EflaState {
        const new_state = try self.state.to(self.allocator, self.device, self.device_id);
        errdefer new_state.deinit();
        const cloned = try self.allocator.create(EflaState);
        errdefer self.allocator.destroy(cloned);
        cloned.* = .{
            .state = new_state,
            .batch_size = self.batch_size,
            .num_heads = self.num_heads,
            .state_dim = self.state_dim,
            .value_dim = self.value_dim,
            .allocator = self.allocator,
            .device = self.device,
            .device_id = self.device_id,
        };
        return cloned;
    }

    pub fn to(self: *EflaState, allocator: std.mem.Allocator, device: tensor_mod.Device, device_id: i32) !*EflaState {
        const new_state = try self.state.to(allocator, device, device_id);
        errdefer new_state.deinit();
        const cloned = try allocator.create(EflaState);
        errdefer allocator.destroy(cloned);
        cloned.* = .{
            .state = new_state,
            .batch_size = self.batch_size,
            .num_heads = self.num_heads,
            .state_dim = self.state_dim,
            .value_dim = self.value_dim,
            .allocator = allocator,
            .device = device,
            .device_id = device_id,
        };
        return cloned;
    }
};

pub const EflaLayer = struct {
    config: config_mod.EflaConfig,
    hidden_dim: usize,
    num_heads: usize,
    head_dim: usize,
    w_k: *Tensor,
    w_v: *Tensor,
    w_o: *Tensor,
    beta_param: ?*Tensor,
    allocator: std.mem.Allocator,
    device: tensor_mod.Device,
    device_id: i32,

    pub fn init(allocator: std.mem.Allocator, config: config_mod.EflaConfig, hidden_dim: usize, num_heads: usize, head_dim: usize, device: tensor_mod.Device, device_id: i32, rng: *std.Random) !*EflaLayer {
        if (hidden_dim == 0 or num_heads == 0 or head_dim == 0 or hidden_dim != num_heads * head_dim or config.chunk_size == 0) return error.InvalidConfiguration;
        const self = try allocator.create(EflaLayer);
        errdefer allocator.destroy(self);
        const scale = @sqrt(2.0 / @as(f64, @floatFromInt(hidden_dim)));
        const w_k_shape = Shape.init(&[_]usize{ hidden_dim, num_heads * head_dim });
        const w_k = try Tensor.randNormal(allocator, w_k_shape, .bf16, device, device_id, rng, 0.0, @floatCast(scale));
        errdefer w_k.deinit();
        const w_v_shape = Shape.init(&[_]usize{ hidden_dim, num_heads * head_dim });
        const w_v = try Tensor.randNormal(allocator, w_v_shape, .bf16, device, device_id, rng, 0.0, @floatCast(scale));
        errdefer w_v.deinit();
        const w_o_shape = Shape.init(&[_]usize{ num_heads * head_dim, hidden_dim });
        const w_o = try Tensor.randNormal(allocator, w_o_shape, .bf16, device, device_id, rng, 0.0, @floatCast(scale));
        errdefer w_o.deinit();
        var beta_param: ?*Tensor = null;
        if (config.learned_beta) {
            const beta_shape = Shape.init(&[_]usize{1});
            const bp = try Tensor.full(allocator, beta_shape, .f32, device, device_id, config.initial_beta);
            errdefer bp.deinit();
            beta_param = bp;
        }
        self.* = .{
            .config = config,
            .hidden_dim = hidden_dim,
            .num_heads = num_heads,
            .head_dim = head_dim,
            .w_k = w_k,
            .w_v = w_v,
            .w_o = w_o,
            .beta_param = beta_param,
            .allocator = allocator,
            .device = device,
            .device_id = device_id,
        };
        return self;
    }

    pub fn deinit(self: *EflaLayer) void {
        self.w_k.deinit();
        self.w_v.deinit();
        self.w_o.deinit();
        if (self.beta_param) |bp| bp.deinit();
        self.allocator.destroy(self);
    }

    fn betaValue(self: *EflaLayer) !f32 {
        if (self.beta_param) |bp| {
            if (bp.dtype != .f32) return error.DTypeMismatch;
            const beta_ptr = bp.typedPtr(f32) orelse return error.InvalidDType;
            return beta_ptr[0];
        }
        return self.config.initial_beta;
    }

    fn validateForwardInputs(self: *EflaLayer, input: *Tensor, state: ?*EflaState) !void {
        if (input.device != self.device or input.device_id != self.device_id) return error.DeviceMismatch;
        if (input.dtype != .bf16) return error.DTypeMismatch;
        if (input.shape.ndim != 3 or input.shape.dim(2) != self.hidden_dim) return error.ShapeMismatch;
        if (state) |s| {
            if (s.device != self.device or s.device_id != self.device_id) return error.DeviceMismatch;
            if (s.state.dtype != .bf16) return error.DTypeMismatch;
            if (s.state.shape.ndim != 4) return error.StateShapeMismatch;
            if (s.batch_size != input.shape.dim(0) or s.num_heads != self.num_heads or s.state_dim != self.head_dim or s.value_dim != self.head_dim) return error.StateShapeMismatch;
            if (s.state.shape.dim(0) != input.shape.dim(0) or s.state.shape.dim(1) != self.num_heads or s.state.shape.dim(2) != self.head_dim or s.state.shape.dim(3) != self.head_dim) return error.StateShapeMismatch;
        }
    }

    fn validateBackwardInputs(self: *EflaLayer, grad_output: *Tensor, input: *Tensor, state: *EflaState) !void {
        try self.validateForwardInputs(input, state);
        if (grad_output.device != self.device or grad_output.device_id != self.device_id) return error.DeviceMismatch;
        if (grad_output.dtype != .bf16) return error.DTypeMismatch;
        if (grad_output.shape.ndim != 3) return error.ShapeMismatch;
        if (grad_output.shape.dim(0) != input.shape.dim(0) or grad_output.shape.dim(1) != input.shape.dim(1) or grad_output.shape.dim(2) != self.hidden_dim) return error.ShapeMismatch;
    }

    pub fn forward(self: *EflaLayer, input: *Tensor, state: ?*EflaState) !struct { output: *Tensor, new_state: *EflaState } {
        try self.validateForwardInputs(input, state);
        if (self.device != .cpu) return self.forwardViaCpu(input, state);
        return self.forwardCpuImpl(input, state);
    }

    fn forwardViaCpu(self: *EflaLayer, input: *Tensor, state: ?*EflaState) !struct { output: *Tensor, new_state: *EflaState } {
        var input_cpu = try cloneTensorToCpu(self.allocator, input);
        defer input_cpu.deinit();
        var state_cpu: ?*EflaState = null;
        defer if (state_cpu) |s| s.deinit();
        if (state) |s| state_cpu = try s.to(self.allocator, .cpu, 0);
        const w_k_cpu = try cloneTensorToCpu(self.allocator, self.w_k);
        defer w_k_cpu.deinit();
        const w_v_cpu = try cloneTensorToCpu(self.allocator, self.w_v);
        defer w_v_cpu.deinit();
        const w_o_cpu = try cloneTensorToCpu(self.allocator, self.w_o);
        defer w_o_cpu.deinit();
        const beta_cpu = if (self.beta_param) |bp| try cloneTensorToCpu(self.allocator, bp) else null;
        defer if (beta_cpu) |b| b.deinit();
        var cpu_layer = self.*;
        cpu_layer.w_k = w_k_cpu;
        cpu_layer.w_v = w_v_cpu;
        cpu_layer.w_o = w_o_cpu;
        cpu_layer.beta_param = beta_cpu;
        cpu_layer.device = .cpu;
        cpu_layer.device_id = 0;
        var cpu_result = try cpu_layer.forwardCpuImpl(input_cpu, state_cpu);
        defer cpu_result.output.deinit();
        defer cpu_result.new_state.deinit();
        const output = try cpu_result.output.to(self.allocator, self.device, self.device_id);
        errdefer output.deinit();
        const new_state = try cpu_result.new_state.to(self.allocator, self.device, self.device_id);
        errdefer new_state.deinit();
        return .{ .output = output, .new_state = new_state };
    }

    fn forwardCpuImpl(self: *EflaLayer, input: *Tensor, state: ?*EflaState) !struct { output: *Tensor, new_state: *EflaState } {
        const batch_size = input.shape.dim(0);
        const seq_len = input.shape.dim(1);
        const k = try self.matmulCpu(input, self.w_k);
        defer k.deinit();
        const v = try self.matmulCpu(input, self.w_v);
        defer v.deinit();
        const k_reshaped = try k.reshape(Shape.init(&[_]usize{ batch_size, seq_len, self.num_heads, self.head_dim }));
        defer k_reshaped.deinit();
        const v_reshaped = try v.reshape(Shape.init(&[_]usize{ batch_size, seq_len, self.num_heads, self.head_dim }));
        defer v_reshaped.deinit();
        var new_state = if (state) |previous_state| try previous_state.clone() else try EflaState.initWithBatch(self.allocator, batch_size, self.num_heads, self.head_dim, self.head_dim, self.device, self.device_id);
        errdefer new_state.deinit();
        const output_heads = try self.eflaForwardCpu(k_reshaped, v_reshaped, new_state);
        defer output_heads.deinit();
        const projected = try self.matmulCpu(output_heads, self.w_o);
        return .{ .output = projected, .new_state = new_state };
    }

    fn eflaForwardCpu(self: *EflaLayer, k: *Tensor, v: *Tensor, state: *EflaState) !*Tensor {
        if (state.batch_size != k.shape.dim(0) or state.num_heads != self.num_heads or state.state_dim != self.head_dim or state.value_dim != self.head_dim) return error.StateShapeMismatch;
        const batch_size = k.shape.dim(0);
        const seq_len = k.shape.dim(1);
        const output_shape = Shape.init(&[_]usize{ batch_size, seq_len, self.num_heads * self.head_dim });
        const output = try Tensor.init(self.allocator, output_shape, .bf16, self.device, self.device_id);
        errdefer output.deinit();
        const beta = try self.betaValue();
        const k_ptr = k.typedPtr(dtype_mod.BF16) orelse return error.InvalidDType;
        const v_ptr = v.typedPtr(dtype_mod.BF16) orelse return error.InvalidDType;
        const s_ptr = state.state.typedPtr(dtype_mod.BF16) orelse return error.InvalidDType;
        const o_ptr = output.typedPtr(dtype_mod.BF16) orelse return error.InvalidDType;
        const state_matrix_size = self.head_dim * self.head_dim;
        var previous_state = try self.allocator.alloc(f32, state_matrix_size);
        defer self.allocator.free(previous_state);
        var projected_state = try self.allocator.alloc(f32, self.head_dim);
        defer self.allocator.free(projected_state);
        var current_state = try self.allocator.alloc(f32, state_matrix_size);
        defer self.allocator.free(current_state);
        for (0..batch_size) |b| {
            for (0..self.num_heads) |h| {
                const state_offset = ((b * self.num_heads) + h) * state_matrix_size;
                for (0..state_matrix_size) |idx| current_state[idx] = s_ptr[state_offset + idx].toFloat32();
                for (0..seq_len) |t| {
                    const token_offset = ((b * seq_len + t) * self.num_heads + h) * self.head_dim;
                    for (0..state_matrix_size) |idx| previous_state[idx] = current_state[idx];
                    var lambda: f32 = 0.0;
                    for (0..self.head_dim) |i| {
                        const k_i = k_ptr[token_offset + i].toFloat32();
                        lambda += k_i * k_i;
                    }
                    const c_t = stableCoefficient(beta, lambda);
                    for (0..self.head_dim) |j| {
                        var sum: f32 = 0.0;
                        for (0..self.head_dim) |i| sum += k_ptr[token_offset + i].toFloat32() * previous_state[i * self.head_dim + j];
                        projected_state[j] = sum;
                    }
                    for (0..self.head_dim) |i| {
                        const k_i = k_ptr[token_offset + i].toFloat32();
                        for (0..self.head_dim) |j| {
                            const v_j = v_ptr[token_offset + j].toFloat32();
                            current_state[i * self.head_dim + j] = previous_state[i * self.head_dim + j] + c_t * k_i * (v_j - projected_state[j]);
                        }
                    }
                    for (0..self.head_dim) |j| {
                        var sum: f32 = 0.0;
                        for (0..self.head_dim) |i| sum += current_state[i * self.head_dim + j] * k_ptr[token_offset + i].toFloat32();
                        o_ptr[token_offset + j] = dtype_mod.BF16.fromFloat32(sum);
                    }
                }
                for (0..state_matrix_size) |idx| s_ptr[state_offset + idx] = dtype_mod.BF16.fromFloat32(current_state[idx]);
            }
        }
        return output;
    }

    pub fn backward(self: *EflaLayer, grad_output: *Tensor, input: *Tensor, state: *EflaState) !struct { grad_input: *Tensor, grad_state: *EflaState } {
        try self.validateBackwardInputs(grad_output, input, state);
        if (self.device != .cpu) return self.backwardViaCpu(grad_output, input, state);
        return self.backwardCpu(grad_output, input, state);
    }

    fn backwardViaCpu(self: *EflaLayer, grad_output: *Tensor, input: *Tensor, state: *EflaState) !struct { grad_input: *Tensor, grad_state: *EflaState } {
        var grad_output_cpu = try cloneTensorToCpu(self.allocator, grad_output);
        defer grad_output_cpu.deinit();
        var input_cpu = try cloneTensorToCpu(self.allocator, input);
        defer input_cpu.deinit();
        var state_cpu = try state.to(self.allocator, .cpu, 0);
        defer state_cpu.deinit();
        const w_k_cpu = try cloneTensorToCpu(self.allocator, self.w_k);
        defer w_k_cpu.deinit();
        const w_v_cpu = try cloneTensorToCpu(self.allocator, self.w_v);
        defer w_v_cpu.deinit();
        const w_o_cpu = try cloneTensorToCpu(self.allocator, self.w_o);
        defer w_o_cpu.deinit();
        const beta_cpu = if (self.beta_param) |bp| try cloneTensorToCpu(self.allocator, bp) else null;
        defer if (beta_cpu) |b| b.deinit();
        var cpu_layer = self.*;
        cpu_layer.w_k = w_k_cpu;
        cpu_layer.w_v = w_v_cpu;
        cpu_layer.w_o = w_o_cpu;
        cpu_layer.beta_param = beta_cpu;
        cpu_layer.device = .cpu;
        cpu_layer.device_id = 0;
        var cpu_result = try cpu_layer.backwardCpu(grad_output_cpu, input_cpu, state_cpu);
        defer cpu_result.grad_input.deinit();
        defer cpu_result.grad_state.deinit();
        try storeGrad(self.w_k, self.allocator, cpu_layer.w_k.grad orelse return error.GradientNotAvailable);
        try storeGrad(self.w_v, self.allocator, cpu_layer.w_v.grad orelse return error.GradientNotAvailable);
        try storeGrad(self.w_o, self.allocator, cpu_layer.w_o.grad orelse return error.GradientNotAvailable);
        if (self.beta_param) |beta_param| {
            const cpu_beta = cpu_layer.beta_param orelse return error.GradientNotAvailable;
            const beta_grad = cpu_beta.grad orelse return error.GradientNotAvailable;
            try storeGrad(beta_param, self.allocator, beta_grad);
        }
        const grad_input = try cpu_result.grad_input.to(self.allocator, self.device, self.device_id);
        errdefer grad_input.deinit();
        const grad_state_dev = try cpu_result.grad_state.to(self.allocator, self.device, self.device_id);
        errdefer grad_state_dev.deinit();
        return .{ .grad_input = grad_input, .grad_state = grad_state_dev };
    }

    fn backwardCpu(self: *EflaLayer, grad_output: *Tensor, input: *Tensor, state: *EflaState) !struct { grad_input: *Tensor, grad_state: *EflaState } {
        const batch_size = input.shape.dim(0);
        const seq_len = input.shape.dim(1);
        const state_matrix_size = self.head_dim * self.head_dim;
        const k_flat = try self.matmulCpu(input, self.w_k);
        defer k_flat.deinit();
        const v_flat = try self.matmulCpu(input, self.w_v);
        defer v_flat.deinit();
        const k = try k_flat.reshape(Shape.init(&[_]usize{ batch_size, seq_len, self.num_heads, self.head_dim }));
        defer k.deinit();
        const v = try v_flat.reshape(Shape.init(&[_]usize{ batch_size, seq_len, self.num_heads, self.head_dim }));
        defer v.deinit();
        var state_forward = try state.clone();
        defer state_forward.deinit();
        const output_heads = try self.eflaForwardCpu(k, v, state_forward);
        defer output_heads.deinit();
        const grad_pre_o = try Tensor.zeros(self.allocator, output_heads.shape, .bf16, self.device, self.device_id);
        defer grad_pre_o.deinit();
        var grad_w_o = try Tensor.zeros(self.allocator, self.w_o.shape, .bf16, self.device, self.device_id);
        defer grad_w_o.deinit();
        const grad_output_ptr = grad_output.typedPtr(dtype_mod.BF16) orelse return error.InvalidDType;
        const output_heads_ptr = output_heads.typedPtr(dtype_mod.BF16) orelse return error.InvalidDType;
        const w_o_ptr = self.w_o.typedPtr(dtype_mod.BF16) orelse return error.InvalidDType;
        const grad_pre_o_ptr = grad_pre_o.typedPtr(dtype_mod.BF16) orelse return error.InvalidDType;
        const grad_w_o_ptr = grad_w_o.typedPtr(dtype_mod.BF16) orelse return error.InvalidDType;
        for (0..grad_w_o.shape.numel()) |i| grad_w_o_ptr[i] = dtype_mod.BF16.fromFloat32(0.0);
        const flat_dim = self.num_heads * self.head_dim;
        for (0..batch_size) |b| {
            for (0..seq_len) |t| {
                const token_offset_hidden = (b * seq_len + t) * self.hidden_dim;
                const token_offset_flat = (b * seq_len + t) * flat_dim;
                for (0..flat_dim) |in_idx| {
                    var sum: f32 = 0.0;
                    for (0..self.hidden_dim) |out_idx| {
                        const g = grad_output_ptr[token_offset_hidden + out_idx].toFloat32();
                        sum += g * w_o_ptr[in_idx * self.hidden_dim + out_idx].toFloat32();
                        grad_w_o_ptr[in_idx * self.hidden_dim + out_idx] = dtype_mod.BF16.fromFloat32(grad_w_o_ptr[in_idx * self.hidden_dim + out_idx].toFloat32() + output_heads_ptr[token_offset_flat + in_idx].toFloat32() * g);
                    }
                    grad_pre_o_ptr[token_offset_flat + in_idx] = dtype_mod.BF16.fromFloat32(sum);
                }
            }
        }
        var token_start_states = try self.allocator.alloc(f32, batch_size * seq_len * self.num_heads * state_matrix_size);
        defer self.allocator.free(token_start_states);
        const k_ptr = k.typedPtr(dtype_mod.BF16) orelse return error.InvalidDType;
        const v_ptr = v.typedPtr(dtype_mod.BF16) orelse return error.InvalidDType;
        const state_ptr = state.state.typedPtr(dtype_mod.BF16) orelse return error.InvalidDType;
        var current_state = try self.allocator.alloc(f32, self.num_heads * state_matrix_size);
        defer self.allocator.free(current_state);
        var projected_state = try self.allocator.alloc(f32, self.head_dim);
        defer self.allocator.free(projected_state);
        const beta = try self.betaValue();
        for (0..batch_size) |b| {
            for (0..self.num_heads) |h| {
                const state_offset = ((b * self.num_heads) + h) * state_matrix_size;
                for (0..state_matrix_size) |idx| current_state[h * state_matrix_size + idx] = state_ptr[state_offset + idx].toFloat32();
            }
            for (0..seq_len) |t| {
                const token_state_offset = ((b * seq_len) + t) * self.num_heads * state_matrix_size;
                @memcpy(token_start_states[token_state_offset .. token_state_offset + self.num_heads * state_matrix_size], current_state);
                for (0..self.num_heads) |h| {
                    const head_offset = ((b * seq_len + t) * self.num_heads + h) * self.head_dim;
                    const current_head_state = current_state[h * state_matrix_size ..][0..state_matrix_size];
                    var lambda: f32 = 0.0;
                    for (0..self.head_dim) |i| {
                        const k_i = k_ptr[head_offset + i].toFloat32();
                        lambda += k_i * k_i;
                    }
                    const c_t = stableCoefficient(beta, lambda);
                    for (0..self.head_dim) |j| {
                        var sum: f32 = 0.0;
                        for (0..self.head_dim) |i| sum += k_ptr[head_offset + i].toFloat32() * current_head_state[i * self.head_dim + j];
                        projected_state[j] = sum;
                    }
                    for (0..self.head_dim) |i| {
                        const k_i = k_ptr[head_offset + i].toFloat32();
                        for (0..self.head_dim) |j| {
                            const v_j = v_ptr[head_offset + j].toFloat32();
                            current_head_state[i * self.head_dim + j] = current_head_state[i * self.head_dim + j] + c_t * k_i * (v_j - projected_state[j]);
                        }
                    }
                }
            }
        }
        var grad_k_flat = try Tensor.zeros(self.allocator, k_flat.shape, .bf16, self.device, self.device_id);
        defer grad_k_flat.deinit();
        var grad_v_flat = try Tensor.zeros(self.allocator, v_flat.shape, .bf16, self.device, self.device_id);
        defer grad_v_flat.deinit();
        var grad_state = try EflaState.initWithBatch(self.allocator, batch_size, self.num_heads, self.head_dim, self.head_dim, self.device, self.device_id);
        errdefer grad_state.deinit();
        const grad_pre_o_head_ptr = grad_pre_o.typedPtr(dtype_mod.BF16) orelse return error.InvalidDType;
        const grad_k_ptr = grad_k_flat.typedPtr(dtype_mod.BF16) orelse return error.InvalidDType;
        const grad_v_ptr = grad_v_flat.typedPtr(dtype_mod.BF16) orelse return error.InvalidDType;
        const grad_state_ptr = grad_state.state.typedPtr(dtype_mod.BF16) orelse return error.InvalidDType;
        for (0..grad_k_flat.shape.numel()) |i| {
            grad_k_ptr[i] = dtype_mod.BF16.fromFloat32(0.0);
            grad_v_ptr[i] = dtype_mod.BF16.fromFloat32(0.0);
        }
        for (0..grad_state.state.shape.numel()) |i| grad_state_ptr[i] = dtype_mod.BF16.fromFloat32(0.0);
        var grad_state_after = try self.allocator.alloc(f32, self.num_heads * state_matrix_size);
        defer self.allocator.free(grad_state_after);
        var grad_state_before = try self.allocator.alloc(f32, self.num_heads * state_matrix_size);
        defer self.allocator.free(grad_state_before);
        var s_prev = try self.allocator.alloc(f32, state_matrix_size);
        defer self.allocator.free(s_prev);
        var s_new = try self.allocator.alloc(f32, state_matrix_size);
        defer self.allocator.free(s_new);
        var p_vec = try self.allocator.alloc(f32, self.head_dim);
        defer self.allocator.free(p_vec);
        var q_vec = try self.allocator.alloc(f32, self.head_dim);
        defer self.allocator.free(q_vec);
        var tmp_vec = try self.allocator.alloc(f32, self.head_dim);
        defer self.allocator.free(tmp_vec);
        var grad_k_vec = try self.allocator.alloc(f32, self.head_dim);
        defer self.allocator.free(grad_k_vec);
        var grad_v_vec = try self.allocator.alloc(f32, self.head_dim);
        defer self.allocator.free(grad_v_vec);
        var grad_p_vec = try self.allocator.alloc(f32, self.head_dim);
        defer self.allocator.free(grad_p_vec);
        var grad_y_vec = try self.allocator.alloc(f32, self.head_dim);
        defer self.allocator.free(grad_y_vec);
        var grad_beta_total: f32 = 0.0;
        var grad_w_k = try Tensor.zeros(self.allocator, self.w_k.shape, .bf16, self.device, self.device_id);
        defer grad_w_k.deinit();
        var grad_w_v = try Tensor.zeros(self.allocator, self.w_v.shape, .bf16, self.device, self.device_id);
        defer grad_w_v.deinit();
        const grad_w_k_ptr = grad_w_k.typedPtr(dtype_mod.BF16) orelse return error.InvalidDType;
        const grad_w_v_ptr = grad_w_v.typedPtr(dtype_mod.BF16) orelse return error.InvalidDType;
        for (0..grad_w_k.shape.numel()) |i| {
            grad_w_k_ptr[i] = dtype_mod.BF16.fromFloat32(0.0);
            grad_w_v_ptr[i] = dtype_mod.BF16.fromFloat32(0.0);
        }
        for (0..batch_size) |b| {
            zeroSlice(grad_state_after);
            var token_rev: usize = seq_len;
            while (token_rev > 0) {
                token_rev -= 1;
                const token_state_offset = ((b * seq_len) + token_rev) * self.num_heads * state_matrix_size;
                const token_offset_flat = (b * seq_len + token_rev) * flat_dim;
                for (0..self.num_heads) |h| {
                    const head_offset = token_offset_flat + h * self.head_dim;
                    @memcpy(s_prev, token_start_states[token_state_offset + h * state_matrix_size .. token_state_offset + (h + 1) * state_matrix_size]);
                    var lambda: f32 = 0.0;
                    for (0..self.head_dim) |i| {
                        const k_i = k_ptr[head_offset + i].toFloat32();
                        lambda += k_i * k_i;
                    }
                    const c_t = stableCoefficient(beta, lambda);
                    const coeff_derivs = stableCoefficientDerivatives(beta, lambda);
                    for (0..self.head_dim) |j| {
                        var sum: f32 = 0.0;
                        for (0..self.head_dim) |i| sum += k_ptr[head_offset + i].toFloat32() * s_prev[i * self.head_dim + j];
                        p_vec[j] = sum;
                        q_vec[j] = v_ptr[head_offset + j].toFloat32() - sum;
                        grad_y_vec[j] = grad_pre_o_head_ptr[head_offset + j].toFloat32();
                    }
                    for (0..self.head_dim) |i| {
                        const k_i = k_ptr[head_offset + i].toFloat32();
                        for (0..self.head_dim) |j| s_new[i * self.head_dim + j] = s_prev[i * self.head_dim + j] + c_t * k_i * q_vec[j];
                    }
                    const grad_state_after_head = grad_state_after[h * state_matrix_size ..][0..state_matrix_size];
                    const grad_state_before_head = grad_state_before[h * state_matrix_size ..][0..state_matrix_size];
                    for (0..state_matrix_size) |idx| grad_state_before_head[idx] = grad_state_after_head[idx];
                    zeroSlice(grad_k_vec);
                    zeroSlice(grad_v_vec);
                    var grad_c: f32 = 0.0;
                    for (0..self.head_dim) |i| {
                        var sum: f32 = 0.0;
                        for (0..self.head_dim) |j| sum += s_new[i * self.head_dim + j] * grad_y_vec[j];
                        grad_k_vec[i] += sum;
                    }
                    for (0..self.head_dim) |i| {
                        const k_i = k_ptr[head_offset + i].toFloat32();
                        for (0..self.head_dim) |j| grad_state_before_head[i * self.head_dim + j] += k_i * grad_y_vec[j];
                    }
                    for (0..self.head_dim) |j| {
                        var sum: f32 = 0.0;
                        for (0..self.head_dim) |i| sum += grad_state_before_head[i * self.head_dim + j] * k_ptr[head_offset + i].toFloat32();
                        tmp_vec[j] = sum;
                    }
                    for (0..self.head_dim) |i| {
                        var sum_gq: f32 = 0.0;
                        for (0..self.head_dim) |j| sum_gq += grad_state_before_head[i * self.head_dim + j] * q_vec[j];
                        grad_k_vec[i] += c_t * sum_gq;
                    }
                    for (0..self.head_dim) |j| {
                        grad_v_vec[j] += c_t * tmp_vec[j];
                        grad_p_vec[j] = -c_t * tmp_vec[j];
                        grad_c += tmp_vec[j] * q_vec[j];
                    }
                    for (0..self.head_dim) |i| {
                        var sum: f32 = 0.0;
                        for (0..self.head_dim) |j| sum += s_prev[i * self.head_dim + j] * grad_p_vec[j];
                        grad_k_vec[i] += sum;
                    }
                    for (0..self.head_dim) |i| {
                        const k_i = k_ptr[head_offset + i].toFloat32();
                        for (0..self.head_dim) |j| grad_state_before_head[i * self.head_dim + j] += k_i * grad_p_vec[j];
                    }
                    const grad_lambda = grad_c * coeff_derivs.dc_dlambda;
                    grad_beta_total += grad_c * coeff_derivs.dc_dbeta;
                    for (0..self.head_dim) |i| grad_k_vec[i] += 2.0 * k_ptr[head_offset + i].toFloat32() * grad_lambda;
                    for (0..self.head_dim) |i| grad_k_ptr[head_offset + i] = dtype_mod.BF16.fromFloat32(grad_k_ptr[head_offset + i].toFloat32() + grad_k_vec[i]);
                    for (0..self.head_dim) |j| grad_v_ptr[head_offset + j] = dtype_mod.BF16.fromFloat32(grad_v_ptr[head_offset + j].toFloat32() + grad_v_vec[j]);
                }
                @memcpy(grad_state_after, grad_state_before);
            }
            for (0..self.num_heads) |h| {
                const base_state_offset = ((b * self.num_heads) + h) * state_matrix_size;
                for (0..state_matrix_size) |idx| grad_state_ptr[base_state_offset + idx] = dtype_mod.BF16.fromFloat32(grad_state_after[h * state_matrix_size + idx]);
            }
        }
        var grad_input = try Tensor.zeros(self.allocator, input.shape, .bf16, self.device, self.device_id);
        errdefer grad_input.deinit();
        const grad_input_ptr = grad_input.typedPtr(dtype_mod.BF16) orelse return error.InvalidDType;
        const input_ptr = input.typedPtr(dtype_mod.BF16) orelse return error.InvalidDType;
        const w_k_ptr = self.w_k.typedPtr(dtype_mod.BF16) orelse return error.InvalidDType;
        const w_v_ptr = self.w_v.typedPtr(dtype_mod.BF16) orelse return error.InvalidDType;
        for (0..grad_input.shape.numel()) |i| grad_input_ptr[i] = dtype_mod.BF16.fromFloat32(0.0);
        for (0..batch_size) |b| {
            for (0..seq_len) |t| {
                const token_offset_hidden = (b * seq_len + t) * self.hidden_dim;
                const token_offset_flat = (b * seq_len + t) * flat_dim;
                for (0..self.hidden_dim) |i| {
                    var sum: f32 = grad_input_ptr[token_offset_hidden + i].toFloat32();
                    for (0..flat_dim) |j| {
                        const gk = grad_k_ptr[token_offset_flat + j].toFloat32();
                        const gv = grad_v_ptr[token_offset_flat + j].toFloat32();
                        sum += gk * w_k_ptr[i * flat_dim + j].toFloat32();
                        sum += gv * w_v_ptr[i * flat_dim + j].toFloat32();
                    }
                    grad_input_ptr[token_offset_hidden + i] = dtype_mod.BF16.fromFloat32(sum);
                }
                for (0..self.hidden_dim) |i| {
                    const in_val = input_ptr[token_offset_hidden + i].toFloat32();
                    for (0..flat_dim) |j| {
                        const idx = i * flat_dim + j;
                        grad_w_k_ptr[idx] = dtype_mod.BF16.fromFloat32(grad_w_k_ptr[idx].toFloat32() + in_val * grad_k_ptr[token_offset_flat + j].toFloat32());
                        grad_w_v_ptr[idx] = dtype_mod.BF16.fromFloat32(grad_w_v_ptr[idx].toFloat32() + in_val * grad_v_ptr[token_offset_flat + j].toFloat32());
                    }
                }
            }
        }
        try storeGrad(self.w_k, self.allocator, grad_w_k);
        try storeGrad(self.w_v, self.allocator, grad_w_v);
        try storeGrad(self.w_o, self.allocator, grad_w_o);
        if (self.beta_param) |beta_param| {
            var beta_grad = try Tensor.zeros(self.allocator, beta_param.shape, beta_param.dtype, self.device, self.device_id);
            defer beta_grad.deinit();
            const beta_grad_ptr = beta_grad.typedPtr(f32) orelse return error.InvalidDType;
            beta_grad_ptr[0] = grad_beta_total;
            try storeGrad(beta_param, self.allocator, beta_grad);
        }
        return .{ .grad_input = grad_input, .grad_state = grad_state };
    }

    fn matmulCpu(self: *EflaLayer, a: *Tensor, b: *Tensor) !*Tensor {
        if (a.device != self.device or a.device_id != self.device_id or b.device != self.device or b.device_id != self.device_id) return error.DeviceMismatch;
        if (a.dtype != .bf16 or b.dtype != .bf16) return error.DTypeMismatch;
        if (b.shape.ndim != 2) return error.InvalidInputRank;
        if (a.shape.ndim != 2 and a.shape.ndim != 3) return error.InvalidInputRank;
        const k_dim = a.shape.dim(a.shape.ndim - 1);
        if (k_dim != b.shape.dim(0)) return error.ShapeMismatch;
        const out_shape = switch (a.shape.ndim) {
            2 => Shape.init(&[_]usize{ a.shape.dim(0), b.shape.dim(1) }),
            3 => Shape.init(&[_]usize{ a.shape.dim(0), a.shape.dim(1), b.shape.dim(1) }),
            else => return error.InvalidInputRank,
        };
        const output = try Tensor.init(self.allocator, out_shape, .bf16, self.device, self.device_id);
        errdefer output.deinit();
        const a_ptr = a.typedPtr(dtype_mod.BF16) orelse return error.InvalidDType;
        const b_ptr = b.typedPtr(dtype_mod.BF16) orelse return error.InvalidDType;
        const o_ptr = output.typedPtr(dtype_mod.BF16) orelse return error.InvalidDType;
        const n_dim = b.shape.dim(1);
        switch (a.shape.ndim) {
            2 => {
                const m_dim = a.shape.dim(0);
                for (0..m_dim) |m| {
                    for (0..n_dim) |n| {
                        var sum: f32 = 0.0;
                        for (0..k_dim) |k_idx| sum += a_ptr[m * k_dim + k_idx].toFloat32() * b_ptr[k_idx * n_dim + n].toFloat32();
                        o_ptr[m * n_dim + n] = dtype_mod.BF16.fromFloat32(sum);
                    }
                }
            },
            3 => {
                const batch_size = a.shape.dim(0);
                const m_dim = a.shape.dim(1);
                for (0..batch_size) |batch_idx| {
                    for (0..m_dim) |m| {
                        for (0..n_dim) |n| {
                            var sum: f32 = 0.0;
                            for (0..k_dim) |k_idx| sum += a_ptr[(batch_idx * m_dim + m) * k_dim + k_idx].toFloat32() * b_ptr[k_idx * n_dim + n].toFloat32();
                            o_ptr[(batch_idx * m_dim + m) * n_dim + n] = dtype_mod.BF16.fromFloat32(sum);
                        }
                    }
                }
            },
            else => return error.InvalidInputRank,
        }
        return output;
    }
};

pub const ChunkedScan = struct {
    seq_len: usize,
    chunk_size: usize,
    num_chunks: usize,

    pub fn init(seq_len: usize, chunk_size: usize) !ChunkedScan {
        if (chunk_size == 0) return error.InvalidChunkSize;
        return .{
            .seq_len = seq_len,
            .chunk_size = chunk_size,
            .num_chunks = if (seq_len == 0) 0 else (seq_len + chunk_size - 1) / chunk_size,
        };
    }

    pub fn getChunkRange(self: ChunkedScan, chunk_idx: usize) !struct { start: usize, end: usize } {
        if (chunk_idx >= self.num_chunks) return error.InvalidChunkIndex;
        const start = chunk_idx * self.chunk_size;
        const end = @min(start + self.chunk_size, self.seq_len);
        return .{ .start = start, .end = end };
    }

    pub fn prefixScan(self: ChunkedScan, chunk_states: []*EflaState) !void {
        if (chunk_states.len != self.num_chunks) return error.InvalidChunkCount;
        if (chunk_states.len <= 1) return;
        for (1..chunk_states.len) |idx| {
            const prev = chunk_states[idx - 1];
            const curr = chunk_states[idx];
            if (prev.batch_size != curr.batch_size or prev.num_heads != curr.num_heads or prev.state_dim != curr.state_dim or prev.value_dim != curr.value_dim or prev.device != curr.device or prev.device_id != curr.device_id) return error.StateShapeMismatch;
            var prev_cpu: ?*Tensor = null;
            var curr_cpu: ?*Tensor = null;
            defer if (prev_cpu) |t| t.deinit();
            defer if (curr_cpu) |t| t.deinit();
            const prev_tensor = if (prev.device == .cpu) prev.state else blk: {
                prev_cpu = try prev.state.to(prev.allocator, .cpu, 0);
                break :blk prev_cpu.?;
            };
            const curr_tensor = if (curr.device == .cpu) curr.state else blk: {
                curr_cpu = try curr.state.to(curr.allocator, .cpu, 0);
                break :blk curr_cpu.?;
            };
            const prev_ptr = prev_tensor.typedPtr(dtype_mod.BF16) orelse return error.InvalidDType;
            const curr_ptr = curr_tensor.typedPtr(dtype_mod.BF16) orelse return error.InvalidDType;
            for (0..curr_tensor.shape.numel()) |i| curr_ptr[i] = dtype_mod.BF16.fromFloat32(curr_ptr[i].toFloat32() + prev_ptr[i].toFloat32());
            if (curr.device != .cpu) {
                if (@hasDecl(Tensor, "copyFrom")) {
                    try curr.state.copyFrom(curr_tensor);
                } else {
                    return error.UnsupportedOperation;
                }
            }
        }
    }
};

test "EFLA state init" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    var state = try EflaState.init(gpa.allocator(), 8, 64, 64, .cpu, 0);
    defer state.deinit();
    try std.testing.expectEqual(@as(usize, 1), state.batch_size);
    try std.testing.expectEqual(@as(usize, 8), state.num_heads);
    try std.testing.expectEqual(@as(usize, 64), state.state_dim);
    try std.testing.expectEqual(@as(usize, 64), state.value_dim);
}

test "ChunkedScan" {
    const scan = try ChunkedScan.init(10000, 1024);
    try std.testing.expectEqual(@as(usize, 1024), scan.chunk_size);
    try std.testing.expectEqual(@as(usize, 10), scan.num_chunks);
    const range = try scan.getChunkRange(5);
    try std.testing.expectEqual(@as(usize, 5120), range.start);
    try std.testing.expectEqual(@as(usize, 6144), range.end);
}
