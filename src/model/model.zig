const std = @import("std");
const tensor_mod = @import("../tensor/tensor.zig");
const dtype_mod = @import("../tensor/dtype.zig");
const config_mod = @import("../util/config.zig");
const nn_mod = @import("../nn/layers.zig");
const efla_mod = @import("efla.zig");
const prism_mod = @import("prism.zig");

pub const Tensor = tensor_mod.Tensor;
pub const BF16 = dtype_mod.BF16;

pub const ModelError = error{
    InvalidConfiguration,
    ShapeMismatch,
    DTypeMismatch,
    DeviceMismatch,
    UnsupportedDevice,
    InvalidInputRank,
    UnsupportedOperation,
    GradientNotAvailable,
    BackwardNotReady,
};

pub const TransformerBlockForwardResult = struct {
    output: *Tensor,
    new_efla_state: ?*efla_mod.EflaState,
    new_prism_state: ?*prism_mod.PrismState,
};

pub const ModelForwardResult = struct {
    logits: *Tensor,
    efla_states: []?*efla_mod.EflaState,
    prism_states: []?*prism_mod.PrismState,

    pub fn deinit(self: *ModelForwardResult, allocator: std.mem.Allocator) void {
        for (self.efla_states) |state| if (state) |s| s.deinit();
        for (self.prism_states) |state| if (state) |s| s.deinit();
        allocator.free(self.efla_states);
        allocator.free(self.prism_states);
        self.logits.deinit();
    }
};

fn maybeDeinitTensor(t: ?*Tensor) void {
    if (t) |v| v.deinit();
}

fn maybeDeinitEflaState(state: ?*efla_mod.EflaState) void {
    if (state) |s| s.deinit();
}

fn maybeDeinitPrismState(state: ?*prism_mod.PrismState) void {
    if (state) |s| s.deinit();
}

fn cloneTensor(allocator: std.mem.Allocator, src: *Tensor) !*Tensor {
    return try src.to(allocator, src.device, src.device_id);
}

fn cloneTensorToCpu(allocator: std.mem.Allocator, src: *Tensor) !*Tensor {
    return try src.to(allocator, .cpu, 0);
}

fn addCpuTensors(allocator: std.mem.Allocator, a: *Tensor, b: *Tensor) !*Tensor {
    if (a.dtype != .bf16 or b.dtype != .bf16) return ModelError.DTypeMismatch;
    if (!a.shape.equalTo(b.shape)) return ModelError.ShapeMismatch;
    const output = try Tensor.init(allocator, a.shape, a.dtype, .cpu, 0);
    errdefer output.deinit();
    const a_ptr = a.typedPtr(BF16) orelse return ModelError.UnsupportedOperation;
    const b_ptr = b.typedPtr(BF16) orelse return ModelError.UnsupportedOperation;
    const o_ptr = output.typedPtr(BF16) orelse return ModelError.UnsupportedOperation;
    for (0..a.shape.numel()) |i| o_ptr[i] = BF16.fromFloat32(a_ptr[i].toFloat32() + b_ptr[i].toFloat32());
    return output;
}

fn addTensorsFast(allocator: std.mem.Allocator, a: *Tensor, b: *Tensor, device: tensor_mod.Device, device_id: i32) !*Tensor {
    if (a.dtype != .bf16 or b.dtype != .bf16) return ModelError.DTypeMismatch;
    if (!a.shape.equalTo(b.shape)) return ModelError.ShapeMismatch;
    if (a.device == .cpu and b.device == .cpu and device == .cpu) return addCpuTensors(allocator, a, b);
    var a_cpu = try cloneTensorToCpu(allocator, a);
    defer a_cpu.deinit();
    var b_cpu = try cloneTensorToCpu(allocator, b);
    defer b_cpu.deinit();
    var out_cpu = try addCpuTensors(allocator, a_cpu, b_cpu);
    defer out_cpu.deinit();
    return try out_cpu.to(allocator, device, device_id);
}

fn freeLinearBackwardResult(result: anytype) void {
    result.grad_weight.deinit();
    if (@hasField(@TypeOf(result), "grad_bias")) {
        if (result.grad_bias) |gb| gb.deinit();
    }
}

fn linearBackwardInput(linear: *nn_mod.Linear, grad_output: *Tensor, input: *Tensor) !*Tensor {
    const result = try linear.backward(grad_output, input);
    errdefer result.grad_input.deinit();
    freeLinearBackwardResult(result);
    return result.grad_input;
}

fn rmsNormBackwardInput(norm: anytype, grad_output: *Tensor, input: *Tensor) !*Tensor {
    const result = try norm.backward(grad_output, input);
    errdefer result.grad_input.deinit();
    result.grad_weight.deinit();
    return result.grad_input;
}

fn embeddingBackwardStore(embedding: *nn_mod.Embedding, grad_output: *Tensor, input: *Tensor) !void {
    const grad_weight = try embedding.backward(grad_output, input);
    grad_weight.deinit();
}

fn makeZeroPrismState(allocator: std.mem.Allocator, block: *prism_mod.PrismLayer, input: *Tensor) !*prism_mod.PrismState {
    return try prism_mod.PrismState.initWithGroups(allocator, input.shape.dim(0), block.num_groups, block.head_dim, block.head_dim, block.device, block.device_id);
}

fn makeZeroEflaState(allocator: std.mem.Allocator, block: *efla_mod.EflaLayer, input: *Tensor) !*efla_mod.EflaState {
    return try efla_mod.EflaState.initWithBatch(allocator, input.shape.dim(0), block.num_heads, block.head_dim, block.head_dim, block.device, block.device_id);
}

pub const TransformerBlock = struct {
    ln1: *nn_mod.RMSNorm,
    efla: *efla_mod.EflaLayer,
    prism: *prism_mod.PrismLayer,
    ln2: *nn_mod.RMSNorm,
    mlp_up: *nn_mod.Linear,
    mlp_down: *nn_mod.Linear,
    activation: nn_mod.GELU,
    config: config_mod.ModelConfig,
    allocator: std.mem.Allocator,
    device: tensor_mod.Device,
    device_id: i32,
    cached_input: ?*Tensor,
    cached_normed1: ?*Tensor,
    cached_efla_output: ?*Tensor,
    cached_residual1: ?*Tensor,
    cached_normed2: ?*Tensor,
    cached_mlp_up: ?*Tensor,
    cached_mlp_activated: ?*Tensor,
    cached_prev_efla_state: ?*efla_mod.EflaState,
    cached_prev_prism_state: ?*prism_mod.PrismState,
    last_grad_efla_state: ?*efla_mod.EflaState,
    last_grad_prism_state: ?*prism_mod.PrismState,

    const Self = @This();

    pub fn init(allocator: std.mem.Allocator, config: config_mod.ModelConfig, device: tensor_mod.Device, device_id: i32, rng: *std.Random) !*Self {
        if (config.hidden_dim == 0 or config.intermediate_dim == 0 or config.num_heads == 0 or config.head_dim == 0) return ModelError.InvalidConfiguration;
        if (config.hidden_dim != config.num_heads * config.head_dim) return ModelError.InvalidConfiguration;
        const self = try allocator.create(Self);
        errdefer allocator.destroy(self);
        const ln1 = try nn_mod.RMSNorm.init(allocator, config.hidden_dim, device, device_id);
        errdefer ln1.deinit();
        const efla = try efla_mod.EflaLayer.init(allocator, config.efla, config.hidden_dim, config.num_heads, config.head_dim, device, device_id, rng);
        errdefer efla.deinit();
        const prism = try prism_mod.PrismLayer.init(allocator, config.prism, config.hidden_dim, config.head_dim, device, device_id, rng);
        errdefer prism.deinit();
        const ln2 = try nn_mod.RMSNorm.init(allocator, config.hidden_dim, device, device_id);
        errdefer ln2.deinit();
        const mlp_up = try nn_mod.Linear.init(allocator, config.hidden_dim, config.intermediate_dim, false, device, device_id, rng);
        errdefer mlp_up.deinit();
        const mlp_down = try nn_mod.Linear.init(allocator, config.intermediate_dim, config.hidden_dim, false, device, device_id, rng);
        errdefer mlp_down.deinit();
        self.* = .{
            .ln1 = ln1,
            .efla = efla,
            .prism = prism,
            .ln2 = ln2,
            .mlp_up = mlp_up,
            .mlp_down = mlp_down,
            .activation = nn_mod.GELU.init(true),
            .config = config,
            .allocator = allocator,
            .device = device,
            .device_id = device_id,
            .cached_input = null,
            .cached_normed1 = null,
            .cached_efla_output = null,
            .cached_residual1 = null,
            .cached_normed2 = null,
            .cached_mlp_up = null,
            .cached_mlp_activated = null,
            .cached_prev_efla_state = null,
            .cached_prev_prism_state = null,
            .last_grad_efla_state = null,
            .last_grad_prism_state = null,
        };
        return self;
    }

    pub fn deinit(self: *Self) void {
        self.clearCache();
        self.ln1.deinit();
        self.efla.deinit();
        self.prism.deinit();
        self.ln2.deinit();
        self.mlp_up.deinit();
        self.mlp_down.deinit();
        self.allocator.destroy(self);
    }

    fn clearCache(self: *Self) void {
        maybeDeinitTensor(self.cached_input);
        self.cached_input = null;
        maybeDeinitTensor(self.cached_normed1);
        self.cached_normed1 = null;
        maybeDeinitTensor(self.cached_efla_output);
        self.cached_efla_output = null;
        maybeDeinitTensor(self.cached_residual1);
        self.cached_residual1 = null;
        maybeDeinitTensor(self.cached_normed2);
        self.cached_normed2 = null;
        maybeDeinitTensor(self.cached_mlp_up);
        self.cached_mlp_up = null;
        maybeDeinitTensor(self.cached_mlp_activated);
        self.cached_mlp_activated = null;
        maybeDeinitEflaState(self.cached_prev_efla_state);
        self.cached_prev_efla_state = null;
        maybeDeinitPrismState(self.cached_prev_prism_state);
        self.cached_prev_prism_state = null;
        maybeDeinitEflaState(self.last_grad_efla_state);
        self.last_grad_efla_state = null;
        maybeDeinitPrismState(self.last_grad_prism_state);
        self.last_grad_prism_state = null;
    }

    pub fn forward(self: *Self, input: *Tensor, efla_state: ?*efla_mod.EflaState, prism_state: ?*prism_mod.PrismState) !TransformerBlockForwardResult {
        try self.validateTensorForBlock(input);
        self.clearCache();
        errdefer self.clearCache();
        self.cached_input = try cloneTensor(self.allocator, input);
        if (efla_state) |s| self.cached_prev_efla_state = try s.clone();
        if (prism_state) |s| self.cached_prev_prism_state = try s.clone();
        const normed = try self.ln1.forward(input);
        errdefer normed.deinit();
        self.cached_normed1 = try cloneTensor(self.allocator, normed);
        const efla_result = try self.efla.forward(normed, efla_state);
        errdefer efla_result.output.deinit();
        const new_efla_state = efla_result.new_state;
        errdefer new_efla_state.deinit();
        self.cached_efla_output = try cloneTensor(self.allocator, efla_result.output);
        const prism_result = try self.prism.forward(normed, efla_result.output, prism_state);
        errdefer prism_result.output.deinit();
        const new_prism_state = prism_result.new_state;
        errdefer new_prism_state.deinit();
        const residual = try addTensorsFast(self.allocator, input, prism_result.output, self.device, self.device_id);
        errdefer residual.deinit();
        self.cached_residual1 = try cloneTensor(self.allocator, residual);
        normed.deinit();
        prism_result.output.deinit();
        efla_result.output.deinit();
        const normed2 = try self.ln2.forward(residual);
        errdefer normed2.deinit();
        self.cached_normed2 = try cloneTensor(self.allocator, normed2);
        const up = try self.mlp_up.forward(normed2);
        errdefer up.deinit();
        self.cached_mlp_up = try cloneTensor(self.allocator, up);
        const activated = try self.activation.forward(self.allocator, up);
        errdefer activated.deinit();
        self.cached_mlp_activated = try cloneTensor(self.allocator, activated);
        const down = try self.mlp_down.forward(activated);
        errdefer down.deinit();
        const output = try addTensorsFast(self.allocator, residual, down, self.device, self.device_id);
        normed2.deinit();
        up.deinit();
        activated.deinit();
        residual.deinit();
        down.deinit();
        return .{ .output = output, .new_efla_state = new_efla_state, .new_prism_state = new_prism_state };
    }

    fn validateTensorForBlock(self: *Self, tensor: *Tensor) !void {
        if (tensor.dtype != .bf16) return ModelError.DTypeMismatch;
        if (tensor.device != self.device or tensor.device_id != self.device_id) return ModelError.DeviceMismatch;
        if (tensor.shape.ndim != 3) return ModelError.InvalidInputRank;
        if (tensor.shape.dim(2) != self.config.hidden_dim) return ModelError.ShapeMismatch;
    }

    fn validateBackwardGradient(self: *Self, grad_output: *Tensor) !void {
        try self.validateTensorForBlock(grad_output);
        if (self.cached_input == null) return ModelError.BackwardNotReady;
        if (!grad_output.shape.equalTo(self.cached_input.?.shape)) return ModelError.ShapeMismatch;
    }

    pub fn backward(self: *Self, grad_output: *Tensor) !*Tensor {
        if (self.cached_input == null or self.cached_normed1 == null or self.cached_efla_output == null or self.cached_residual1 == null or self.cached_normed2 == null or self.cached_mlp_up == null or self.cached_mlp_activated == null) return ModelError.BackwardNotReady;
        try self.validateBackwardGradient(grad_output);
        maybeDeinitEflaState(self.last_grad_efla_state);
        self.last_grad_efla_state = null;
        maybeDeinitPrismState(self.last_grad_prism_state);
        self.last_grad_prism_state = null;
        const grad_residual1 = try cloneTensor(self.allocator, grad_output);
        defer grad_residual1.deinit();
        const grad_down = try linearBackwardInput(self.mlp_down, grad_output, self.cached_mlp_activated.?);
        defer grad_down.deinit();
        const grad_activated = try geluBackward(self.allocator, grad_down, self.cached_mlp_up.?, self.device, self.device_id);
        defer grad_activated.deinit();
        const grad_up = try linearBackwardInput(self.mlp_up, grad_activated, self.cached_normed2.?);
        defer grad_up.deinit();
        const grad_ln2 = try rmsNormBackwardInput(self.ln2, grad_up, self.cached_residual1.?);
        defer grad_ln2.deinit();
        const grad_after_first_residual = try addTensorsFast(self.allocator, grad_residual1, grad_ln2, self.device, self.device_id);
        defer grad_after_first_residual.deinit();
        var prism_state_holder: ?*prism_mod.PrismState = null;
        defer if (prism_state_holder) |s| s.deinit();
        const prism_state = self.cached_prev_prism_state orelse blk: {
            prism_state_holder = try makeZeroPrismState(self.allocator, self.prism, self.cached_normed1.?);
            break :blk prism_state_holder.?;
        };
        const prism_result = try self.prism.backward(grad_after_first_residual, self.cached_normed1.?, self.cached_efla_output.?, prism_state);
        self.last_grad_prism_state = prism_result.grad_state;
        defer prism_result.grad_v.deinit();
        var efla_state_holder: ?*efla_mod.EflaState = null;
        defer if (efla_state_holder) |s| s.deinit();
        const efla_state = self.cached_prev_efla_state orelse blk: {
            efla_state_holder = try makeZeroEflaState(self.allocator, self.efla, self.cached_normed1.?);
            break :blk efla_state_holder.?;
        };
        const efla_result = try self.efla.backward(prism_result.grad_v, self.cached_normed1.?, efla_state);
        self.last_grad_efla_state = efla_result.grad_state;
        const total_normed1_grad = try addTensorsFast(self.allocator, prism_result.grad_input, efla_result.grad_input, self.device, self.device_id);
        prism_result.grad_input.deinit();
        efla_result.grad_input.deinit();
        defer total_normed1_grad.deinit();
        const grad_ln1 = try rmsNormBackwardInput(self.ln1, total_normed1_grad, self.cached_input.?);
        defer grad_ln1.deinit();
        return try addTensorsFast(self.allocator, grad_after_first_residual, grad_ln1, self.device, self.device_id);
    }
};

pub const EflaModel = struct {
    embed_tokens: *nn_mod.Embedding,
    blocks: []*TransformerBlock,
    final_norm: *nn_mod.RMSNorm,
    lm_head: ?*nn_mod.Linear,
    config: config_mod.ModelConfig,
    allocator: std.mem.Allocator,
    device: tensor_mod.Device,
    device_id: i32,
    tied_embeddings: bool,
    cached_input_ids: ?*Tensor,
    cached_final_input: ?*Tensor,
    cached_final_normed: ?*Tensor,

    const Self = @This();

    pub fn init(allocator: std.mem.Allocator, config: config_mod.ModelConfig, device: tensor_mod.Device, device_id: i32, rng: *std.Random) !*Self {
        if (config.hidden_dim == 0 or config.intermediate_dim == 0 or config.num_layers == 0 or config.vocab_size == 0) return ModelError.InvalidConfiguration;
        if (config.num_heads == 0 or config.head_dim == 0) return ModelError.InvalidConfiguration;
        if (config.hidden_dim != config.num_heads * config.head_dim) return ModelError.InvalidConfiguration;
        const self = try allocator.create(Self);
        errdefer allocator.destroy(self);
        const embed_tokens = try nn_mod.Embedding.init(allocator, config.vocab_size, config.hidden_dim, device, device_id, rng);
        errdefer embed_tokens.deinit();
        const blocks = try allocator.alloc(*TransformerBlock, config.num_layers);
        errdefer allocator.free(blocks);
        var initialized_blocks: usize = 0;
        errdefer for (blocks[0..initialized_blocks]) |block| block.deinit();
        for (0..config.num_layers) |i| {
            blocks[i] = try TransformerBlock.init(allocator, config, device, device_id, rng);
            initialized_blocks += 1;
        }
        const final_norm = try nn_mod.RMSNorm.init(allocator, config.hidden_dim, device, device_id);
        errdefer final_norm.deinit();
        var lm_head: ?*nn_mod.Linear = null;
        errdefer if (lm_head) |head| head.deinit();
        if (!config.tie_embeddings) lm_head = try nn_mod.Linear.init(allocator, config.hidden_dim, config.vocab_size, false, device, device_id, rng);
        self.* = .{
            .embed_tokens = embed_tokens,
            .blocks = blocks,
            .final_norm = final_norm,
            .lm_head = lm_head,
            .config = config,
            .allocator = allocator,
            .device = device,
            .device_id = device_id,
            .tied_embeddings = config.tie_embeddings,
            .cached_input_ids = null,
            .cached_final_input = null,
            .cached_final_normed = null,
        };
        return self;
    }

    pub fn deinit(self: *Self) void {
        self.clearCache();
        self.embed_tokens.deinit();
        for (self.blocks) |block| block.deinit();
        self.allocator.free(self.blocks);
        self.final_norm.deinit();
        if (self.lm_head) |head| head.deinit();
        self.allocator.destroy(self);
    }

    fn clearCache(self: *Self) void {
        maybeDeinitTensor(self.cached_input_ids);
        self.cached_input_ids = null;
        maybeDeinitTensor(self.cached_final_input);
        self.cached_final_input = null;
        maybeDeinitTensor(self.cached_final_normed);
        self.cached_final_normed = null;
    }

    pub fn forward(self: *Self, input_ids: *Tensor) !*Tensor {
        var result = try self.forwardWithStates(input_ids, null, null);
        for (result.efla_states) |state| if (state) |s| s.deinit();
        for (result.prism_states) |state| if (state) |s| s.deinit();
        self.allocator.free(result.efla_states);
        self.allocator.free(result.prism_states);
        return result.logits;
    }

    pub fn forwardWithStates(self: *Self, input_ids: *Tensor, efla_states_in: ?[]const ?*efla_mod.EflaState, prism_states_in: ?[]const ?*prism_mod.PrismState) !ModelForwardResult {
        if (input_ids.device != self.device or input_ids.device_id != self.device_id) return ModelError.DeviceMismatch;
        if (input_ids.shape.ndim != 2) return ModelError.InvalidInputRank;
        self.clearCache();
        errdefer self.clearCache();
        self.cached_input_ids = try cloneTensor(self.allocator, input_ids);
        const embeds = try self.embed_tokens.forward(input_ids);
        errdefer embeds.deinit();
        var hidden: *Tensor = embeds;
        const efla_states_out = try self.allocator.alloc(?*efla_mod.EflaState, self.blocks.len);
        errdefer self.allocator.free(efla_states_out);
        const prism_states_out = try self.allocator.alloc(?*prism_mod.PrismState, self.blocks.len);
        errdefer self.allocator.free(prism_states_out);
        errdefer hidden.deinit();
        for (0..self.blocks.len) |i| {
            efla_states_out[i] = null;
            prism_states_out[i] = null;
        }
        errdefer {
            for (efla_states_out) |state| if (state) |s| s.deinit();
            for (prism_states_out) |state| if (state) |s| s.deinit();
        }
        for (self.blocks, 0..) |block, i| {
            const efla_state = if (efla_states_in) |states| (if (i < states.len) states[i] else null) else null;
            const prism_state = if (prism_states_in) |states| (if (i < states.len) states[i] else null) else null;
            const result = try block.forward(hidden, efla_state, prism_state);
            hidden.deinit();
            hidden = result.output;
            efla_states_out[i] = result.new_efla_state;
            prism_states_out[i] = result.new_prism_state;
        }
        self.cached_final_input = try cloneTensor(self.allocator, hidden);
        const normed = try self.final_norm.forward(hidden);
        errdefer normed.deinit();
        self.cached_final_normed = try cloneTensor(self.allocator, normed);
        hidden.deinit();
        const logits = try self.projectToVocab(normed);
        normed.deinit();
        return .{ .logits = logits, .efla_states = efla_states_out, .prism_states = prism_states_out };
    }

    fn validateBackwardGradient(self: *Self, grad_output: *Tensor) !void {
        if (grad_output.device != self.device or grad_output.device_id != self.device_id) return ModelError.DeviceMismatch;
        if (grad_output.dtype != .bf16) return ModelError.DTypeMismatch;
        if (grad_output.shape.ndim != 3) return ModelError.InvalidInputRank;
        if (self.cached_final_normed == null) return ModelError.BackwardNotReady;
        if (grad_output.shape.dim(0) != self.cached_final_normed.?.shape.dim(0) or grad_output.shape.dim(1) != self.cached_final_normed.?.shape.dim(1) or grad_output.shape.dim(2) != self.config.vocab_size) return ModelError.ShapeMismatch;
    }

    pub fn backward(self: *Self, grad_output: *Tensor) !*Tensor {
        if (self.cached_input_ids == null or self.cached_final_input == null or self.cached_final_normed == null) return ModelError.BackwardNotReady;
        try self.validateBackwardGradient(grad_output);
        var grad_hidden: *Tensor = undefined;
        if (!self.tied_embeddings) {
            const head = self.lm_head orelse return ModelError.InvalidConfiguration;
            grad_hidden = try linearBackwardInput(head, grad_output, self.cached_final_normed.?);
        } else {
            grad_hidden = try vocabProjectBackward(self.allocator, grad_output, self.cached_final_normed.?, self.embed_tokens.weight, self.config, self.device, self.device_id);
        }
        const grad_after_norm = try rmsNormBackwardInput(self.final_norm, grad_hidden, self.cached_final_input.?);
        grad_hidden.deinit();
        grad_hidden = grad_after_norm;
        var i: usize = self.blocks.len;
        while (i > 0) {
            i -= 1;
            const grad_block = try self.blocks[i].backward(grad_hidden);
            grad_hidden.deinit();
            grad_hidden = grad_block;
        }
        try embeddingBackwardStore(self.embed_tokens, grad_hidden, self.cached_input_ids.?);
        return grad_hidden;
    }

    pub fn collectParameters(self: *Self, allocator: std.mem.Allocator) ![]*Tensor {
        var params: std.ArrayList(*Tensor) = .empty;
        errdefer params.deinit(allocator);
        try params.append(allocator, self.embed_tokens.weight);
        for (self.blocks) |block| {
            try params.append(allocator, block.ln1.weight);
            try params.append(allocator, block.efla.w_k);
            try params.append(allocator, block.efla.w_v);
            try params.append(allocator, block.efla.w_o);
            if (block.efla.beta_param) |beta_param| try params.append(allocator, beta_param);
            for (block.prism.w_beta) |w| try params.append(allocator, w);
            for (block.prism.w_k) |w| try params.append(allocator, w);
            for (block.prism.w_p) |w| try params.append(allocator, w);
            try params.append(allocator, block.prism.shortconv.weight);
            if (block.prism.shortconv.bias) |bias| try params.append(allocator, bias);
            try params.append(allocator, block.ln2.weight);
            try params.append(allocator, block.mlp_up.weight);
            try params.append(allocator, block.mlp_down.weight);
        }
        try params.append(allocator, self.final_norm.weight);
        if (self.lm_head) |head| try params.append(allocator, head.weight);
        return params.toOwnedSlice(allocator);
    }

    pub fn getParameterNames(self: *Self, allocator: std.mem.Allocator) ![]const []const u8 {
        var names: std.ArrayList([]const u8) = .empty;
        errdefer {
            for (names.items) |name| allocator.free(name);
            names.deinit(allocator);
        }
        try names.append(allocator, try allocator.dupe(u8, "embed_tokens.weight"));
        for (self.blocks, 0..) |block, i| {
            try names.append(allocator, try std.fmt.allocPrint(allocator, "blocks.{d}.ln1.weight", .{i}));
            try names.append(allocator, try std.fmt.allocPrint(allocator, "blocks.{d}.efla.w_k", .{i}));
            try names.append(allocator, try std.fmt.allocPrint(allocator, "blocks.{d}.efla.w_v", .{i}));
            try names.append(allocator, try std.fmt.allocPrint(allocator, "blocks.{d}.efla.w_o", .{i}));
            if (block.efla.beta_param != null) try names.append(allocator, try std.fmt.allocPrint(allocator, "blocks.{d}.efla.beta_param", .{i}));
            for (0..block.prism.w_beta.len) |j| try names.append(allocator, try std.fmt.allocPrint(allocator, "blocks.{d}.prism.w_beta.{d}", .{ i, j }));
            for (0..block.prism.w_k.len) |j| try names.append(allocator, try std.fmt.allocPrint(allocator, "blocks.{d}.prism.w_k.{d}", .{ i, j }));
            for (0..block.prism.w_p.len) |j| try names.append(allocator, try std.fmt.allocPrint(allocator, "blocks.{d}.prism.w_p.{d}", .{ i, j }));
            try names.append(allocator, try std.fmt.allocPrint(allocator, "blocks.{d}.prism.shortconv.weight", .{i}));
            if (block.prism.shortconv.bias != null) try names.append(allocator, try std.fmt.allocPrint(allocator, "blocks.{d}.prism.shortconv.bias", .{i}));
            try names.append(allocator, try std.fmt.allocPrint(allocator, "blocks.{d}.ln2.weight", .{i}));
            try names.append(allocator, try std.fmt.allocPrint(allocator, "blocks.{d}.mlp_up.weight", .{i}));
            try names.append(allocator, try std.fmt.allocPrint(allocator, "blocks.{d}.mlp_down.weight", .{i}));
        }
        try names.append(allocator, try allocator.dupe(u8, "final_norm.weight"));
        if (self.lm_head != null) try names.append(allocator, try allocator.dupe(u8, "lm_head.weight"));
        return names.toOwnedSlice(allocator);
    }

    pub fn collectGradients(self: *Self, allocator: std.mem.Allocator) ![]*Tensor {
        var grads: std.ArrayList(*Tensor) = .empty;
        errdefer grads.deinit(allocator);
        const embed_grad = self.embed_tokens.weight.grad orelse return ModelError.GradientNotAvailable;
        try grads.append(allocator, embed_grad);
        for (self.blocks) |block| {
            try grads.append(allocator, block.ln1.weight.grad orelse return ModelError.GradientNotAvailable);
            try grads.append(allocator, block.efla.w_k.grad orelse return ModelError.GradientNotAvailable);
            try grads.append(allocator, block.efla.w_v.grad orelse return ModelError.GradientNotAvailable);
            try grads.append(allocator, block.efla.w_o.grad orelse return ModelError.GradientNotAvailable);
            if (block.efla.beta_param) |beta_param| try grads.append(allocator, beta_param.grad orelse return ModelError.GradientNotAvailable);
            for (block.prism.w_beta) |w| try grads.append(allocator, w.grad orelse return ModelError.GradientNotAvailable);
            for (block.prism.w_k) |w| try grads.append(allocator, w.grad orelse return ModelError.GradientNotAvailable);
            for (block.prism.w_p) |w| try grads.append(allocator, w.grad orelse return ModelError.GradientNotAvailable);
            try grads.append(allocator, block.prism.shortconv.weight.grad orelse return ModelError.GradientNotAvailable);
            if (block.prism.shortconv.bias) |bias| try grads.append(allocator, bias.grad orelse return ModelError.GradientNotAvailable);
            try grads.append(allocator, block.ln2.weight.grad orelse return ModelError.GradientNotAvailable);
            try grads.append(allocator, block.mlp_up.weight.grad orelse return ModelError.GradientNotAvailable);
            try grads.append(allocator, block.mlp_down.weight.grad orelse return ModelError.GradientNotAvailable);
        }
        try grads.append(allocator, self.final_norm.weight.grad orelse return ModelError.GradientNotAvailable);
        if (self.lm_head) |head| try grads.append(allocator, head.weight.grad orelse return ModelError.GradientNotAvailable);
        return grads.toOwnedSlice(allocator);
    }

    pub fn countParameters(self: *Self) u64 {
        var count: u64 = 0;
        count += tensorNumel(self.embed_tokens.weight);
        for (self.blocks) |block| {
            count += tensorNumel(block.ln1.weight);
            count += tensorNumel(block.efla.w_k);
            count += tensorNumel(block.efla.w_v);
            count += tensorNumel(block.efla.w_o);
            if (block.efla.beta_param) |beta_param| count += tensorNumel(beta_param);
            for (block.prism.w_beta) |w| count += tensorNumel(w);
            for (block.prism.w_k) |w| count += tensorNumel(w);
            for (block.prism.w_p) |w| count += tensorNumel(w);
            count += tensorNumel(block.prism.shortconv.weight);
            if (block.prism.shortconv.bias) |bias| count += tensorNumel(bias);
            count += tensorNumel(block.ln2.weight);
            count += tensorNumel(block.mlp_up.weight);
            count += tensorNumel(block.mlp_down.weight);
        }
        count += tensorNumel(self.final_norm.weight);
        if (self.lm_head) |head| count += tensorNumel(head.weight);
        return count;
    }

    fn projectToVocab(self: *Self, hidden: *Tensor) !*Tensor {
        if (!self.tied_embeddings) {
            const head = self.lm_head orelse return ModelError.InvalidConfiguration;
            return head.forward(hidden);
        }
        if (hidden.shape.ndim != 2 and hidden.shape.ndim != 3) return ModelError.InvalidInputRank;
        if (hidden.shape.dim(hidden.shape.ndim - 1) != self.config.hidden_dim) return ModelError.ShapeMismatch;
        var hidden_holder: ?*Tensor = null;
        defer if (hidden_holder) |t| t.deinit();
        const hidden_cpu = if (hidden.device == .cpu) hidden else blk: {
            hidden_holder = try cloneTensorToCpu(self.allocator, hidden);
            break :blk hidden_holder.?;
        };
        var weight_holder: ?*Tensor = null;
        defer if (weight_holder) |t| t.deinit();
        const weight_cpu = if (self.embed_tokens.weight.device == .cpu) self.embed_tokens.weight else blk: {
            weight_holder = try cloneTensorToCpu(self.allocator, self.embed_tokens.weight);
            break :blk weight_holder.?;
        };
        const output_shape = if (hidden_cpu.shape.ndim == 2) tensor_mod.Shape.init(&[_]usize{ hidden_cpu.shape.dim(0), self.config.vocab_size }) else tensor_mod.Shape.init(&[_]usize{ hidden_cpu.shape.dim(0), hidden_cpu.shape.dim(1), self.config.vocab_size });
        var output_cpu = try Tensor.init(self.allocator, output_shape, .bf16, .cpu, 0);
        errdefer output_cpu.deinit();
        const hidden_ptr = hidden_cpu.typedPtr(BF16) orelse return ModelError.UnsupportedOperation;
        const weight_ptr = weight_cpu.typedPtr(BF16) orelse return ModelError.UnsupportedOperation;
        const output_ptr = output_cpu.typedPtr(BF16) orelse return ModelError.UnsupportedOperation;
        const hidden_dim = self.config.hidden_dim;
        const vocab_size = self.config.vocab_size;
        if (hidden_cpu.shape.ndim == 2) {
            const rows = hidden_cpu.shape.dim(0);
            for (0..rows) |row| matVecRowBF16(output_ptr[row * vocab_size ..][0..vocab_size], hidden_ptr[row * hidden_dim ..][0..hidden_dim], weight_ptr, vocab_size, hidden_dim);
        } else {
            const batch_size = hidden_cpu.shape.dim(0);
            const seq_len = hidden_cpu.shape.dim(1);
            for (0..batch_size) |batch_idx| {
                for (0..seq_len) |token_idx| {
                    const row_offset = (batch_idx * seq_len + token_idx) * hidden_dim;
                    const out_offset = (batch_idx * seq_len + token_idx) * vocab_size;
                    matVecRowBF16(output_ptr[out_offset..][0..vocab_size], hidden_ptr[row_offset..][0..hidden_dim], weight_ptr, vocab_size, hidden_dim);
                }
            }
        }
        if (self.device == .cpu) return output_cpu;
        defer output_cpu.deinit();
        return try output_cpu.to(self.allocator, self.device, self.device_id);
    }
};

fn matVecRowBF16(output: []BF16, input: []const BF16, weight: []const BF16, num_rows: usize, num_cols: usize) void {
    for (0..num_rows) |row| {
        var sum: f32 = 0.0;
        const weight_row = weight[row * num_cols ..][0..num_cols];
        for (0..num_cols) |k| sum += input[k].toFloat32() * weight_row[k].toFloat32();
        output[row] = BF16.fromFloat32(sum);
    }
}

fn geluBackward(allocator: std.mem.Allocator, grad_output: *Tensor, input: *Tensor, device: tensor_mod.Device, device_id: i32) !*Tensor {
    if (grad_output.dtype != .bf16 or input.dtype != .bf16) return ModelError.DTypeMismatch;
    if (!grad_output.shape.equalTo(input.shape)) return ModelError.ShapeMismatch;
    var grad_output_cpu = try cloneTensorToCpu(allocator, grad_output);
    defer grad_output_cpu.deinit();
    var input_cpu = try cloneTensorToCpu(allocator, input);
    defer input_cpu.deinit();
    const output_cpu = try Tensor.init(allocator, input_cpu.shape, .bf16, .cpu, 0);
    errdefer output_cpu.deinit();
    const grad_out_ptr = grad_output_cpu.typedPtr(BF16) orelse return ModelError.DTypeMismatch;
    const input_ptr = input_cpu.typedPtr(BF16) orelse return ModelError.DTypeMismatch;
    const out_ptr = output_cpu.typedPtr(BF16) orelse return ModelError.DTypeMismatch;
    const sqrt_2_over_pi: f32 = 0.7978845608028654;
    const coeff: f32 = 0.044715;
    for (0..input_cpu.shape.numel()) |i| {
        const x = input_ptr[i].toFloat32();
        const x3 = x * x * x;
        const inner = sqrt_2_over_pi * (x + coeff * x3);
        const tanh_inner = std.math.tanh(inner);
        const sech2 = 1.0 - tanh_inner * tanh_inner;
        const gelu_grad = 0.5 * (1.0 + tanh_inner) + 0.5 * x * sech2 * sqrt_2_over_pi * (1.0 + 3.0 * coeff * x * x);
        out_ptr[i] = BF16.fromFloat32(grad_out_ptr[i].toFloat32() * gelu_grad);
    }
    if (device == .cpu) return output_cpu;
    defer output_cpu.deinit();
    return try output_cpu.to(allocator, device, device_id);
}

fn ensureGradTensor(allocator: std.mem.Allocator, param: *Tensor) !*Tensor {
    if (param.grad) |g| return g;
    const grad = try Tensor.zeros(allocator, param.shape, param.dtype, param.device, param.device_id);
    param.grad = grad;
    return grad;
}

fn vocabProjectBackward(allocator: std.mem.Allocator, grad_output: *Tensor, hidden_input: *Tensor, embed_weight: *Tensor, config: config_mod.ModelConfig, device: tensor_mod.Device, device_id: i32) !*Tensor {
    if (grad_output.dtype != .bf16 or hidden_input.dtype != .bf16 or embed_weight.dtype != .bf16) return ModelError.DTypeMismatch;
    if (grad_output.shape.ndim != 2 and grad_output.shape.ndim != 3) return ModelError.InvalidInputRank;
    if (hidden_input.shape.ndim != grad_output.shape.ndim) return ModelError.InvalidInputRank;
    if (hidden_input.shape.dim(hidden_input.shape.ndim - 1) != config.hidden_dim) return ModelError.ShapeMismatch;
    if (grad_output.shape.dim(grad_output.shape.ndim - 1) != config.vocab_size) return ModelError.ShapeMismatch;
    if (hidden_input.shape.ndim == 2) {
        if (hidden_input.shape.dim(0) != grad_output.shape.dim(0)) return ModelError.ShapeMismatch;
    } else {
        if (hidden_input.shape.dim(0) != grad_output.shape.dim(0) or hidden_input.shape.dim(1) != grad_output.shape.dim(1)) return ModelError.ShapeMismatch;
    }
    var grad_holder: ?*Tensor = null;
    defer if (grad_holder) |t| t.deinit();
    const grad_cpu = if (grad_output.device == .cpu) grad_output else blk: {
        grad_holder = try cloneTensorToCpu(allocator, grad_output);
        break :blk grad_holder.?;
    };
    var hidden_holder: ?*Tensor = null;
    defer if (hidden_holder) |t| t.deinit();
    const hidden_cpu = if (hidden_input.device == .cpu) hidden_input else blk: {
        hidden_holder = try cloneTensorToCpu(allocator, hidden_input);
        break :blk hidden_holder.?;
    };
    var weight_holder: ?*Tensor = null;
    defer if (weight_holder) |t| t.deinit();
    const weight_cpu = if (embed_weight.device == .cpu) embed_weight else blk: {
        weight_holder = try cloneTensorToCpu(allocator, embed_weight);
        break :blk weight_holder.?;
    };
    const hidden_dim = config.hidden_dim;
    const vocab_size = config.vocab_size;
    const output_shape = if (grad_cpu.shape.ndim == 2) tensor_mod.Shape.init(&[_]usize{ grad_cpu.shape.dim(0), hidden_dim }) else tensor_mod.Shape.init(&[_]usize{ grad_cpu.shape.dim(0), grad_cpu.shape.dim(1), hidden_dim });
    const output_cpu = try Tensor.init(allocator, output_shape, .bf16, .cpu, 0);
    errdefer output_cpu.deinit();
    const grad_ptr = grad_cpu.typedPtr(BF16) orelse return ModelError.UnsupportedOperation;
    const weight_ptr = weight_cpu.typedPtr(BF16) orelse return ModelError.UnsupportedOperation;
    const hidden_ptr = hidden_cpu.typedPtr(BF16) orelse return ModelError.UnsupportedOperation;
    const out_ptr = output_cpu.typedPtr(BF16) orelse return ModelError.UnsupportedOperation;
    const grad_weight = try ensureGradTensor(allocator, embed_weight);
    var grad_weight_cpu_holder: ?*Tensor = null;
    defer if (grad_weight_cpu_holder) |t| t.deinit();
    const grad_weight_cpu = if (embed_weight.device == .cpu) grad_weight else blk: {
        grad_weight_cpu_holder = try cloneTensorToCpu(allocator, grad_weight);
        break :blk grad_weight_cpu_holder.?;
    };
    const gw_ptr = grad_weight_cpu.typedPtr(BF16) orelse return ModelError.UnsupportedOperation;
    const total_tokens = output_cpu.shape.numel() / hidden_dim;
    for (0..total_tokens) |token| {
        const grad_offset = token * vocab_size;
        const out_offset = token * hidden_dim;
        const hidden_offset = token * hidden_dim;
        for (0..hidden_dim) |h| {
            var sum: f32 = 0.0;
            for (0..vocab_size) |v_idx| sum += grad_ptr[grad_offset + v_idx].toFloat32() * weight_ptr[v_idx * hidden_dim + h].toFloat32();
            out_ptr[out_offset + h] = BF16.fromFloat32(sum);
        }
        for (0..vocab_size) |v_idx| {
            const g_val = grad_ptr[grad_offset + v_idx].toFloat32();
            if (g_val == 0.0) continue;
            for (0..hidden_dim) |h| {
                const idx = v_idx * hidden_dim + h;
                gw_ptr[idx] = BF16.fromFloat32(gw_ptr[idx].toFloat32() + g_val * hidden_ptr[hidden_offset + h].toFloat32());
            }
        }
    }
    if (embed_weight.device != .cpu) {
        var dev_copy = try grad_weight_cpu.to(allocator, embed_weight.device, embed_weight.device_id);
        defer dev_copy.deinit();
        if (@hasDecl(Tensor, "copyFrom")) {
            try grad_weight.copyFrom(dev_copy);
        } else {
            return ModelError.UnsupportedOperation;
        }
    }
    if (device == .cpu) return output_cpu;
    defer output_cpu.deinit();
    return try output_cpu.to(allocator, device, device_id);
}

fn tensorNumel(tensor: *Tensor) u64 {
    return @intCast(tensor.shape.numel());
}

test "EflaModel parameter count matches collected parameters and names" {
    var gpa: std.heap.GeneralPurposeAllocator(.{}) = .init;
    defer {
        const status = gpa.deinit();
        std.testing.expect(status == .ok) catch unreachable;
    }
    var prng = std.Random.DefaultPrng.init(42);
    var rng = prng.random();
    var config = config_mod.ModelConfig.default1T(gpa.allocator());
    config.hidden_dim = 256;
    config.num_layers = 2;
    config.intermediate_dim = 512;
    config.num_heads = 4;
    config.head_dim = config.hidden_dim / config.num_heads;
    config.tie_embeddings = false;
    const model = try EflaModel.init(gpa.allocator(), config, .cpu, 0, &rng);
    defer model.deinit();
    const params = try model.collectParameters(gpa.allocator());
    defer gpa.allocator().free(params);
    const names = try model.getParameterNames(gpa.allocator());
    defer {
        for (names) |name| gpa.allocator().free(name);
        gpa.allocator().free(names);
    }
    try std.testing.expectEqual(params.len, names.len);
    var manual_count: u64 = 0;
    for (params) |param| manual_count += @intCast(param.shape.numel());
    try std.testing.expectEqual(manual_count, model.countParameters());
}
