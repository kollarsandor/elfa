const std = @import("std");
const Config = @import("main.zig").Config;
const memory = @import("memory.zig");
const model = @import("model.zig");

pub const Trainer = struct {
    allocator: std.mem.Allocator,
    config: *const Config,
    memory_pool: *memory.MemoryPool,
    current_step: usize,
    current_epoch: usize,
    global_step: usize,
    checkpoint_dir: []const u8,
    log_file: ?std.fs.File,
    loss_history: std.ArrayList(f32),
    learning_rate_schedule: LearningRateSchedule,
    gradient_accumulation_steps: usize,
    max_grad_norm: f32,
    warmup_steps: usize,
    
    const LearningRateSchedule = struct {
        base_lr: f32,
        current_lr: f32,
        min_lr: f32,
        warmup_steps: usize,
        total_steps: usize,
        schedule_type: ScheduleType,
        
        const ScheduleType = enum {
            constant,
            linear,
            cosine,
            polynomial,
        };
        
        fn getLearningRate(self: LearningRateSchedule, step: usize) f32 {
            if (step < self.warmup_steps) {
                return self.base_lr * @as(f32, @floatFromInt(step)) / @as(f32, @floatFromInt(self.warmup_steps));
            }
            
            const decay_step = step - self.warmup_steps;
            const decay_total = self.total_steps - self.warmup_steps;
            
            switch (self.schedule_type) {
                .constant => return self.base_lr,
                .linear => {
                    const decay_factor = 1.0 - @as(f32, @floatFromInt(decay_step)) / @as(f32, @floatFromInt(decay_total));
                    return @max(self.min_lr, self.base_lr * decay_factor);
                },
                .cosine => {
                    const progress = @as(f32, @floatFromInt(decay_step)) / @as(f32, @floatFromInt(decay_total));
                    const decay_factor = 0.5 * (1.0 + @cos(progress * std.math.pi));
                    return @max(self.min_lr, self.min_lr + (self.base_lr - self.min_lr) * decay_factor);
                },
                .polynomial => {
                    const progress = @as(f32, @floatFromInt(decay_step)) / @as(f32, @floatFromInt(decay_total));
                    const decay_factor = std.math.pow(f32, 1.0 - progress, 2.0);
                    return @max(self.min_lr, self.base_lr * decay_factor);
                },
            }
        }
    };
    
    pub fn init(allocator: std.mem.Allocator, config: *const Config, memory_pool: *memory.MemoryPool) !Trainer {
        const loss_history = std.ArrayList(f32).init(allocator);
        
        const lr_schedule = LearningRateSchedule{
            .base_lr = config.learning_rate,
            .current_lr = config.learning_rate,
            .min_lr = 1e-6,
            .warmup_steps = 2000,
            .total_steps = 1000000,
            .schedule_type = .cosine,
        };
        
        return Trainer{
            .allocator = allocator,
            .config = config,
            .memory_pool = memory_pool,
            .current_step = 0,
            .current_epoch = 0,
            .global_step = 0,
            .checkpoint_dir = "/checkpoints",
            .log_file = null,
            .loss_history = loss_history,
            .learning_rate_schedule = lr_schedule,
            .gradient_accumulation_steps = 4,
            .max_grad_norm = 1.0,
            .warmup_steps = 2000,
        };
    }
    
    pub fn deinit(self: *Trainer) void {
        self.loss_history.deinit();
        if (self.log_file) |file| {
            file.close();
        }
    }
    
    pub fn trainStep(self: *Trainer, batch_loss: f32) !void {
        self.current_step += 1;
        self.global_step += 1;
        
        try self.loss_history.append(batch_loss);
        
        const current_lr = self.learning_rate_schedule.getLearningRate(self.global_step);
        self.learning_rate_schedule.current_lr = current_lr;
        
        if (self.current_step % 10 == 0) {
            try self.logMetrics();
        }
    }
    
    pub fn updateStep(self: *Trainer, step: usize, loss: f32) !void {
        self.current_step = step;
        self.global_step = step;
        
        try self.loss_history.append(loss);
        
        const current_lr = self.learning_rate_schedule.getLearningRate(step);
        self.learning_rate_schedule.current_lr = current_lr;
        
        if (step % 100 == 0) {
            try self.logTrainingStep(step, loss, current_lr);
        }
    }
    
    pub fn saveCheckpoint(self: *Trainer, path: []const u8, step: usize) !void {
        const checkpoint_path = try std.fmt.allocPrint(self.allocator, "{s}/checkpoint_{d}.bin", .{ path, step });
        defer self.allocator.free(checkpoint_path);
        
        const file = try std.fs.cwd().createFile(checkpoint_path, .{});
        defer file.close();
        
        const writer = file.writer();
        
        try writer.writeInt(u64, step, .little);
        try writer.writeInt(u64, self.current_epoch, .little);
        try writer.writeInt(u64, self.global_step, .little);
        try writer.writeInt(u64, self.loss_history.items.len, .little);
        
        for (self.loss_history.items) |loss| {
            try writer.writeAll(std.mem.asBytes(&loss));
        }
        
        try writer.writeAll(std.mem.asBytes(&self.learning_rate_schedule.current_lr));
        
        const manifest_path = try std.fmt.allocPrint(self.allocator, "{s}/manifest.json", .{path});
        defer self.allocator.free(manifest_path);
        
        const manifest_file = try std.fs.cwd().createFile(manifest_path, .{});
        defer manifest_file.close();
        
        const manifest_writer = manifest_file.writer();
        try manifest_writer.print("{{\"latest_step\": {d}, \"timestamp\": {d}, \"loss\": {d:.6}}}", .{ step, std.time.timestamp(), self.loss_history.items[self.loss_history.items.len - 1] });
    }
    
    pub fn loadCheckpoint(self: *Trainer, path: []const u8) !void {
        const manifest_path = try std.fmt.allocPrint(self.allocator, "{s}/manifest.json", .{path});
        defer self.allocator.free(manifest_path);
        
        const manifest_file = std.fs.cwd().openFile(manifest_path, .{}) catch {
            self.current_step = 0;
            return;
        };
        defer manifest_file.close();
        
        const manifest_content = try manifest_file.readToEndAlloc(self.allocator, 1024);
        defer self.allocator.free(manifest_content);
        
        var latest_step: usize = 0;
        
        const step_prefix = "\"latest_step\": ";
        if (std.mem.indexOf(u8, manifest_content, step_prefix)) |idx| {
            const start = idx + step_prefix.len;
            const end = std.mem.indexOf(u8, manifest_content[start..], ",") orelse manifest_content.len - start;
            latest_step = try std.fmt.parseInt(usize, manifest_content[start..start + end], 10);
        }
        
        if (latest_step == 0) {
            self.current_step = 0;
            return;
        }
        
        const checkpoint_path = try std.fmt.allocPrint(self.allocator, "{s}/checkpoint_{d}.bin", .{ path, latest_step });
        defer self.allocator.free(checkpoint_path);
        
        const file = try std.fs.cwd().openFile(checkpoint_path, .{});
        defer file.close();
        
        const reader = file.reader();
        
        self.current_step = try reader.readInt(u64, .little);
        self.current_epoch = try reader.readInt(u64, .little);
        self.global_step = try reader.readInt(u64, .little);
        
        const loss_count = try reader.readInt(u64, .little);
        
        self.loss_history.clearRetainingCapacity();
        
        var i: u64 = 0;
        while (i < loss_count) : (i += 1) {
            var loss_bytes: [4]u8 = undefined;
            _ = try reader.readAll(&loss_bytes);
            const loss = std.mem.bytesToValue(f32, &loss_bytes);
            try self.loss_history.append(loss);
        }
        
        var lr_bytes: [4]u8 = undefined;
        _ = try reader.readAll(&lr_bytes);
        self.learning_rate_schedule.current_lr = std.mem.bytesToValue(f32, &lr_bytes);
    }
    
    pub fn getCurrentStep(self: Trainer) usize {
        return self.current_step;
    }
    
    pub fn getCurrentEpoch(self: Trainer) usize {
        return self.current_epoch;
    }
    
    pub fn getGlobalStep(self: Trainer) usize {
        return self.global_step;
    }
    
    pub fn getCurrentLearningRate(self: Trainer) f32 {
        return self.learning_rate_schedule.current_lr;
    }
    
    pub fn setLearningRate(self: *Trainer, lr: f32) void {
        self.learning_rate_schedule.current_lr = lr;
    }
    
    pub fn clipGradients(self: *Trainer, gradients: []f32) !void {
        var global_norm: f32 = 0.0;
        
        for (gradients) |grad| {
            global_norm += grad * grad;
        }
        
        global_norm = @sqrt(global_norm);
        
        if (global_norm > self.max_grad_norm) {
            const scale = self.max_grad_norm / global_norm;
            for (gradients) |*grad| {
                grad.* *= scale;
            }
        }
    }
    
    fn logMetrics(self: *Trainer) !void {
        if (self.log_file == null) {
            self.log_file = try std.fs.cwd().createFile("training.log", .{ .truncate = false });
        }
        
        const writer = self.log_file.?.writer();
        
        if (self.loss_history.items.len > 0) {
            const recent_loss = self.loss_history.items[self.loss_history.items.len - 1];
            try writer.print("Step {d}: Loss = {d:.6}, LR = {e:.6}\n", .{ self.global_step, recent_loss, self.learning_rate_schedule.current_lr });
        }
    }
    
    fn logTrainingStep(self: *Trainer, step: usize, loss: f32, lr: f32) !void {
        if (self.log_file == null) {
            self.log_file = try std.fs.cwd().createFile("training.log", .{ .truncate = false });
        }
        
        const writer = self.log_file.?.writer();
        try writer.print("[{d}] Step {d}: Loss = {d:.6}, LR = {e:.6}\n", .{ std.time.timestamp(), step, loss, lr });
    }
    
    pub fn logEvaluationResults(self: *Trainer, step: usize, results: anytype) !void {
        if (self.log_file == null) {
            self.log_file = try std.fs.cwd().createFile("training.log", .{ .truncate = false });
        }
        
        const writer = self.log_file.?.writer();
        try writer.print("[{d}] Evaluation at Step {d}:\n", .{ std.time.timestamp(), step });
        try writer.print("  Needle Recall: {d:.4}\n", .{results.needle_recall});
        try writer.print("  Needle Precision: {d:.4}\n", .{results.needle_precision});
        try writer.print("  ROUGE-1: {d:.4}\n", .{results.summary_rouge1});
        try writer.print("  ROUGE-2: {d:.4}\n", .{results.summary_rouge2});
        try writer.print("  ROUGE-L: {d:.4}\n", .{results.summary_rougeL});
        try writer.print("  Perplexity: {d:.4}\n", .{results.perplexity});
        try writer.print("  Token Accuracy: {d:.4}\n", .{results.token_accuracy});
    }
    
    pub fn getAverageLoss(self: Trainer, window: usize) f32 {
        if (self.loss_history.items.len == 0) {
            return 0.0;
        }
        
        const start = if (self.loss_history.items.len > window) self.loss_history.items.len - window else 0;
        var sum: f32 = 0.0;
        
        for (self.loss_history.items[start..]) |loss| {
            sum += loss;
        }
        
        return sum / @as(f32, @floatFromInt(self.loss_history.items.len - start));
    }
    
    pub fn shouldStopEarly(self: Trainer, patience: usize) bool {
        if (self.loss_history.items.len < patience + 10) {
            return false;
        }
        
        const recent_avg = self.getAverageLoss(10);
        const older_avg = blk: {
            var sum: f32 = 0.0;
            const start = self.loss_history.items.len - patience - 10;
            const end = self.loss_history.items.len - 10;
            for (self.loss_history.items[start..end]) |loss| {
                sum += loss;
            }
            break :blk sum / @as(f32, @floatFromInt(patience));
        };
        
        return recent_avg >= older_avg;
    }
};
