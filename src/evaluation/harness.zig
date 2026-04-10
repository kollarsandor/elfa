const std = @import("std");
const Config = @import("../main.zig").Config;
const model = @import("../model.zig");
const tokenizer = @import("../tokenizer.zig");
const NeedleHaystack = @import("needle_haystack.zig").NeedleHaystack;
const LongFormSummary = @import("long_form_summary.zig").LongFormSummary;

pub const EvaluationHarness = struct {
    allocator: std.mem.Allocator,
    config: *const Config,
    needle_haystack: NeedleHaystack,
    long_form_summary: LongFormSummary,
    results: EvaluationResults,
    
    pub const EvaluationResults = struct {
        needle_recall: f32 = 0.0,
        needle_precision: f32 = 0.0,
        summary_rouge1: f32 = 0.0,
        summary_rouge2: f32 = 0.0,
        summary_rougeL: f32 = 0.0,
        perplexity: f32 = 0.0,
        token_accuracy: f32 = 0.0,
    };
    
    pub fn init(allocator: std.mem.Allocator, config: *const Config) !EvaluationHarness {
        const needle_haystack = try NeedleHaystack.init(allocator, config);
        const long_form_summary = try LongFormSummary.init(allocator, config);
        
        return EvaluationHarness{
            .allocator = allocator,
            .config = config,
            .needle_haystack = needle_haystack,
            .long_form_summary = long_form_summary,
            .results = EvaluationResults{},
        };
    }
    
    pub fn deinit(self: *EvaluationHarness) void {
        self.needle_haystack.deinit();
        self.long_form_summary.deinit();
    }
    
    pub fn runEvaluation(self: *EvaluationHarness, model_instance: *model.EFLAModel, tok: *tokenizer.EntropyTokenizer) !EvaluationResults {
        const needle_results = try self.runNeedleHaystack(model_instance, tok);
        const summary_results = try self.runLongFormSummary(model_instance, tok);
        const perplexity = try self.calculatePerplexity(model_instance, tok);
        const accuracy = try self.calculateTokenAccuracy(model_instance, tok);
        
        return EvaluationResults{
            .needle_recall = needle_results.recall,
            .needle_precision = needle_results.precision,
            .summary_rouge1 = summary_results.rouge1,
            .summary_rouge2 = summary_results.rouge2,
            .summary_rougeL = summary_results.rougeL,
            .perplexity = perplexity,
            .token_accuracy = accuracy,
        };
    }
    
    fn runNeedleHaystack(self: *EvaluationHarness, model_instance: *model.EFLAModel, tok: *tokenizer.EntropyTokenizer) !struct { recall: f32, precision: f32 } {
        const num_tests = 100;
        var total_recall: f32 = 0.0;
        var total_precision: f32 = 0.0;
        
        for (0..num_tests) |test_idx| {
            const test_case = try self.needle_haystack.generateTestCase(test_idx);
            defer self.needle_haystack.freeTestCase(test_case);
            
            const input_embeddings = try tok.encode(test_case.haystack);
            defer self.allocator.free(input_embeddings);
            
            const hidden_states = try model_instance.forward(@ptrCast(input_embeddings), input_embeddings.len);
            defer self.allocator.free(hidden_states);
            
            const logits = try model_instance.outputProjection(hidden_states);
            defer self.allocator.free(logits);
            
            const generated = try self.generateFromLogits(logits, tok);
            defer self.allocator.free(generated);
            
            const found = std.mem.indexOf(u8, generated, test_case.needle) != null;
            
            if (found) {
                total_recall += 1.0;
                total_precision += 1.0;
            }
        }
        
        return .{
            .recall = total_recall / @as(f32, @floatFromInt(num_tests)),
            .precision = total_precision / @as(f32, @floatFromInt(num_tests)),
        };
    }
    
    fn runLongFormSummary(self: *EvaluationHarness, model_instance: *model.EFLAModel, tok: *tokenizer.EntropyTokenizer) !struct { rouge1: f32, rouge2: f32, rougeL: f32 } {
        const num_tests = 50;
        var total_rouge1: f32 = 0.0;
        var total_rouge2: f32 = 0.0;
        var total_rougeL: f32 = 0.0;
        
        for (0..num_tests) |test_idx| {
            const test_case = try self.long_form_summary.generateTestCase(test_idx);
            defer self.long_form_summary.freeTestCase(test_case);
            
            const input_embeddings = try tok.encode(test_case.document);
            defer self.allocator.free(input_embeddings);
            
            const hidden_states = try model_instance.forward(@ptrCast(input_embeddings), input_embeddings.len);
            defer self.allocator.free(hidden_states);
            
            const logits = try model_instance.outputProjection(hidden_states);
            defer self.allocator.free(logits);
            
            const generated_summary = try self.generateFromLogits(logits, tok);
            defer self.allocator.free(generated_summary);
            
            const rouge1 = try self.calculateRougeN(generated_summary, test_case.reference_summary, 1);
            const rouge2 = try self.calculateRougeN(generated_summary, test_case.reference_summary, 2);
            const rougeL = try self.calculateRougeL(generated_summary, test_case.reference_summary);
            
            total_rouge1 += rouge1;
            total_rouge2 += rouge2;
            total_rougeL += rougeL;
        }
        
        return .{
            .rouge1 = total_rouge1 / @as(f32, @floatFromInt(num_tests)),
            .rouge2 = total_rouge2 / @as(f32, @floatFromInt(num_tests)),
            .rougeL = total_rougeL / @as(f32, @floatFromInt(num_tests)),
        };
    }
    
    fn calculatePerplexity(self: *EvaluationHarness, model_instance: *model.EFLAModel, tok: *tokenizer.EntropyTokenizer) !f32 {
        const eval_texts = try self.loadEvalTexts();
        defer {
            for (eval_texts) |text| {
                self.allocator.free(text);
            }
            self.allocator.free(eval_texts);
        }
        
        var total_log_prob: f32 = 0.0;
        var total_tokens: usize = 0;
        
        for (eval_texts) |text| {
            const tokens = try tok.encode(text);
            defer self.allocator.free(tokens);
            
            const input_embeddings = try self.allocator.alloc(f32, tokens.len * self.config.hidden_dim);
            defer self.allocator.free(input_embeddings);
            
            for (tokens, 0..) |token, i| {
                @memset(input_embeddings[i * self.config.hidden_dim .. (i + 1) * self.config.hidden_dim], @as(f32, @floatFromInt(token)));
            }
            
            const hidden_states = try model_instance.forward(input_embeddings, tokens.len);
            defer self.allocator.free(hidden_states);
            
            const logits = try model_instance.outputProjection(hidden_states);
            defer self.allocator.free(logits);
            
            for (1..tokens.len) |t| {
                const logit_slice = logits[t * self.config.vocab_size .. (t + 1) * self.config.vocab_size];
                const target_token = tokens[t];
                
                var max_logit: f32 = logit_slice[0];
                for (logit_slice) |l| {
                    if (l > max_logit) max_logit = l;
                }
                
                var sum_exp: f32 = 0.0;
                for (logit_slice) |l| {
                    sum_exp += @exp(l - max_logit);
                }
                
                const log_prob = logit_slice[target_token] - max_logit - @log(sum_exp);
                total_log_prob += log_prob;
                total_tokens += 1;
            }
        }
        
        const avg_log_prob = total_log_prob / @as(f32, @floatFromInt(total_tokens));
        return @exp(-avg_log_prob);
    }
    
    fn calculateTokenAccuracy(self: *EvaluationHarness, model_instance: *model.EFLAModel, tok: *tokenizer.EntropyTokenizer) !f32 {
        const eval_texts = try self.loadEvalTexts();
        defer {
            for (eval_texts) |text| {
                self.allocator.free(text);
            }
            self.allocator.free(eval_texts);
        }
        
        var correct: usize = 0;
        var total: usize = 0;
        
        for (eval_texts) |text| {
            const tokens = try tok.encode(text);
            defer self.allocator.free(tokens);
            
            const input_embeddings = try self.allocator.alloc(f32, tokens.len * self.config.hidden_dim);
            defer self.allocator.free(input_embeddings);
            
            for (tokens, 0..) |token, i| {
                @memset(input_embeddings[i * self.config.hidden_dim .. (i + 1) * self.config.hidden_dim], @as(f32, @floatFromInt(token)));
            }
            
            const hidden_states = try model_instance.forward(input_embeddings, tokens.len);
            defer self.allocator.free(hidden_states);
            
            const logits = try model_instance.outputProjection(hidden_states);
            defer self.allocator.free(logits);
            
            for (1..tokens.len) |t| {
                const logit_slice = logits[t * self.config.vocab_size .. (t + 1) * self.config.vocab_size];
                
                var max_idx: usize = 0;
                var max_val: f32 = logit_slice[0];
                for (logit_slice, 0..) |val, i| {
                    if (val > max_val) {
                        max_val = val;
                        max_idx = i;
                    }
                }
                
                if (max_idx == tokens[t]) {
                    correct += 1;
                }
                total += 1;
            }
        }
        
        return @as(f32, @floatFromInt(correct)) / @as(f32, @floatFromInt(total));
    }
    
    fn generateFromLogits(self: *EvaluationHarness, logits: []f32, tok: *tokenizer.EntropyTokenizer) ![]u8 {
        const max_gen_len = 1000;
        var generated_tokens = std.ArrayList(u32).init(self.allocator);
        defer generated_tokens.deinit();
        
        var current_pos = logits.len / self.config.vocab_size - 1;
        
        for (0..max_gen_len) |_| {
            const logit_slice = logits[current_pos * self.config.vocab_size .. (current_pos + 1) * self.config.vocab_size];
            
            var max_idx: usize = 0;
            var max_val: f32 = logit_slice[0];
            for (logit_slice, 0..) |val, i| {
                if (val > max_val) {
                    max_val = val;
                    max_idx = i;
                }
            }
            
            try generated_tokens.append(@intCast(max_idx));
            
            if (max_idx == tok.getEODToken()) {
                break;
            }
            
            current_pos += 1;
            if (current_pos >= logits.len / self.config.vocab_size) {
                break;
            }
        }
        
        return tok.decode(generated_tokens.items);
    }
    
    fn calculateRougeN(self: *EvaluationHarness, generated: []const u8, reference: []const u8, n: usize) !f32 {
        const generated_ngrams = try self.getNgrams(generated, n);
        defer {
            for (generated_ngrams) |ng| {
                self.allocator.free(ng);
            }
            self.allocator.free(generated_ngrams);
        }
        
        const reference_ngrams = try self.getNgrams(reference, n);
        defer {
            for (reference_ngrams) |ng| {
                self.allocator.free(ng);
            }
            self.allocator.free(reference_ngrams);
        }
        
        var overlap: usize = 0;
        for (generated_ngrams) |gen_ng| {
            for (reference_ngrams) |ref_ng| {
                if (std.mem.eql(u8, gen_ng, ref_ng)) {
                    overlap += 1;
                    break;
                }
            }
        }
        
        if (generated_ngrams.len == 0 or reference_ngrams.len == 0) {
            return 0.0;
        }
        
        const precision = @as(f32, @floatFromInt(overlap)) / @as(f32, @floatFromInt(generated_ngrams.len));
        const recall = @as(f32, @floatFromInt(overlap)) / @as(f32, @floatFromInt(reference_ngrams.len));
        
        if (precision + recall == 0.0) {
            return 0.0;
        }
        
        return 2.0 * precision * recall / (precision + recall);
    }
    
    fn calculateRougeL(self: *EvaluationHarness, generated: []const u8, reference: []const u8) !f32 {
        const lcs_len = try self.lcsLength(generated, reference);
        
        if (generated.len == 0 or reference.len == 0) {
            return 0.0;
        }
        
        const precision = @as(f32, @floatFromInt(lcs_len)) / @as(f32, @floatFromInt(generated.len));
        const recall = @as(f32, @floatFromInt(lcs_len)) / @as(f32, @floatFromInt(reference.len));
        
        if (precision + recall == 0.0) {
            return 0.0;
        }
        
        return 2.0 * precision * recall / (precision + recall);
    }
    
    fn getNgrams(self: *EvaluationHarness, text: []const u8, n: usize) ![][]u8 {
        var ngrams = std.ArrayList([]u8).init(self.allocator);
        errdefer {
            for (ngrams.items) |ng| {
                self.allocator.free(ng);
            }
            ngrams.deinit();
        }
        
        var words = std.ArrayList([]const u8).init(self.allocator);
        defer words.deinit();
        
        var it = std.mem.tokenize(u8, text, " \n\t.,!?;:");
        while (it.next()) |word| {
            try words.append(word);
        }
        
        if (words.items.len < n) {
            return ngrams.toOwnedSlice();
        }
        
        for (0..words.items.len - n + 1) |i| {
            var ngram_len: usize = 0;
            for (0..n) |j| {
                ngram_len += words.items[i + j].len + 1;
            }
            
            const ngram = try self.allocator.alloc(u8, ngram_len - 1);
            var pos: usize = 0;
            
            for (0..n) |j| {
                @memcpy(ngram[pos..pos + words.items[i + j].len], words.items[i + j]);
                pos += words.items[i + j].len;
                if (j < n - 1) {
                    ngram[pos] = ' ';
                    pos += 1;
                }
            }
            
            try ngrams.append(ngram);
        }
        
        return ngrams.toOwnedSlice();
    }
    
    fn lcsLength(self: *EvaluationHarness, a: []const u8, b: []const u8) !usize {
        const m = a.len;
        const n = b.len;
        
        if (m == 0 or n == 0) {
            return 0;
        }
        
        const prev = try self.allocator.alloc(usize, n + 1);
        defer self.allocator.free(prev);
        const curr = try self.allocator.alloc(usize, n + 1);
        defer self.allocator.free(curr);
        
        @memset(prev, 0);
        @memset(curr, 0);
        
        for (1..m + 1) |i| {
            for (1..n + 1) |j| {
                if (a[i - 1] == b[j - 1]) {
                    curr[j] = prev[j - 1] + 1;
                } else {
                    curr[j] = @max(prev[j], curr[j - 1]);
                }
            }
            @memcpy(prev, curr);
        }
        
        return curr[n];
    }
    
    fn loadEvalTexts(self: *EvaluationHarness) ![][]u8 {
        var texts = std.ArrayList([]u8).init(self.allocator);
        errdefer {
            for (texts.items) |text| {
                self.allocator.free(text);
            }
            texts.deinit();
        }
        
        const eval_dir = try std.fs.cwd().openDir("/data/eval", .{ .iterate = true });
        defer eval_dir.close();
        
        var iter = eval_dir.iterate();
        while (try iter.next()) |entry| {
            if (entry.kind == .file) {
                const file_path = try std.fs.path.join(self.allocator, &[_][]const u8{ "/data/eval", entry.name });
                defer self.allocator.free(file_path);
                
                const file = try std.fs.cwd().openFile(file_path, .{});
                defer file.close();
                
                const content = try file.readToEndAlloc(self.allocator, 104857600);
                try texts.append(content);
            }
        }
        
        return texts.toOwnedSlice();
    }
    
    pub fn saveResults(self: *EvaluationHarness, path: []const u8, step: usize) !void {
        const file = try std.fs.cwd().createFile(path, .{});
        defer file.close();
        
        const writer = file.writer();
        
        try writer.print("Step: {d}\n", .{step});
        try writer.print("Needle Recall: {d:.4}\n", .{self.results.needle_recall});
        try writer.print("Needle Precision: {d:.4}\n", .{self.results.needle_precision});
        try writer.print("ROUGE-1: {d:.4}\n", .{self.results.summary_rouge1});
        try writer.print("ROUGE-2: {d:.4}\n", .{self.results.summary_rouge2});
        try writer.print("ROUGE-L: {d:.4}\n", .{self.results.summary_rougeL});
        try writer.print("Perplexity: {d:.4}\n", .{self.results.perplexity});
        try writer.print("Token Accuracy: {d:.4}\n", .{self.results.token_accuracy});
    }
};
