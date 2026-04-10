const std = @import("std");
const Config = @import("../main.zig").Config;

pub const LongFormSummary = struct {
    allocator: std.mem.Allocator,
    config: *const Config,
    documents: [][]const u8,
    reference_summaries: [][]const u8,
    
    pub const TestCase = struct {
        document: []const u8,
        reference_summary: []const u8,
        document_id: usize,
    };
    
    pub fn init(allocator: std.mem.Allocator, config: *const Config) !LongFormSummary {
        const num_docs = 100;
        const documents = try allocator.alloc([]const u8, num_docs);
        const summaries = try allocator.alloc([]const u8, num_docs);
        
        const topics = [_][]const u8{
            "artificial intelligence",
            "climate change",
            "space exploration",
            "medical research",
            "economic policy",
            "historical events",
            "technological innovation",
            "environmental conservation",
            "social movements",
            "scientific discoveries",
        };
        
        for (0..num_docs) |i| {
            const topic = topics[i % topics.len];
            documents[i] = try generateDocument(allocator, topic, config.max_seq_len / 10);
            summaries[i] = try generateSummary(allocator, topic);
        }
        
        return LongFormSummary{
            .allocator = allocator,
            .config = config,
            .documents = documents,
            .reference_summaries = summaries,
        };
    }
    
    pub fn deinit(self: *LongFormSummary) void {
        for (self.documents) |doc| {
            self.allocator.free(doc);
        }
        self.allocator.free(self.documents);
        
        for (self.reference_summaries) |summary| {
            self.allocator.free(summary);
        }
        self.allocator.free(self.reference_summaries);
    }
    
    pub fn generateTestCase(self: *LongFormSummary, seed: usize) !TestCase {
        const doc_idx = seed % self.documents.len;
        
        return TestCase{
            .document = self.documents[doc_idx],
            .reference_summary = self.reference_summaries[doc_idx],
            .document_id = doc_idx,
        };
    }
    
    pub fn freeTestCase(self: *LongFormSummary, test_case: TestCase) void {
        _ = self;
        _ = test_case;
    }
    
    fn generateDocument(allocator: std.mem.Allocator, topic: []const u8, target_paragraphs: usize) ![]const u8 {
        var doc = std.ArrayList(u8).init(allocator);
        errdefer doc.deinit();
        
        const intro_templates = [_][]const u8{
            "This comprehensive analysis examines the multifaceted nature of ",
            "In recent years, significant developments have occurred in the field of ",
            "The following document provides an in-depth exploration of ",
            "Researchers and practitioners alike have shown increasing interest in ",
        };
        
        const body_templates = [_][]const u8{
            "The historical context reveals important patterns that continue to influence current developments. ",
            "Multiple stakeholders have contributed to the evolution of this domain through various initiatives. ",
            "Empirical evidence suggests a correlation between key factors and observed outcomes. ",
            "Theoretical frameworks provide valuable insights into underlying mechanisms. ",
            "Practical applications demonstrate the real-world impact of these concepts. ",
            "Challenges and limitations remain significant considerations for future progress. ",
            "Comparative analysis with related fields highlights unique characteristics. ",
            "Methodological approaches have evolved to address emerging questions. ",
            "Data collected from diverse sources supports the main hypotheses. ",
            "Expert opinions converge on several key recommendations. ",
        };
        
        const conclusion_templates = [_][]const u8{
            "In conclusion, the evidence supports continued investment in this area. ",
            "Future research should address the identified gaps in current knowledge. ",
            "The implications of these findings extend beyond the immediate context. ",
            "Stakeholders must consider both opportunities and risks moving forward. ",
        };
        
        var rng = std.rand.DefaultPrng.init(target_paragraphs);
        const random = rng.random();
        
        const intro = intro_templates[random.intRangeAtMost(usize, 0, intro_templates.len - 1)];
        try doc.appendSlice(intro);
        try doc.appendSlice(topic);
        try doc.appendSlice(". ");
        
        for (0..target_paragraphs) |_| {
            const template = body_templates[random.intRangeAtMost(usize, 0, body_templates.len - 1)];
            try doc.appendSlice(template);
            
            const additional_sentences = random.intRangeAtMost(usize, 3, 8);
            for (0..additional_sentences) |_| {
                const sentence = body_templates[random.intRangeAtMost(usize, 0, body_templates.len - 1)];
                try doc.appendSlice(sentence);
            }
            
            try doc.appendSlice("\n\n");
        }
        
        const conclusion = conclusion_templates[random.intRangeAtMost(usize, 0, conclusion_templates.len - 1)];
        try doc.appendSlice(conclusion);
        
        return try doc.toOwnedSlice();
    }
    
    fn generateSummary(allocator: std.mem.Allocator, topic: []const u8) ![]const u8 {
        const summary_templates = [_][]const u8{
            "This document discusses the key aspects of ",
            "The main findings related to ",
            "A comprehensive overview of ",
            "Critical analysis of developments in ",
        };
        
        const conclusion_phrases = [_][]const u8{
            " highlights important trends and future directions.",
            " reveals both opportunities and challenges for stakeholders.",
            " provides valuable insights for practitioners and researchers.",
            " emphasizes the need for continued investigation and action.",
        };
        
        var rng = std.rand.DefaultPrng.init(topic.len);
        const random = rng.random();
        
        const template = summary_templates[random.intRangeAtMost(usize, 0, summary_templates.len - 1)];
        const conclusion = conclusion_phrases[random.intRangeAtMost(usize, 0, conclusion_phrases.len - 1)];
        
        return try std.fmt.allocPrint(allocator, "{s}{s}{s}", .{ template, topic, conclusion });
    }
    
    pub fn evaluateCoherence(self: *LongFormSummary, generated_summary: []const u8) f32 {
        _ = self;
        
        if (generated_summary.len == 0) {
            return 0.0;
        }
        
        var sentence_count: usize = 0;
        var it = std.mem.tokenize(u8, generated_summary, ".!?");
        while (it.next()) |_| {
            sentence_count += 1;
        }
        
        if (sentence_count == 0) {
            return 0.0;
        }
        
        const avg_sentence_length = @as(f32, @floatFromInt(generated_summary.len)) / @as(f32, @floatFromInt(sentence_count));
        
        const optimal_length: f32 = 20.0;
        const deviation = @abs(avg_sentence_length - optimal_length) / optimal_length;
        
        return 1.0 - @min(deviation, 1.0);
    }
    
    pub fn evaluateCoverage(self: *LongFormSummary, generated_summary: []const u8, reference_summary: []const u8) f32 {
        _ = self;
        
        if (generated_summary.len == 0 or reference_summary.len == 0) {
            return 0.0;
        }
        
        var ref_words = std.mem.tokenize(u8, reference_summary, " \n\t.,!?;:");
        var ref_word_count: usize = 0;
        while (ref_words.next()) |_| {
            ref_word_count += 1;
        }
        
        var covered: usize = 0;
        var ref_it = std.mem.tokenize(u8, reference_summary, " \n\t.,!?;:");
        while (ref_it.next()) |ref_word| {
            if (std.mem.indexOf(u8, generated_summary, ref_word) != null) {
                covered += 1;
            }
        }
        
        return @as(f32, @floatFromInt(covered)) / @as(f32, @floatFromInt(ref_word_count));
    }
    
    pub fn evaluateConciseness(self: *LongFormSummary, generated_summary: []const u8, reference_summary: []const u8) f32 {
        _ = self;
        
        if (generated_summary.len == 0) {
            return 0.0;
        }
        
        const target_length = reference_summary.len;
        const actual_length = generated_summary.len;
        
        const ratio = @as(f32, @floatFromInt(actual_length)) / @as(f32, @floatFromInt(target_length));
        
        if (ratio <= 1.0) {
            return 1.0;
        } else {
            return 1.0 / ratio;
        }
    }
};
