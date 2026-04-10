def stable_softmax [n] (xs: [n]f32) : [n]f32 =
    let max_x = reduce f32.max f32.lowest xs
    let exps = map (\x -> f32.exp (x - max_x)) xs
    let sum_exps = reduce (+) 0.0 exps
    in map (\e -> e / sum_exps) exps

def log_softmax [n] (xs: [n]f32) : [n]f32 =
    let max_x = reduce f32.max f32.lowest xs
    let shifted = map (\x -> x - max_x) xs
    let sum_exp = reduce (+) 0.0 (map f32.exp shifted)
    let log_sum_exp = f32.log sum_exp
    in map (\x -> x - max_x - log_sum_exp) xs

def softmax_temperature [n] (xs: [n]f32) (temperature: f32) : [n]f32 =
    let scaled = map (\x -> x / temperature) xs
    in stable_softmax scaled

def top_k_softmax [n] (xs: [n]f32) (k: i64) : [n]f32 =
    let indexed = zip xs (iota n)
    let sorted = sort_by (\(x, _) -> -x) indexed
    let top_k = take k sorted
    
    let mask = replicate n false
    let mask' = scatter mask (map (\(_, idx) -> idx) top_k) (replicate k true)
    
    let masked = map2 (\x m -> if m then x else f32.lowest) xs mask'
    in stable_softmax masked

def top_p_softmax [n] (xs: [n]f32) (p: f32) : [n]f32 =
    let sorted = sort (>) xs
    let cumsum = scan (+) 0.0 sorted
    
    let threshold_idx = reduce (\acc i ->
        if cumsum[i] <= p then i else acc
    ) 0 (iota n)
    
    let threshold = sorted[threshold_idx]
    
    let masked = map (\x -> if x >= threshold then x else f32.lowest) xs
    in stable_softmax masked

def sparse_softmax [n] (xs: [n]f32) (sparsity_threshold: f32) : [n]f32 =
    let max_x = reduce f32.max f32.lowest xs
    let shifted = map (\x -> x - max_x) xs
    
    let masked = map (\x ->
        if x > sparsity_threshold then x else f32.lowest
    ) shifted
    
    let exps = map (\x -> if x == f32.lowest then 0.0 else f32.exp x) masked
    let sum_exps = reduce (+) 0.0 exps
    
    in map (\e -> if e == 0.0 then 0.0 else e / sum_exps) exps

def softmax_2d [m][n] (xss: [m][n]f32) : [m][n]f32 =
    map stable_softmax xss

def softmax_backward [n] (output: [n]f32) (grad_output: [n]f32) : [n]f32 =
    let dot = reduce (+) 0.0 (map2 (*) output grad_output)
    in map2 (\o go -> o * (go - dot)) output grad_output

def fused_softmax_cross_entropy [n][batch_size]
    (logits: [batch_size][n]f32)
    (targets: [batch_size]i64)
    : (f32, [batch_size][n]f32) =
    let losses = map2 (\logit_row target ->
        let max_logit = reduce f32.max f32.lowest logit_row
        let shifted = map (\l -> l - max_logit) logit_row
        let sum_exp = reduce (+) 0.0 (map f32.exp shifted)
        let log_sum_exp = f32.log sum_exp
        let target_logit = shifted[target]
        in -(target_logit - log_sum_exp)
    ) logits targets
    
    let loss = reduce (+) 0.0 losses / f32.i64 batch_size
    
    let grad_logits = map2 (\logit_row target ->
        let probs = stable_softmax logit_row
        in map (\(i, p) ->
            if i == target then p - 1.0 else p
        ) (zip (iota n) probs)
    ) logits targets
    
    in (loss, grad_logits)

def label_smoothing [n][batch_size]
    (targets: [batch_size]i64)
    (smoothing: f32)
    : [batch_size][n]f32 =
    map (\target ->
        map (\i ->
            if i == target then 1.0 - smoothing + smoothing / f32.i64 n
            else smoothing / f32.i64 n
        ) (iota n)
    ) targets

def softmax_focal_loss [n][batch_size]
    (logits: [batch_size][n]f32)
    (targets: [batch_size]i64)
    (gamma: f32)
    (alpha: f32)
    : f32 =
    let losses = map2 (\logit_row target ->
        let probs = stable_softmax logit_row
        let pt = probs[target]
        let ce_loss = -f32.log pt
        in alpha * (1.0 - pt) ** gamma * ce_loss
    ) logits targets
    
    in reduce (+) 0.0 losses / f32.i64 batch_size

def memory_efficient_softmax [seq_len][batch][heads][head_dim]
    (scores: [batch][heads][seq_len][seq_len]f32)
    : [batch][heads][seq_len][seq_len]f32 =
    map (\batch_scores ->
        map (\head_scores ->
            map (\query_scores ->
                stable_softmax query_scores
            ) head_scores
        ) batch_scores
    ) scores

def online_softmax [n] (xs: [n]f32) : [n]f32 =
    let (max_val, sum_exp) = loop (m, s) = (f32.lowest, 0.0) for x in xs do
        let new_m = f32.max m x
        let new_s = s * f32.exp (m - new_m) + f32.exp (x - new_m)
        in (new_m, new_s)
    
    in map (\x -> f32.exp (x - max_val) / sum_exp) xs

def softmax_with_dropout [n] (xs: [n]f32) (dropout_rate: f32) (seed: i64) : [n]f32 =
    let probs = stable_softmax xs
    let rng = rng_engine seed
    let mask = map (\_ ->
        let (_, rand_val) = rand rng
        in if rand_val > dropout_rate then 1.0 / (1.0 - dropout_rate) else 0.0
    ) (iota n)
    in map2 (*) probs mask
