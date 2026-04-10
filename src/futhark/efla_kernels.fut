def dot_product [n] (xs: [n]f32) (ys: [n]f32) : f32 =
    reduce (+) 0.0 (map2 (*) xs ys)

def matrix_multiply [m][n][p] (A: [m][n]f32) (B: [n][p]f32) : [m][p]f32 =
    map (\a_row ->
        map (\b_col ->
            dot_product a_row b_col
        ) (transpose B)
    ) A

def softmax [n] (xs: [n]f32) : [n]f32 =
    let max_x = reduce f32.max f32.lowest xs
    let exps = map (\x -> f32.exp (x - max_x)) xs
    let sum_exps = reduce (+) 0.0 exps
    in map (\e -> e / sum_exps) exps

def layer_norm [n] (xs: [n]f32) (gamma: f32) (beta: f32) (eps: f32) : [n]f32 =
    let mean = reduce (+) 0.0 xs / f32.i64 n
    let var = reduce (+) 0.0 (map (\x -> (x - mean) ** 2) xs) / f32.i64 n
    let std_dev = f32.sqrt (var + eps)
    in map (\x -> (x - mean) / std_dev * gamma + beta) xs

def rms_norm [n] (xs: [n]f32) (gamma: f32) (eps: f32) : [n]f32 =
    let sum_sq = reduce (+) 0.0 (map (\x -> x * x) xs)
    let rms = f32.sqrt (sum_sq / f32.i64 n + eps)
    in map (\x -> x / rms * gamma) xs

def linear_attention_forward [seq_len][dim][head_dim]
    (Q: [seq_len][dim]f32)
    (K: [seq_len][dim]f32)
    (V: [seq_len][dim]f32)
    (num_heads: i64)
    : [seq_len][dim]f32 =
    let head_dim_per_head = dim / num_heads
    in map (\t ->
        let q_t = Q[t]
        in map (\h ->
            let head_start = h * head_dim_per_head
            let head_end = head_start + head_dim_per_head
            let q_head = q_t[head_start:head_end]
            
            let state = replicate head_dim_per_head 0.0
            let state' = loop state for s < t+1 do
                let k_s = K[s][head_start:head_end]
                let v_s = V[s][head_start:head_end]
                in map3 (\st k v -> st + k * v) state k_s v_s
            
            let q_sum = reduce (+) 0.0 q_head
            in if q_sum > 0.0 then
                reduce (+) 0.0 (map2 (\q s -> q * s / q_sum) q_head state')
            else
                0.0
        ) (iota num_heads)
    ) (iota seq_len)

def chunkwise_linear_attention [chunk_size][dim][head_dim]
    (Q_chunk: [chunk_size][dim]f32)
    (K_chunk: [chunk_size][dim]f32)
    (V_chunk: [chunk_size][dim]f32)
    (prev_state: [dim]f32)
    (num_heads: i64)
    : ([chunk_size][dim]f32, [dim]f32) =
    let output = map (\t ->
        let q_t = Q_chunk[t]
        in map (\h ->
            let head_dim_per_head = dim / num_heads
            let head_start = h * head_dim_per_head
            let head_end = head_start + head_dim_per_head
            let q_head = q_t[head_start:head_end]
            
            let state = prev_state[head_start:head_end]
            let state' = loop state for s < t+1 do
                let k_s = K_chunk[s][head_start:head_end]
                let v_s = V_chunk[s][head_start:head_end]
                in map3 (\st k v -> st + k * v) state k_s v_s
            
            let q_sum = reduce (+) 0.0 q_head
            in if q_sum > 0.0 then
                reduce (+) 0.0 (map2 (\q s -> q * s / q_sum) q_head state')
            else
                0.0
        ) (iota num_heads)
    ) (iota chunk_size)
    
    let final_state = map (\h ->
        let head_dim_per_head = dim / num_heads
        let head_start = h * head_dim_per_head
        let head_end = head_start + head_dim_per_head
        let state = prev_state[head_start:head_end]
        in loop state for s < chunk_size do
            let k_s = K_chunk[s][head_start:head_end]
            let v_s = V_chunk[s][head_start:head_end]
            in map3 (\st k v -> st + k * v) state k_s v_s
    ) (iota num_heads)
    
    in (output, final_state)

def ffn_forward [seq_len][dim][intermediate]
    (hidden_states: [seq_len][dim]f32)
    (up_weights: [intermediate][dim]f32)
    (down_weights: [dim][intermediate]f32)
    (gate_weights: [intermediate][dim]f32)
    (up_bias: [intermediate]f32)
    (down_bias: [dim]f32)
    (gate_bias: [intermediate]f32)
    : [seq_len][dim]f32 =
    let up_proj = map (\h ->
        map (\i ->
            up_bias[i] + dot_product h up_weights[i]
        ) (iota intermediate)
    ) hidden_states
    
    let gate_proj = map (\h ->
        map (\i ->
            let gate_val = gate_bias[i] + dot_product h gate_weights[i]
            in if gate_val > 0.0 then gate_val else 0.0
        ) (iota intermediate)
    ) hidden_states
    
    let gated = map2 (map2 (*)) up_proj gate_proj
    
    in map (\g ->
        map (\d ->
            down_bias[d] + dot_product g down_weights[d]
        ) (iota dim)
    ) gated

def embedding_lookup [vocab_size][dim][seq_len]
    (token_ids: [seq_len]i64)
    (embedding_weights: [vocab_size][dim]f32)
    : [seq_len][dim]f32 =
    map (\token_id ->
        embedding_weights[token_id]
    ) token_ids

def output_projection [seq_len][dim][vocab_size]
    (hidden_states: [seq_len][dim]f32)
    (output_weights: [vocab_size][dim]f32)
    : [seq_len][vocab_size]f32 =
    map (\h ->
        map (\v ->
            dot_product h output_weights[v]
        ) (iota vocab_size)
    ) hidden_states

def cross_entropy_loss [seq_len][vocab_size]
    (logits: [seq_len][vocab_size]f32)
    (targets: [seq_len]i64)
    : f32 =
    let losses = map2 (\logit_row target ->
        let max_logit = reduce f32.max f32.lowest logit_row
        let shifted = map (\l -> l - max_logit) logit_row
        let sum_exp = reduce (+) 0.0 (map f32.exp shifted)
        let log_sum_exp = f32.log sum_exp
        let target_logit = shifted[target]
        in -(target_logit - log_sum_exp)
    ) logits targets
    in reduce (+) 0.0 losses / f32.i64 seq_len

def adamw_update [n]
    (params: [n]f32)
    (grads: [n]f32)
    (m: [n]f32)
    (v: [n]f32)
    (lr: f32)
    (beta1: f32)
    (beta2: f32)
    (eps: f32)
    (weight_decay: f32)
    (step: i64)
    : ([n]f32, [n]f32, [n]f32) =
    let m_new = map2 (\m_i g -> beta1 * m_i + (1.0 - beta1) * g) m grads
    let v_new = map2 (\v_i g -> beta2 * v_i + (1.0 - beta2) * g * g) v grads
    
    let bias_correction1 = 1.0 - beta1 ** f32.i64 step
    let bias_correction2 = 1.0 - beta2 ** f32.i64 step
    
    let m_hat = map (\m_i -> m_i / bias_correction1) m_new
    let v_hat = map (\v_i -> v_i / bias_correction2) v_new
    
    let params_new = map3 (\p m_h v_h ->
        p - lr * (m_h / (f32.sqrt v_h + eps) + weight_decay * p)
    ) params m_hat v_hat
    
    in (params_new, m_new, v_new)

def gradient_clipping [n] (grads: [n]f32) (max_norm: f32) : [n]f32 =
    let global_norm = f32.sqrt (reduce (+) 0.0 (map (\g -> g * g) grads))
    in if global_norm > max_norm then
        map (\g -> g * max_norm / global_norm) grads
    else
        grads

def apply_quantization [n] (tensor: [n]f32) (bits: i64) : [n]f32 =
    if bits >= 16 then
        tensor
    else
        let num_levels = 2 ** bits
        let qmax = f32.i64 (num_levels - 1)
        let min_val = reduce f32.min f32.highest tensor
        let max_val = reduce f32.max f32.lowest tensor
        let scale = (max_val - min_val) / qmax
        let zero_point = -min_val / scale
        in map (\x ->
            let quantized = f32.round (x / scale + zero_point)
            let clamped = f32.max 0.0 (f32.min qmax quantized)
            in (clamped - zero_point) * scale
        ) tensor

def tensor_parallel_allreduce [n] (tensor: [n]f32) (world_size: i64) : [n]f32 =
    map (\x -> x / f32.i64 world_size) tensor

def sequence_parallel_scatter [seq_len][dim] (tensor: [seq_len][dim]f32) (world_size: i64) (rank: i64) : [][dim]f32 =
    let local_seq_len = seq_len / world_size
    let start = rank * local_seq_len
    let end = start + local_seq_len
    in tensor[start:end]

def sequence_parallel_gather [local_seq_len][dim] (local_tensor: [local_seq_len][dim]f32) (world_size: i64) : [][dim]f32 =
    local_tensor

def rotary_positional_encoding [seq_len][dim] (hidden_states: [seq_len][dim]f32) (base: f32) : [seq_len][dim]f32 =
    map (\(pos, row) ->
        map (\(i, x) ->
            if i % 2 == 0 then
                let angle = f32.i64 pos / (base ** (f32.i64 i / f32.i64 dim))
                in x * f32.cos angle
            else
                let angle = f32.i64 pos / (base ** (f32.i64 (i - 1) / f32.i64 dim))
                in x * f32.sin angle
        ) (zip (iota dim) row)
    ) (zip (iota seq_len) hidden_states)

def gqa_attention [seq_len][dim][kv_heads][head_dim]
    (Q: [seq_len][dim]f32)
    (K: [seq_len][kv_heads][head_dim]f32)
    (V: [seq_len][kv_heads][head_dim]f32)
    (num_heads: i64)
    : [seq_len][dim]f32 =
    let heads_per_kv = num_heads / kv_heads
    in map (\t ->
        map (\h ->
            let kv_head = h / heads_per_kv
            let q_head_start = h * head_dim
            let q_head_end = q_head_start + head_dim
            let q_head = Q[t][q_head_start:q_head_end]
            
            let scores = map (\s ->
                let k_head = K[s][kv_head]
                in dot_product q_head k_head / f32.sqrt (f32.i64 head_dim)
            ) (iota seq_len)
            
            let attn_weights = softmax scores
            
            in reduce (+) 0.0 (map2 (\w s ->
                let v_head = V[s][kv_head]
                in w * reduce (+) 0.0 v_head
            ) attn_weights (iota seq_len))
        ) (iota num_heads)
    ) (iota seq_len)
