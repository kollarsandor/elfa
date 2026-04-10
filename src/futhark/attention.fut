def scaled_dot_product_attention [seq_len][head_dim]
    (Q: [seq_len][head_dim]f32)
    (K: [seq_len][head_dim]f32)
    (V: [seq_len][head_dim]f32)
    : [seq_len][head_dim]f32 =
    let scores = map (\q_i ->
        map (\k_j ->
            let dot = reduce (+) 0.0 (map2 (*) q_i k_j)
            in dot / f32.sqrt (f32.i64 head_dim)
        ) K
    ) Q
    
    let attn_weights = map softmax scores
    
    in map (\w_row ->
        map (\v_j ->
            reduce (+) 0.0 (map2 (*) w_row v_j)
        ) (transpose V)
    ) attn_weights

def causal_attention_mask [seq_len] : [seq_len][seq_len]f32 =
    map (\i ->
        map (\j ->
            if j <= i then 1.0 else 0.0
        ) (iota seq_len)
    ) (iota seq_len)

def causal_scaled_dot_product_attention [seq_len][head_dim]
    (Q: [seq_len][head_dim]f32)
    (K: [seq_len][head_dim]f32)
    (V: [seq_len][head_dim]f32)
    : [seq_len][head_dim]f32 =
    let mask = causal_attention_mask seq_len
    
    let scores = map2 (\q_i mask_i ->
        map2 (\k_j m ->
            let dot = reduce (+) 0.0 (map2 (*) q_i k_j)
            in (dot / f32.sqrt (f32.i64 head_dim)) * m
        ) K mask_i
    ) Q mask
    
    let attn_weights = map (\s ->
        let masked = map (\x -> if x == 0.0 then f32.lowest else x) s
        in softmax masked
    ) scores
    
    in map (\w_row ->
        map (\v_j ->
            reduce (+) 0.0 (map2 (*) w_row v_j)
        ) (transpose V)
    ) attn_weights

def multi_head_attention [seq_len][dim][num_heads][head_dim]
    (hidden_states: [seq_len][dim]f32)
    (q_weights: [num_heads][head_dim][dim]f32)
    (k_weights: [num_heads][head_dim][dim]f32)
    (v_weights: [num_heads][head_dim][dim]f32)
    (o_weights: [dim][num_heads][head_dim]f32)
    (q_bias: [num_heads][head_dim]f32)
    (k_bias: [num_heads][head_dim]f32)
    (v_bias: [num_heads][head_dim]f32)
    (o_bias: [dim]f32)
    : [seq_len][dim]f32 =
    let Q = map (\h ->
        map (\t ->
            q_bias[h] + map (\d -> dot_product hidden_states[t] q_weights[h][d]) (iota head_dim)
        ) (iota seq_len)
    ) (iota num_heads)
    
    let K = map (\h ->
        map (\t ->
            k_bias[h] + map (\d -> dot_product hidden_states[t] k_weights[h][d]) (iota head_dim)
        ) (iota seq_len)
    ) (iota num_heads)
    
    let V = map (\h ->
        map (\t ->
            v_bias[h] + map (\d -> dot_product hidden_states[t] v_weights[h][d]) (iota head_dim)
        ) (iota seq_len)
    ) (iota num_heads)
    
    let head_outputs = map (\h ->
        causal_scaled_dot_product_attention Q[h] K[h] V[h]
    ) (iota num_heads)
    
    in map (\t ->
        o_bias + map (\d ->
            reduce (+) 0.0 (map (\h ->
                dot_product head_outputs[h][t] o_weights[d][h]
            ) (iota num_heads))
        ) (iota dim)
    ) (iota seq_len)

def flash_attention_forward [seq_len][head_dim][block_size]
    (Q: [seq_len][head_dim]f32)
    (K: [seq_len][head_dim]f32)
    (V: [seq_len][head_dim]f32)
    : [seq_len][head_dim]f32 =
    let num_blocks = (seq_len + block_size - 1) / block_size
    
    let output = replicate seq_len (replicate head_dim 0.0)
    let lse = replicate seq_len 0.0
    
    let (output', lse') = loop (output, lse) for block_idx < num_blocks do
        let block_start = block_idx * block_size
        let block_end = i64.min (block_start + block_size) seq_len
        let block_len = block_end - block_start
        
        let q_block = Q[block_start:block_end]
        
        let (block_out, block_lse) = loop (acc_out, acc_lse) = (replicate block_len (replicate head_dim 0.0), replicate block_len 0.0)
            for kv_block_idx < num_blocks do
                let kv_start = kv_block_idx * block_size
                let kv_end = i64.min (kv_start + block_size) seq_len
                
                let k_block = K[kv_start:kv_end]
                let v_block = V[kv_start:kv_end]
                
                let scores = map (\q_i ->
                    map (\k_j ->
                        if kv_start + j <= block_start + i then
                            let dot = reduce (+) 0.0 (map2 (*) q_i k_j)
                            in dot / f32.sqrt (f32.i64 head_dim)
                        else
                            f32.lowest
                    ) k_block
                ) q_block
                
                let block_max = map (reduce f32.max f32.lowest) scores
                let exp_scores = map2 (\s_row m ->
                    map (\s -> if s == f32.lowest then 0.0 else f32.exp (s - m)) s_row
                ) scores block_max
                
                let block_sum = map (reduce (+) 0.0) exp_scores
                
                let new_lse = map3 (\old_lse new_max new_sum ->
                    f32.log (f32.exp old_lse + new_sum * f32.exp new_max)
                ) acc_lse block_max block_sum
                
                let block_out_new = map3 (\old_out exp_s_row v_block ->
                    map (\j ->
                        let weighted_v = reduce (+) 0.0 (map2 (*) exp_s_row (map (\v -> v[j]) v_block))
                        in old_out[j] + weighted_v
                    ) (iota head_dim)
                ) acc_out exp_scores v_block
                
                in (block_out_new, new_lse)
        
        let output_new = map (\i ->
            if block_start + i < seq_len then
                block_out[i]
            else
                output[block_start + i]
        ) (iota block_len)
        
        let lse_new = map (\i ->
            if block_start + i < seq_len then
                block_lse[i]
            else
                lse[block_start + i]
        ) (iota block_len)
        
        in (output_new, lse_new)
    
    in output'

def linear_attention_associative_scan [seq_len][head_dim]
    (Q: [seq_len][head_dim]f32)
    (K: [seq_len][head_dim]f32)
    (V: [seq_len][head_dim]f32)
    : [seq_len][head_dim]f32 =
    let kv_products = map2 (map2 (*)) K V
    
    let cumulative_kv = scan (map2 (+)) (replicate head_dim 0.0) kv_products
    
    in map2 (\q_t cum_kv_t ->
        map2 (*) q_t cum_kv_t
    ) Q cumulative_kv

def chunked_linear_attention [seq_len][head_dim][chunk_size]
    (Q: [seq_len][head_dim]f32)
    (K: [seq_len][head_dim]f32)
    (V: [seq_len][head_dim]f32)
    : [seq_len][head_dim]f32 =
    let num_chunks = (seq_len + chunk_size - 1) / chunk_size
    
    let chunk_outputs = map (\chunk_idx ->
        let start = chunk_idx * chunk_size
        let end = i64.min (start + chunk_size) seq_len
        let len = end - start
        
        let q_chunk = Q[start:end]
        let k_chunk = K[start:end]
        let v_chunk = V[start:end]
        
        in linear_attention_associative_scan len head_dim q_chunk k_chunk v_chunk
    ) (iota num_chunks)
    
    in flatten chunk_outputs

def sliding_window_attention [seq_len][head_dim][window_size]
    (Q: [seq_len][head_dim]f32)
    (K: [seq_len][head_dim]f32)
    (V: [seq_len][head_dim]f32)
    : [seq_len][head_dim]f32 =
    map (\t ->
        let window_start = i64.max 0 (t - window_size)
        let window_k = K[window_start:t+1]
        let window_v = V[window_start:t+1]
        
        let q_t = Q[t]
        
        let scores = map (\k_j ->
            dot_product q_t k_j / f32.sqrt (f32.i64 head_dim)
        ) window_k
        
        let attn_weights = softmax scores
        
        in map (\v_j ->
            reduce (+) 0.0 (map2 (*) attn_weights v_j)
        ) (transpose window_v)
    ) (iota seq_len)
