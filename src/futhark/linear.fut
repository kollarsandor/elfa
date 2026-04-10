def linear_forward [in_features][out_features][batch_size]
    (input: [batch_size][in_features]f32)
    (weight: [out_features][in_features]f32)
    (bias: [out_features]f32)
    : [batch_size][out_features]f32 =
    map (\x ->
        map (\i ->
            bias[i] + reduce (+) 0.0 (map2 (*) x weight[i])
        ) (iota out_features)
    ) input

def linear_backward_input [in_features][out_features][batch_size]
    (grad_output: [batch_size][out_features]f32)
    (weight: [out_features][in_features]f32)
    : [batch_size][in_features]f32 =
    map (\grad_out ->
        map (\j ->
            reduce (+) 0.0 (map2 (\i go -> go * weight[i][j]) (iota out_features) grad_out)
        ) (iota in_features)
    ) grad_output

def linear_backward_weight [in_features][out_features][batch_size]
    (input: [batch_size][in_features]f32)
    (grad_output: [batch_size][out_features]f32)
    : [out_features][in_features]f32 =
    map (\i ->
        map (\j ->
            reduce (+) 0.0 (map2 (\x go -> x[j] * go) input grad_output)
        ) (iota in_features)
    ) (iota out_features)

def linear_backward_bias [out_features][batch_size]
    (grad_output: [batch_size][out_features]f32)
    : [out_features]f32 =
    map (\i ->
        reduce (+) 0.0 (map (\go -> go[i]) grad_output)
    ) (iota out_features)

def fused_linear_gelu [in_features][out_features][batch_size]
    (input: [batch_size][in_features]f32)
    (weight: [out_features][in_features]f32)
    (bias: [out_features]f32)
    : [batch_size][out_features]f32 =
    let linear_out = linear_forward input weight bias
    in map (map (\x ->
        let cdf = 0.5 * (1.0 + f32.tanh (0.7978845608 * (x + 0.044715 * x * x * x)))
        in x * cdf
    )) linear_out

def fused_linear_relu [in_features][out_features][batch_size]
    (input: [batch_size][in_features]f32)
    (weight: [out_features][in_features]f32)
    (bias: [out_features]f32)
    : [batch_size][out_features]f32 =
    let linear_out = linear_forward input weight bias
    in map (map (\x -> if x > 0.0 then x else 0.0)) linear_out

def fused_linear_silu [in_features][out_features][batch_size]
    (input: [batch_size][in_features]f32)
    (weight: [out_features][in_features]f32)
    (bias: [out_features]f32)
    : [batch_size][out_features]f32 =
    let linear_out = linear_forward input weight bias
    in map (map (\x -> x / (1.0 + f32.exp (-x)))) linear_out

def grouped_linear [groups][in_per_group][out_per_group][batch_size]
    (input: [batch_size][groups][in_per_group]f32)
    (weight: [groups][out_per_group][in_per_group]f32)
    (bias: [groups][out_per_group]f32)
    : [batch_size][groups][out_per_group]f32 =
    map (\x ->
        map (\g ->
            map (\o ->
                bias[g][o] + reduce (+) 0.0 (map2 (*) x[g] weight[g][o])
            ) (iota out_per_group)
        ) (iota groups)
    ) input

def depthwise_separable_conv1d [batch][channels][seq_len][kernel_size]
    (input: [batch][channels][seq_len]f32)
    (depth_weight: [channels][kernel_size]f32)
    (point_weight: [channels][channels]f32)
    (depth_bias: [channels]f32)
    (point_bias: [channels]f32)
    : [batch][channels][seq_len]f32 =
    let padding = kernel_size / 2
    
    let depthwise = map (\b ->
        map (\c ->
            map (\t ->
                let window = map (\k ->
                    let idx = t + k - padding
                    in if idx >= 0 && idx < seq_len then
                        input[b][c][idx]
                    else
                        0.0
                ) (iota kernel_size)
                in depth_bias[c] + reduce (+) 0.0 (map2 (*) window depth_weight[c])
            ) (iota seq_len)
        ) (iota channels)
    ) (iota batch)
    
    in map (\b ->
        map (\c ->
            map (\t ->
                point_bias[c] + reduce (+) 0.0 (map (\c_in ->
                    depthwise[b][c_in][t] * point_weight[c][c_in]
                ) (iota channels))
            ) (iota seq_len)
        ) (iota channels)
    ) (iota batch)

def low_rank_linear [in_features][out_features][rank][batch_size]
    (input: [batch_size][in_features]f32)
    (u: [out_features][rank]f32)
    (v: [rank][in_features]f32)
    (bias: [out_features]f32)
    : [batch_size][out_features]f32 =
    let intermediate = map (\x ->
        map (\r ->
            reduce (+) 0.0 (map2 (*) x v[r])
        ) (iota rank)
    ) input
    
    in map (\h ->
        map (\o ->
            bias[o] + reduce (+) 0.0 (map2 (*) h u[o])
        ) (iota out_features)
    ) intermediate

def sparse_linear [in_features][out_features][batch_size][sparsity_pattern]
    (input: [batch_size][in_features]f32)
    (weight_values: []f32)
    (weight_indices: [][2]i64)
    (bias: [out_features]f32)
    : [batch_size][out_features]f32 =
    let dense_weight = scatter (replicate out_features (replicate in_features 0.0))
        (map (\idx -> idx[0]) weight_indices)
        (map (\idx -> idx[1]) weight_indices)
        weight_values
    
    in linear_forward input dense_weight bias

def quantized_linear_forward [in_features][out_features][batch_size]
    (input: [batch_size][in_features]f32)
    (weight_int8: [out_features][in_features]i8)
    (weight_scale: f32)
    (weight_zero_point: i8)
    (bias: [out_features]f32)
    : [batch_size][out_features]f32 =
    let weight_f32 = map (map (\w ->
        (f32.i8 w - f32.i8 weight_zero_point) * weight_scale
    )) weight_int8
    
    in linear_forward input weight_f32 bias

def block_sparse_linear [in_features][out_features][batch_size][block_size]
    (input: [batch_size][in_features]f32)
    (weight_blocks: [][block_size][block_size]f32)
    (block_coords: [][2]i64)
    (bias: [out_features]f32)
    : [batch_size][out_features]f32 =
    let num_blocks = length weight_blocks
    let blocks_per_row = out_features / block_size
    let blocks_per_col = in_features / block_size
    
    let output_blocks = replicate blocks_per_row (replicate block_size 0.0)
    
    let result = loop output_blocks for b < num_blocks do
        let (block_row, block_col) = block_coords[b]
        let block = weight_blocks[b]
        
        let input_block = map (\batch_idx ->
            input[batch_idx][block_col * block_size : (block_col + 1) * block_size]
        ) (iota batch_size)
        
        let output_contrib = map (\x ->
            map (\i ->
                reduce (+) 0.0 (map2 (*) x block[i])
            ) (iota block_size)
        ) input_block
        
        in map2 (map2 (+)) output_blocks output_contrib
    
    in map (\row -> row + bias) (flatten result)
