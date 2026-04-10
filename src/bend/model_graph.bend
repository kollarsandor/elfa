type List <T> = (Cons head~T tail~(List T)) | (Nil)

type Tensor = (Tensor data~(List f32) shape~(List i64) strides~(List i64))

type Layer = 
  | (Linear weight~Tensor bias~Tensor in_features~i64 out_features~i64)
  | (Attention q_weight~Tensor k_weight~Tensor v_weight~Tensor o_weight~Tensor num_heads~i64 head_dim~i64)
  | (LayerNorm gamma~Tensor beta~Tensor eps~f32)
  | (FeedForward up_weight~Tensor down_weight~Tensor gate_weight~Tensor intermediate_size~i64)
  | (Embedding weight~Tensor vocab_size~i64 dim~i64)
  | (EFlALayer q~Tensor k~Tensor v~Tensor o~Tensor ffn_up~Tensor ffn_down~Tensor ffn_gate~Tensor)

def List/append <T> (list: (List T)) (value: T) : (List T) =
  match list {
    Nil: (Cons value (Nil))
    Cons: (Cons list.head (List/append list.tail value))
  }

def List/concat <T> (a: (List T)) (b: (List T)) : (List T) =
  match a {
    Nil: b
    Cons: (Cons a.head (List/concat a.tail b))
  }

def List/map <A> <B> (f: A -> B) (list: (List A)) : (List B) =
  match list {
    Nil: (Nil)
    Cons: (Cons (f list.head) (List/map f list.tail))
  }

def List/reduce <T> (f: T -> T -> T) (init: T) (list: (List T)) : T =
  match list {
    Nil: init
    Cons: (f list.head (List/reduce f init list.tail))
  }

def List/length <T> (list: (List T)) : i64 =
  match list {
    Nil: 0
    Cons: (+ 1 (List/length list.tail))
  }

def List/get <T> (list: (List T)) (idx: i64) : T =
  match list {
    Nil: panic "Index out of bounds"
    Cons: if (== idx 0) { list.head } else { (List/get list.tail (- idx 1)) }
  }

def Tensor/new (data: (List f32)) (shape: (List i64)) : Tensor =
  let strides = (Tensor/compute_strides shape)
  (Tensor data shape strides)

def Tensor/compute_strides (shape: (List i64)) : (List i64) =
  match shape {
    Nil: (Nil)
    Cons: 
      let tail_strides = (Tensor/compute_strides shape.tail)
      let stride = (List/reduce (*) 1 shape.tail)
      (Cons stride tail_strides)
  }

def Tensor/flatten (tensor: Tensor) : (List f32) =
  match tensor {
    Tensor: tensor.data
  }

def Tensor/reshape (tensor: Tensor) (new_shape: (List i64)) : Tensor =
  (Tensor/tensor.data new_shape (Tensor/compute_strides new_shape))

def Tensor/slice (tensor: Tensor) (start: i64) (end: i64) : Tensor =
  match tensor {
    Tensor: 
      let sliced_data = (List/slice tensor.data start end)
      (Tensor sliced_data tensor.shape tensor.strides)
  }

def List/slice <T> (list: (List T)) (start: i64) (end: i64) : (List T) =
  if (<= end start) {
    (Nil)
  } else {
    match list {
      Nil: (Nil)
      Cons: 
        if (== start 0) {
          (Cons list.head (List/slice list.tail 0 (- end 1)))
        } else {
          (List/slice list.tail (- start 1) (- end 1))
        }
    }
  }

def Linear/forward (layer: Layer) (input: Tensor) : Tensor =
  match layer {
    Linear: 
      let flat_input = (Tensor/flatten input)
      let flat_weight = (Tensor/flatten layer.weight)
      let result = (List/map (\x -> (* x flat_weight)) flat_input)
      (Tensor/new result (Cons layer.out_features (Nil)))
  }

def Attention/forward (layer: Layer) (input: Tensor) : Tensor =
  match layer {
    Attention:
      let Q = (Linear/forward (Linear layer.q_weight (Tensor/new (Nil) (Nil)) (* layer.num_heads layer.head_dim) (* layer.num_heads layer.head_dim)) input)
      let K = (Linear/forward (Linear layer.k_weight (Tensor/new (Nil) (Nil)) (* layer.num_heads layer.head_dim) (* layer.num_heads layer.head_dim)) input)
      let V = (Linear/forward (Linear layer.v_weight (Tensor/new (Nil) (Nil)) (* layer.num_heads layer.head_dim) (* layer.num_heads layer.head_dim)) input)
      let scores = (Attention/compute_scores Q K layer.head_dim)
      let weights = (Attention/softmax scores)
      let output = (Attention/apply weights V)
      (Linear/forward (Linear layer.o_weight (Tensor/new (Nil) (Nil)) (* layer.num_heads layer.head_dim) (* layer.num_heads layer.head_dim)) output)
  }

def Attention/compute_scores (Q: Tensor) (K: Tensor) (head_dim: i64) : Tensor =
  let q_data = (Tensor/flatten Q)
  let k_data = (Tensor/flatten K)
  let scores = (List/map (\q -> (List/map (\k -> (/ (* q k) (sqrt (f32 head_dim)))) k_data)) q_data)
  (Tensor/new scores (Cons (List/length q_data) (Cons (List/length k_data) (Nil))))

def Attention/softmax (scores: Tensor) : Tensor =
  let flat_scores = (Tensor/flatten scores)
  let max_score = (List/reduce f32.max f32.lowest flat_scores)
  let exp_scores = (List/map (\s -> (f32.exp (- s max_score))) flat_scores)
  let sum_exp = (List/reduce (+) 0.0 exp_scores)
  let normalized = (List/map (\e -> (/ e sum_exp)) exp_scores)
  (Tensor/new normalized scores.shape)

def Attention/apply (weights: Tensor) (V: Tensor) : Tensor =
  let w_data = (Tensor/flatten weights)
  let v_data = (Tensor/flatten V)
  let output = (List/map (\w -> (List/reduce (+) 0.0 (List/map2 (*) w v_data))) w_data)
  (Tensor/new output V.shape)

def LayerNorm/forward (layer: Layer) (input: Tensor) : Tensor =
  match layer {
    LayerNorm:
      let flat_input = (Tensor/flatten input)
      let mean = (/ (List/reduce (+) 0.0 flat_input) (f32 (List/length flat_input)))
      let variance = (/ (List/reduce (+) 0.0 (List/map (\x -> (pow (- x mean) 2)) flat_input)) (f32 (List/length flat_input)))
      let std_dev = (sqrt (+ variance layer.eps))
      let normalized = (List/map (\x -> (/ (- x mean) std_dev)) flat_input)
      let scaled = (List/map2 (*) normalized (Tensor/flatten layer.gamma))
      let shifted = (List/map2 (+) scaled (Tensor/flatten layer.beta))
      (Tensor/new shifted input.shape)
  }

def FeedForward/forward (layer: Layer) (input: Tensor) : Tensor =
  match layer {
    FeedForward:
      let up_proj = (Linear/forward (Linear layer.up_weight (Tensor/new (Nil) (Nil)) layer.intermediate_size layer.intermediate_size) input)
      let gate_proj = (Linear/forward (Linear layer.gate_weight (Tensor/new (Nil) (Nil)) layer.intermediate_size layer.intermediate_size) input)
      let activated = (List/map2 (\u g -> (if (> g 0) { (* u g) } else { 0 })) (Tensor/flatten up_proj) (Tensor/flatten gate_proj))
      let activated_tensor = (Tensor/new activated up_proj.shape)
      (Linear/forward (Linear layer.down_weight (Tensor/new (Nil) (Nil)) layer.intermediate_size layer.intermediate_size) activated_tensor)
  }

def EFlALayer/forward (layer: Layer) (input: Tensor) (seq_len: i64) (chunk_size: i64) : Tensor =
  match layer {
    EFlALayer:
      let normed = (LayerNorm/forward (LayerNorm (Tensor/new (Nil) (Nil)) (Tensor/new (Nil) (Nil)) 1e-5) input)
      let Q = (Linear/forward (Linear layer.q (Tensor/new (Nil) (Nil)) 120000 120000) normed)
      let K = (Linear/forward (Linear layer.k (Tensor/new (Nil) (Nil)) 120000 120000) normed)
      let V = (Linear/forward (Linear layer.v (Tensor/new (Nil) (Nil)) 120000 120000) normed)
      let attn_out = (EFlALayer/chunkwise_attention Q K V seq_len chunk_size)
      let proj_out = (Linear/forward (Linear layer.o (Tensor/new (Nil) (Nil)) 120000 120000) attn_out)
      let residual = (List/map2 (+) (Tensor/flatten input) (Tensor/flatten proj_out))
      let ffn_normed = (LayerNorm/forward (LayerNorm (Tensor/new (Nil) (Nil)) (Tensor/new (Nil) (Nil)) 1e-5) (Tensor/new residual input.shape))
      let ffn_out = (FeedForward/forward (FeedForward layer.ffn_up layer.ffn_down layer.ffn_gate 480000) ffn_normed)
      let final_residual = (List/map2 (+) residual (Tensor/flatten ffn_out))
      (Tensor/new final_residual input.shape)
  }

def EFlALayer/chunkwise_attention (Q: Tensor) (K: Tensor) (V: Tensor) (seq_len: i64) (chunk_size: i64) : Tensor =
  let num_chunks = (/ (+ seq_len (- chunk_size 1)) chunk_size)
  let q_chunks = (EFlALayer/split_into_chunks Q chunk_size num_chunks)
  let k_chunks = (EFlALayer/split_into_chunks K chunk_size num_chunks)
  let v_chunks = (EFlALayer/split_into_chunks V chunk_size num_chunks)
  let outputs = (EFlALayer/process_chunks q_chunks k_chunks v_chunks (Tensor/new (Nil) (Nil)))
  (EFlALayer/concatenate_chunks outputs)

def EFlALayer/split_into_chunks (tensor: Tensor) (chunk_size: i64) (num_chunks: i64) : (List Tensor) =
  (List/map (\i -> (Tensor/slice tensor (* i chunk_size) (* (+ i 1) chunk_size))) (List/range 0 num_chunks))

def List/range (start: i64) (end: i64) : (List i64) =
  if (>= start end) {
    (Nil)
  } else {
    (Cons start (List/range (+ start 1) end))
  }

def EFlALayer/process_chunks (q_chunks: (List Tensor)) (k_chunks: (List Tensor)) (v_chunks: (List Tensor)) (prev_state: Tensor) : (List Tensor) =
  match q_chunks {
    Nil: (Nil)
    Cons:
      let q_chunk = q_chunks.head
      let k_chunk = (List/get k_chunks 0)
      let v_chunk = (List/get v_chunks 0)
      let (output, new_state) = (EFlALayer/chunk_attention q_chunk k_chunk v_chunk prev_state)
      let remaining_outputs = (EFlALayer/process_chunks q_chunks.tail (List/tail k_chunks) (List/tail v_chunks) new_state)
      (Cons output remaining_outputs)
  }

def List/tail <T> (list: (List T)) : (List T) =
  match list {
    Nil: (Nil)
    Cons: list.tail
  }

def EFlALayer/chunk_attention (Q: Tensor) (K: Tensor) (V: Tensor) (prev_state: Tensor) : (Tensor, Tensor) =
  let q_data = (Tensor/flatten Q)
  let k_data = (Tensor/flatten K)
  let v_data = (Tensor/flatten V)
  let state_data = if (== (List/length (Tensor/flatten prev_state)) 0) { (replicate (List/length q_data) 0) } else { (Tensor/flatten prev_state) }
  let new_state = (List/map3 (\s k v -> (+ s (* k v))) state_data k_data v_data)
  let output = (List/map2 (\q s -> (* q s)) q_data new_state)
  ((Tensor/new output Q.shape), (Tensor/new new_state Q.shape))

def EFlALayer/concatenate_chunks (chunks: (List Tensor)) : Tensor =
  match chunks {
    Nil: (Tensor/new (Nil) (Nil))
    Cons: 
      let flat_chunks = (List/map Tensor/flatten chunks)
      let concatenated = (List/reduce (List/concat) (Nil) flat_chunks)
      (Tensor/new concatenated (Cons (List/length concatenated) (Nil)))
  }

def Model/forward (layers: (List Layer)) (input: Tensor) (seq_len: i64) (chunk_size: i64) : Tensor =
  match layers {
    Nil: input
    Cons:
      let layer = layers.head
      let output = match layer {
        EFlALayer: (EFlALayer/forward layer input seq_len chunk_size)
        Linear: (Linear/forward layer input)
        Attention: (Attention/forward layer input)
        LayerNorm: (LayerNorm/forward layer input)
        FeedForward: (FeedForward/forward layer input)
        Embedding: input
      }
      (Model/forward layers.tail output seq_len chunk_size)
  }

def Model/build_efla (num_layers: i64) (hidden_dim: i64) (num_heads: i64) (head_dim: i64) : (List Layer) =
  (List/map (\_ -> 
    (EFlALayer 
      (Tensor/new (Nil) (Nil))
      (Tensor/new (Nil) (Nil))
      (Tensor/new (Nil) (Nil))
      (Tensor/new (Nil) (Nil))
      (Tensor/new (Nil) (Nil))
      (Tensor/new (Nil) (Nil))
      (Tensor/new (Nil) (Nil))
    )
  ) (List/range 0 num_layers))

def main =
  let model = (Model/build_efla 120 120000 96 1250)
  let input = (Tensor/new (Cons 1.0 (Cons 2.0 (Cons 3.0 (Nil)))) (Cons 3 (Nil)))
  let output = (Model/forward model input 50000000 8192)
  output
