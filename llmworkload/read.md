# GGML Learning Notes

## GGML Context and Memory Management

`ggml_context` is the main memory management structure in GGML, acting as a container that keeps everything in a continuous memory space. When initialized, it performs two key memory allocations on the heap:

1. **Context Structure (~40 bytes)**: Contains metadata like memory size, buffer pointers, and object tracking (struct member pointers).
2. **Memory Buffer (user-defined size)**: Main storage for tensors and computations.

**Q:** Advantage of keeping everything in continuous space? Memory access! Aren't default allocations also happening continuously?

```c
struct ggml_context {
size_t mem_size;
void * mem_buffer;
bool mem_buffer_owned;
bool no_alloc;
int n_objects;
struct ggml_object * objects_begin;
struct ggml_object * objects_end;
};
// Example: Allocating context with 16MB buffer
struct ggml_init_params params = {
.mem_size = 1610241024, // 16 MB
.mem_buffer = NULL, // Let GGML handle allocation; if pre-allocated memory by user, then provide buffer_pointer
.no_alloc = false,
};
struct ggml_context ctx = ggml_init(params);
```

Memory allocation happens in two steps:
1. `GGML_MALLOC(sizeof(struct ggml_context))` → ~40 bytes for context.
2. `ggml_aligned_malloc(mem_size)` → 16MB for tensor storage (when `mem_buffer = NULL`).

The `mem_buffer` can either be allocated by GGML (NULL case) or pre-allocated by the user (when a mem_buffer of allocated memory is given), providing flexibility in memory management while ensuring proper alignment for SIMD operations.

### `ggml_tensor`

This data structure is used for tensor metadata and actual tensor data container.

```c
// n-dimensional tensor
struct ggml_tensor {
enum ggml_type type;
struct ggml_backend_buffer buffer;
int64_t ne[GGML_MAX_DIMS]; // number of elements
size_t nb[GGML_MAX_DIMS]; // stride in bytes:
                        // nb = ggml_type_size(type)
                        // nb = nb (ne / ggml_blck_size(type)) + padding
                        // nb[i] = nb[i-1] ne[i-1]
enum ggml_op op; // compute data
int32_t op_params[GGML_MAX_OP_PARAMS / sizeof(int32_t)]; // op params - allocated as int32_t for alignment
int32_t flags;
struct ggml_tensor src[GGML_MAX_SRC]; // source tensor and offset for views
struct ggml_tensor view_src;
size_t view_offs;
void *data;
char name[GGML_MAX_NAME];
void *extra; // extra things e.g. for ggml-cuda.cu
char padding;

```

### `ggml_cgraph`

This data structure is used for holding computation graphs: the "order of computation" that will be transferred to the backend.

```c
struct ggml_cgraph {
int size; // maximum number of nodes/leafs/grads/grad_accs
int n_nodes; // number of nodes currently in use
int n_leafs; // number of leafs currently in use
struct ggml_tensor nodes; // tensors with data that can change if the graph is evaluated
struct ggml_tensor grads; // the outputs of these tensors are the gradients of the nodes
struct ggml_tensor grad_accs; // accumulators for node gradients
struct ggml_tensor leafs; // tensors with constant data
struct ggml_hash_set visited_hash_set;
enum ggml_cgraph_eval_order order;
};

```


Please note that all these objects such as `ggml_tensor`, `ggml_cgraph`, etc., are created in the same container `ggml_context`. Every time we create an object, we store its metadata and shift the pointer `.mem_buffer` by that much offset, then store the actual data content and shift the `.mem_buffer` to its final end.

### Benefits

This way, all objects are stored in continuous space along with their respective metadata. Any benefits?

### Graph Building

`ggml_build_forward_expand(ctx, gf, n_threads)`: Method to create a computation graph by recusrively traversing from parent tensor.


### Graph Computation

The function call flow is as follows:
- `ggml_build_forward_expand()` → `ggml_build_forward_impl()` → `ggml_visit_parents()`: This final function builds a recursive traversal of graphs, given a parent node and traverses back to the child node recursively to build the graph.

**Q:** Why this method? Shouldn't all nodes automatically get connected as we create node tensors while defining the flow of compuation?

### Actual Computation

`ggml_graph_compute_with_ctx()`: This function does actual computation planning and execution.
1. **Planning**: 
   - `ggml_graph_plan()`: Plans computation for each node by dividing them into tasks and allocating these tasks over CPU/GPU/NPU.
     - Initialize thread count.
     - Initialize planning variables.
     - Analyze each node in the graph and count total memory buffer required for entire computation (`work_size`) along with all nodes in `cgraph`. Calculate max threads required for each node, ensuring it is less than max core support.
     - Add padding for cache line alignment.
     - Create final plan: returns max buffer required for holding intermediate data for computation and max threads required. Memory allocation does not happen here.

2. **Buffer Allocation**:
   - `ggml_new_buffer()`: Allocates an object in `ggml_context` that blocks that much memory (`work_size`) for intermediate computation requirements for the entire graph.

3. **Execution**:
   - `ggml_graph_compute()`: 
     1. Multiple threads compute all node computations.
     2. Thread-level synchronization.
     3. Call flow: 
        ```
        ggml_graph_compute() → ggml_graph_compute_thread() → ggml_compute_forward(&params, node) →
        ```
        **ggml_compute_forward(&params, node)**: This function calls various low-level tensor operation libraries depending on node operations.





------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------                                            LLM specific frontend in llama.cpp



`ggml_backend_load_all()`: loads the system library for backend device (cuda.so)

`struct llama_model_params XYZ = llama_model_default_params()`: this struct has features that tells how ur LLM will be computed These parameters control how the LLaMA model is loaded and executed, allowing for:
- GPU/CPU distribution
- Memory management
- Multi-device computation
- Progress monitoring
- Model validation
- Performance optimization

`struct llama_model`: this struct has all the features of LLM models (Weights, Activations, Tokenization) along with how it is stored processed and its performace metrics, so many thing. I have just put imp one for understanding

Imp is hyperparameters, vocab,

```cpp
struct llama_model {
    std::string name = "n/a";

    llama_hparams hparams = {};
    llama_vocab vocab;
    std::vector<llama_layer> layers; //VVI entire activation tensors go through the layers

    struct ggml_tensor * tok_embd = nullptr;
    struct ggml_tensor * pos_embd = nullptr;
    struct ggml_tensor * output = nullptr;

    // list of devices used in this model
    std::vector<ggml_backend_dev_t> devices;

    // lists of buffer types used for each layer
    using buft_list_t = std::vector<std::pair<ggml_backend_dev_t, ggml_backend_buffer_type_t>>;
    buft_list_t cpu_buft_list;
    std::map<ggml_backend_dev_t, buft_list_t> gpu_buft_list;

    // contexts where the model tensors metadata is stored
    std::vector<ggml_context_ptr> ctxs;

    // the model memory buffers for the tensor data
    std::vector<ggml_backend_buffer_ptr> bufs;

    // model memory mapped files
    llama_mmaps mappings;
};
```

`llama_context_params`: This gives the properties of context or prompt: like max context length, batch_size and many more
`llama_model`: about model arch details, model weights and everything

`llama_context`: entire config (model + prompt): that need to be used llm inference

```cpp
struct llama_context {
    llama_context(const llama_model & model) : model(model), t_start_us(model.t_start_us), t_load_us(model.t_load_us) {}

    const struct llama_model & model;

    struct llama_cparams        cparams;
    struct llama_sbatch         sbatch;  // TODO: revisit if needed
    struct llama_kv_cache       kv_self;
    struct llama_control_vector cvec;

    std::unordered_map<struct llama_lora_adapter *, float> lora_adapters;

    std::vector<ggml_backend_ptr> backends;
    std::vector<std::pair<ggml_backend_t, ggml_backend_set_n_threads_t>> set_n_threads_fns;

    ggml_backend_t backend_cpu = nullptr;

    ggml_threadpool_t threadpool       = nullptr;
    ggml_threadpool_t threadpool_batch = nullptr;

    bool has_evaluated_once = false;

    mutable int64_t t_start_us;
    mutable int64_t t_load_us;
    mutable int64_t t_p_eval_us = 0;
    mutable int64_t t_eval_us   = 0;

    mutable int64_t t_compute_start_us = 0;
    mutable int64_t n_queued_tokens = 0;

    mutable int32_t n_p_eval = 0;
    mutable int32_t n_eval   = 0;

    ggml_backend_buffer_ptr buf_output;

    size_t  logits_size = 0;
    float * logits      = nullptr;

    std::vector<int32_t> output_ids;
    size_t  output_size = 0;
    int32_t n_outputs   = 0;

    bool logits_all = false;

    size_t  embd_size = 0;
    float * embd      = nullptr;

    std::map<llama_seq_id, std::vector<float>> embd_seq;

    bool is_encoding = false;

    int n_pos_per_token = 1;

    std::vector<float> embd_enc;
    std::vector<std::set<llama_seq_id>> seq_ids_enc;

    std::vector<uint8_t> buf_compute_meta;
    ggml_backend_sched_ptr sched;

    ggml_abort_callback abort_callback      = nullptr;
    void *              abort_callback_data = nullptr;

    struct ggml_tensor * inp_tokens;
    struct ggml_tensor * inp_embd;
    struct ggml_tensor * inp_pos;
    struct ggml_tensor * inp_out_ids;
    struct ggml_tensor * inp_KQ_mask;
    struct ggml_tensor * inp_KQ_mask_swa;
    struct ggml_tensor * inp_K_shift;
    struct ggml_tensor * inp_mean;
    struct ggml_tensor * inp_cls;
    struct ggml_tensor * inp_s_copy;
    struct ggml_tensor * inp_s_mask;
    struct ggml_tensor * inp_s_seq;
    struct ggml_tensor * inp_pos_bucket;
    struct ggml_tensor * inp_embd_enc;
    struct ggml_tensor * inp_KQ_mask_cross;
};
```

`llama_batch`: this is the container data structure that holds entire data structure that need to be fed at the first stage of LLM inference. during decode() it is sent with tokens ID string that further in LLM specific computation graphs gets converted in input embeddings.

```cpp
typedef struct llama_batch {
    int32_t n_tokens;

    llama_token  *  token;
    float        *  embd;
    llama_pos    *  pos;
    int32_t      *  n_seq_id;
    llama_seq_id ** seq_id;
    int8_t       *  logits;
} llama_batch;
```

`llama_decode(ctx, batch)`: this is the main function that starts entire decode stage (given a batch it predicts next token), it takes input as llama_context and llama_batch. so this happens in for loop where you keep decoding until you hit EOS character or you terminate if (prompt_token + predicted token) hit the context length, or we set manually how many we want to predict in the for loop of generating text

the llama_batch keep updating as we predict the new tokens, then that token will be included in the new batch, and after prefill (first token decode)

call stack:

`llama_decode(ctx, batch)` --> `llama_decode_impl(*ctx, batch)` --> `llama_decode_impl(llama_context & lctx, llama_batch inp_batch)` --> `llama_build_graph(lctx, ubatch, false)` --> `result = llm.build_llama()`

`llama_decode_impl(llama_context & lctx, llama_batch inp_batch)`: This function does all the work of LLM inference.
1. Breaks the large batch into small ones
2. `llama_build_graph(lctx, ubatch, false)`: builds computation graph Q: how the batch details required in building computational graph? should not it be independent from batch details??
3. `ggml_backend_sched_alloc_graph(lctx.sched.get(), gf)`: allocate resources, this decides for entire computation for that ubatch the resources required, internally doing what we saw in ggml allocation, but for specific backend.
4. `llama_set_inputs(lctx, ubatch)`
5. `llama_graph_compute(lctx, gf, n_threads, threadpool)`
6. extract the logits: last tensor would be resulting token and second last token would be its embedding, on which classification operation done to result token.
7. get the KV Cache update

2. `static struct ggml_cgraph * llama_build_graph(llama_context & lctx, const llama_ubatch & ubatch, bool worst_case)`
   a. `llm_build_context llm(lctx, ubatch, cb, worst_case)`: this is container used for building entire graph. this will also include all the LLM models implementation graph as member methods build_xyzmodel(). and from here onward inside the function such as for example build_llama(), it starts calling ggml library functions for building llm graphs by giving ggml_context as input
   below is the flow of graph:
   1. Input token ID string to inpL embedding
   2. `inpSA = inpL` // Save input for residual connection
   3. Pre-Attention Normalization:
      `cur = llm_build_norm(ctx0, inpL, hparams, model.layers[il].attn_norm, NULL, LLM_NORM_RMS, cb, il)`
   4. Self-Attention Block:
      ```cpp
      // Create Q, K, V matrices
      Qcur = llm_build_lora_mm(lctx, ctx0, model.layers[il].wq, cur);
      Qcur = ggml_add(ctx0, Qcur, model.layers[il].bq);  // Add bias if present

      Kcur = llm_build_lora_mm(lctx, ctx0, model.layers[il].wk, cur);
      Kcur = ggml_add(ctx0, Kcur, model.layers[il].bk);  // Add bias if present

      Vcur = llm_build_lora_mm(lctx, ctx0, model.layers[il].wv, cur);
      Vcur = ggml_add(ctx0, Vcur, model.layers[il].bv);  // Add bias if present

      // Apply RoPE to Q and K
      Qcur = ggml_rope_ext(...);
      Kcur = ggml_rope_ext(...);

      // Compute attention and combine with values (MHA computation goes inside this)
      cur = llm_build_kv(ctx0, lctx, kv_self, gf, model.layers[il].wo, model.layers[il].bo,
      Kcur, Vcur, Qcur, KQ_mask, n_tokens, kv_head, n_kv, kq_scale, cb, il);
      ```

   5. Post-Attention Residual Connection:
      ```cpp
      struct ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpSA)
      ```

   6. MoE (Mixture of Experts) Branch
      (Modified form of ffn Standard FFN: All inputs → Single FFN → Output
      MoE: Input → Router → Selected Expert(s) → Combine Outputs → Output)

      ```cpp
      // Pre-FFN Normalization
      cur = llm_build_norm(ctx0, ffn_inp, hparams, model.layers[il].ffn_norm, NULL, LLM_NORM_RMS, cb, il);

      // MoE computation
      cur = llm_build_moe_ffn(ctx0, lctx, cur,
          model.layers[il].ffn_gate_inp,     // Router network
          model.layers[il].ffn_up_exps,      // Expert up-projection
          model.layers[il].ffn_gate_exps,    // Expert gates
          model.layers[il].ffn_down_exps,    // Expert down-projection
          nullptr,
          n_expert, n_expert_used,
          LLM_FFN_SILU, true,
          false, 0.0,
          LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX,
          cb, il)
      ```

   7. Final Residual Connection and Layer Output after FFN:
      ```cpp
      cur = ggml_add(ctx0, cur, ffn_inp);  // Add residual connection
      cur = lctx.cvec.apply_to(ctx0, cur, il);  // Apply any context vectors
      inpL = cur;  // Prepare for next layer
      ```

   8. After completing steps above for all layer attention, it exits the transformer block
   9. Final Normalization:
      ```cpp
      cur = llm_build_norm(ctx0, cur, hparams, model.output_norm, NULL, LLM_NORM_RMS, cb, -1)
      ```

   10. Language Model Head (lm_head):
       ```cpp
       cur = llm_build_lora_mm(lctx, ctx0, model.output, cur)
       ```

   11. Final Graph building by traversing from parent node:
       ```cpp
       ggml_build_forward_expand(gf, cur)
       ```

`llama_graph_compute(lctx, gf, n_threads, threadpool)` CALL STACK:
`ggml_backend_sched_graph_compute_async(lctx.sched.get(), gf)` --> `ggml_backend_sched_compute_splits(sched)` --> `ggml_backend_graph_compute_async(split_backend, &split->graph)` -->

```cpp
enum ggml_status ggml_backend_graph_compute_async(ggml_backend_t backend, struct ggml_cgraph * cgraph) {
    return backend->iface.graph_compute(backend, cgraph);
}
```

This is where it will start calling back-end specific compute kernels, for CPU it is `ggml_backend_cpu_graph_compute(ggml_backend_t backend, struct ggml_cgraph * cgraph)` --> `ggml_graph_compute(cgraph, &cplan)`; which we have already mentioned how down the line that distributes

ISSUE: How to ROI marker need to be places on the entire graph traversal ????


            







            











            






















