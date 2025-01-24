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






