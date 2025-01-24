# GGML Learning Notes

## GGML Context and Memory Management

`ggml_context` is the main memory management structure in GGML like a container keeping everything in continuous memory space. When initialized, it performs two key memory allocations on the heap:

1. Context Structure (~40 bytes): Contains metadata like memory size, buffer pointers, and object tracking (struct member pointers)
2. Memory Buffer (user-defined size): Main storage for tensors and computations

Q: Advantage of keeping everything in continuous space? Memory access! Aren't default allocations also happening continuously?

struct ggml_context {
    size_t mem_size;
    void * mem_buffer;
    bool   mem_buffer_owned;
    bool   no_alloc;

    int    n_objects;

    struct ggml_object * objects_begin;
    struct ggml_object * objects_end;
};

// Example: Allocating context with 16MB buffer
struct ggml_init_params params = {
    .mem_size = 1610241024, // 16 MB
    .mem_buffer = NULL, // Let GGML handle allocation if pre-allocated memory by user then provider buffer_pointer
    .no_alloc = false,
};

struct ggml_context ctx = ggml_init(params);


Memory allocation happens in two steps:
1. `GGML_MALLOC(sizeof(struct ggml_context))` → ~40 bytes for context
2. `ggml_aligned_malloc(mem_size)` → 16MB for tensor storage (when `mem_buffer = NULL`)

The `mem_buffer` can either be allocated by GGML (NULL case) or pre-allocated by the user(when mem_buffer of allocated memory given), providing flexibility in memory management while ensuring proper alignment for SIMD operations.

`ggml_tensor`: data structure used for tensor metadata and actual tensor data container.

c
// n-dimensional tensor
struct ggml_tensor {
enum ggml_type type;
struct ggml_backend_buffer buffer;
int64_t ne[GGML_MAX_DIMS]; // number of elements
size_t nb[GGML_MAX_DIMS]; // stride in bytes:
// nb[0] = ggml_type_size(type)
// nb[1] = nb[0] (ne[0] / ggml_blck_size(type)) + padding
// nb[i] = nb[i-1] ne[i-1]
// compute data
enum ggml_op op;
// op params - allocated as int32_t for alignment
int32_t op_params[GGML_MAX_OP_PARAMS / sizeof(int32_t)];
int32_t flags;
struct ggml_tensor src[GGML_MAX_SRC];
// source tensor and offset for views
struct ggml_tensor view_src;
size_t view_offs;
void data;
char name[GGML_MAX_NAME];
void extra; // extra things e.g. for ggml-cuda.cu
char padding[8];
};


`ggml_cgraph`: data structure used for holding computation graphs: "order of computation" that will be transferred to the backend.

c
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

!alt text


Please Note all these objects such as ggml_tensor, ggml_cgraph, everything is created in the same container ggml_context. Every time we create some object, we store its metadata and shift the pointer .mem_buffer by that much offset and then store the actual data content and then shift the .mem_buffer to final end. 

This way all the objects are stored in continuous space along with its respective metadata. Any benefits?

`ggml_build_forward_expand`: method to create computation graph by adding tensor to the graph and updating the graph.

`struct ggml_tensor * c = ggml_add(ctx, a, b);`: this syntax will create add graph with two tensor a and b as leaf and c as parent and when we would do `ggml_build_forward_expand(gf, c);` then that would result in marking c as final result or final parent in the graph

ggml_build_forward_expand() -->ggml_build_forward_impl() --> ggml_visit_parents(): this final function builds recursive travel of graphs, given parent node and travels way back and the child node recursively to build the graph

Q. Why this method, should not all the nodes automatically get connected the way computation happens?

`ggml_graph_compute_with_ctx()`: this function does actual computation planning and computation. 
a. `ggml_graph_plan()`: does planning of computation of each node by dividing them in tasks and then allocating these tasks over CPU/GPU/NPU
    1. Initialize thread count
    2. Initialize planning variables
    3. Analyze each node in the graph and count total memory buffer required for entire comp(work_size) and along all the nodes in cgraph, calculation max threads required for each node and final max_task filed holds the max no of threads needed, if any node will require less thread then the other threads will idle? (it should def be less than max core support that is ensured)
    4. Add padding for cache line alignment
    5. Create final plan: here func returns max buffer required for holding intermediate data for computation, and max threads required for computation, but memory allocation does not happen here.

b. `ggml_new_buffer()`: allocates an object in ggml_context that blocks that much of memory(work_size) for intermediate computation requirement for entire graph

c. `ggml_graph_compute()`: 
    1. Multiple threads do the computation of all the nodes computation 
    2. Thread level synchronization 
    3. ggml_graph_compute()-->ggml_graph_compute_thread()-->ggml_compute_forward(&params, node)-->
       **ggml_compute_forward(&params, node)**: this is the function where depending on node OPS the different low level tensor operation library is called



