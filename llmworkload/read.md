# GGML Learning Notes

## GGML Context and Memory Management

`ggml_context` is the main memory management structure in GGML like a container keeping everything in contineous memory space. When initialized, it performs two key memory allocations on the heap:

   1. Context Structure (~40 bytes): Contains metadata like memory size, buffer pointers, and object tracking (struct memeber pointers)
   2. Memory Buffer (user-defined size): Main storage for tensors and computations

Q: Advnatage of keeping everything in contineous space? Memory access ! arent default allocations also happens contineously? 

```c
struct ggml_context {
    size_t mem_size;
    void * mem_buffer;
    bool   mem_buffer_owned;
    bool   no_alloc;

    int    n_objects;

    struct ggml_object * objects_begin;
    struct ggml_object * objects_end;
};
```c
// Example: Allocating context with 16MB buffer
struct ggml_init_params params = {
.mem_size = 1610241024, // 16 MB
.mem_buffer = NULL, // Let GGML handle allocation if pre-allocated memory by user then privider buffer_pointer
.no_alloc = false,
};

struct ggml_context ctx = ggml_init(params);
```


Memory allocation happens in two steps:
1. `GGML_MALLOC(sizeof(struct ggml_context))` → ~40 bytes for context
2. `ggml_aligned_malloc(mem_size)` → 16MB for tensor storage (when `mem_buffer = NULL`)

The `mem_buffer` can either be allocated by GGML (NULL case) or pre-allocated by the user(when mem_buffer of allocvated memory given), providing flexibility in memory management while ensuring proper alignment for SIMD operations.


'ggml_tensor': data structure used for tensor meta data and actual tensor data conatiner.

```c
  // n-dimensional tensor
    struct ggml_tensor {
        enum ggml_type type;

        struct ggml_backend_buffer * buffer;

        int64_t ne[GGML_MAX_DIMS]; // number of elements
        size_t  nb[GGML_MAX_DIMS]; // stride in bytes:
                                   // nb[0] = ggml_type_size(type)
                                   // nb[1] = nb[0]   * (ne[0] / ggml_blck_size(type)) + padding
                                   // nb[i] = nb[i-1] * ne[i-1]

        // compute data
        enum ggml_op op;

        // op params - allocated as int32_t for alignment
        int32_t op_params[GGML_MAX_OP_PARAMS / sizeof(int32_t)];

        int32_t flags;

        struct ggml_tensor * src[GGML_MAX_SRC];

        // source tensor and offset for views
        struct ggml_tensor * view_src;
        size_t               view_offs;

        void * data;

        char name[GGML_MAX_NAME];

        void * extra; // extra things e.g. for ggml-cuda.cu

        char padding[8];
    };  

```

'ggml_cgraph': data structure used for holding compuation graphs 

Please Note all theese objects such as ggml_tensor , ggml_cgraph, everything is created in the same container ggml_context. every time we create some object , we store its meta data and shift the pointer .mem_buffer by that much offset and then store the actual data content and then shift the .mem_buffer to final end. 

This way all the objects are stored in contigeous space along with its respective metadata. any benifits ?

'ggml_build_forward_expand': method to create computation graph by adding tensor to the graph and uodating the graph.


struct ggml_tensor * c = ggml_add(ctx, a, b); : this sysntax will create add graph with two tensor a and b as leaf and c as parent and when we would do  ggml_build_forward_expand(gf, c); then that would result in marking c as final result or final parent in the graph

ggml_build_forward_expand() -->ggml_build_forward_impl() --> ggml_visit_parents() : this final function build recursive travel of graphs and mark final result as parent node, last one is graph




