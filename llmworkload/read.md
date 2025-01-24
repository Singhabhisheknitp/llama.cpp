# GGML Learning Notes

## GGML Context and Memory Management

The `ggml_context` is the main memory management structure in GGML like a container keeping everything in contineous memory space. When initialized, it performs two key memory allocations on the heap:

Q: Advnatage of keeping everything in contineous space? Memory access ! arent default allocations also happens contineously? 

1. **Context Structure** (~40 bytes): Contains metadata like memory size, buffer pointers, and object tracking (struct memeber pointers)
2. **Memory Buffer** (user-defined size): Main storage for tensors and computations


// Example: Allocating context with 16MB buffer
struct ggml_init_params params = {
.mem_size = 1610241024, // 16 MB
.mem_buffer = NULL, // Let GGML handle allocation
.no_alloc = false,
};
struct ggml_context ctx = ggml_init(params);


Memory allocation happens in two steps:
1. `GGML_MALLOC(sizeof(struct ggml_context))` → ~40 bytes for context
2. `ggml_aligned_malloc(mem_size)` → 16MB for tensor storage (when `mem_buffer = NULL`)

The `mem_buffer` can either be allocated by GGML (NULL case) or pre-allocated by the user(when mem_buffer of allocvated memory given), providing flexibility in memory management while ensuring proper alignment for SIMD operations.
