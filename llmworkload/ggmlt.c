#include "ggml.h"
#include "ggml-cpu.h"
#include <stdio.h>
#include <stdlib.h>

int main() {
   
   // 1. Allocate `ggml_context` to store tensor data
   // ggml_context: A "container" that holds objects such as tensors, graphs, and optionally data
    struct ggml_init_params params = {
        .mem_size   = 16*1024*1024,  // 16 MB  explcit lumpsum size
        .mem_buffer = NULL,
        .no_alloc   = false,
    }; 
    struct ggml_context * ctx = ggml_init(params);
  


    // 2. Create tensors and set data
    // Create two 2x2 tensors
    struct ggml_tensor * a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2, 2);
    struct ggml_tensor * b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2, 2);

    // Initialize tensor values
    float * a_data = ggml_get_data_f32(a);
    float * b_data = ggml_get_data_f32(b);

    // Set values for tensor a
    a_data[0] = 1.0f;  // [1.0, 2.0]
    a_data[1] = 2.0f;  // [3.0, 4.0]
    a_data[2] = 3.0f;
    a_data[3] = 4.0f;

    // Set values for tensor b
    b_data[0] = 5.0f;  // [5.0, 6.0]
    b_data[1] = 6.0f;  // [7.0, 8.0]
    b_data[2] = 7.0f;
    b_data[3] = 8.0f;


    // 3. Create a `ggml_cgraph` for mul_mat operation
    struct ggml_cgraph * gf = ggml_new_graph(ctx);
    // aad of gtwo ggml tensors
    struct ggml_tensor * c = ggml_add(ctx, a, b);
    // Mark the "c" tensor to be computed
    ggml_build_forward_expand(gf, c);



    // 4. Run the computation
    int n_threads = 1; // Optional: number of threads to perform some operations with multi-threading
    ggml_graph_compute_with_ctx(ctx, gf, n_threads);   

    // 5. Retrieve results (output tensors) 
   float * result_data = (float *) c->data;

   // 6. print the tensor result data 
   for (int i = 0; i < 4; i++) {
    printf("Result[%d]: %.1f\n", i, result_data[i]);
   }


    // 7. Free memory and exit
    ggml_free(ctx);

    return 0;
}
