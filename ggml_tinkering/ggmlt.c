#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-impl.h"
#include <stdio.h>
#include <stdlib.h>

int main() {
   
   // 1. Allocate `ggml_context` to store tensor data
    struct ggml_init_params params = {
        .mem_size   = 100*1024*1024,  // 16 MB  explcit lumpsum size
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


    // 3. Create a `ggml_cgraph` for some operation
   
    struct ggml_cgraph * gf = ggml_new_graph(ctx);
    printf("Number of nodes in graph: %d\n", gf->n_nodes);
    // Create multiple operations
    struct ggml_tensor * c = ggml_add(ctx, a, b);          
    struct ggml_tensor * roi_start = ggml_sim_roi_start_impl(ctx, c);          
    struct ggml_tensor * f = ggml_scale(ctx, roi_start, 0.5f);  
    struct ggml_tensor * roi_end = ggml_sim_roi_end_impl(ctx, f);
    struct ggml_tensor * g = ggml_relu(ctx, roi_end);            


    ggml_build_forward_expand(gf, g);
    printf("Number of nodes in graph: %d\n", gf->n_nodes);

        // struct ggml_tensor * node = gf->nodes[2];
        // printf("Node name: %s\n",node->name);
        // printf("Node type: %d\n", node->type);
        // printf("Node data: %p\n", node->data);
        // printf("Node ne: %ld\n", node->ne[0]); // need to check this as this should be 2D tensor 
        // printf("Node nb: %ld\n", node->nb[0]);
        // printf("Node op: %d\n", node->op);
        // printf("Node op_params: %d\n", node->op_params[0]);
        // printf("Node flags: %d\n", node->flags);
        // printf("Node src: %p\n", node->src[0]);
        // printf("Node view_src: %p\n", node->view_src);
        // printf("Node view_offs: %ld\n", node->view_offs);
        // int64_t total_elements = 1;
        // for (int i = 0; i < GGML_MAX_DIMS; i++) {
        // total_elements *= node->ne[i];
        // printf("Total elements: %ld\n", total_elements);
        // }
    
    
    // printf("Number of nodes in graph: %d\n", gf->n_nodes);



    // 4. Run the computation
    int n_threads = 1; // Optional: number of threads to perform some operations with multi-threading
    ggml_graph_compute_with_ctx(ctx, gf, n_threads);   

    // 5. Retrieve results (output tensors) 
//    float * result_data = ggml_get_data_f32(c);

  

   // 6. print the tensor result data 
    // printf("\nResult tensor c (a + b):\n");
    // for (int i = 0; i < 4; i++) {
    //     printf("Result[%d]: %.1f\n", i, result_data[i]);
    // }


    // Print computational graph to a file
    FILE* dot_file = fopen("graph.dot", "w");
    if (dot_file) {
        ggml_graph_dump_dot(gf, gf, "graph.dot");
        fclose(dot_file);
        printf("\nComputational graph has been written to 'graph.dot'\n");
        // printf("To visualize it, install Graphviz and run: dot -Tpng graph.dot -o graph.png\n");
    } else {
        printf("Failed to open file for writing graph\n");
    }

    // 7. Free memory and exit
    ggml_free(ctx);

    return 0;
}
  
