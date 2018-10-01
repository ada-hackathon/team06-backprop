/*
void get_delta_matrix_weights2(TYPE delta_weights2[nodes_per_layer*nodes_per_layer],
                               TYPE output_difference[nodes_per_layer],
                               TYPE last_activations[nodes_per_layer]) {
    int i, j;
    for( i = 0; i < nodes_per_layer; i++) {
        for( j = 0; j < nodes_per_layer; j++) {
            delta_weights2[i*nodes_per_layer + j] = last_activations[i] * output_difference[j];
        }
    }
}
*/

__global
void get_delta_matrix_weights2(__global TYPE* delta_weights2,
                               __global TYPE* output_difference,
                               __global TYPE* last_activations,
                               int nodes_per_layer) {
    int i = get_global_id(0), j;
    for( j = 0; j < nodes_per_layer; j++) {
        delta_weights2[i*nodes_per_layer + j] = last_activations[i] * output_difference[j];
    }
}