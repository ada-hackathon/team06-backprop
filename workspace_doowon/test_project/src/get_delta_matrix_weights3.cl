/*
void get_delta_matrix_weights3(TYPE delta_weights3[nodes_per_layer*possible_outputs],
                               TYPE output_difference[possible_outputs],
                               TYPE last_activations[nodes_per_layer]) {
    int i, j;
    for( i = 0; i < nodes_per_layer; i++) {
        for( j = 0; j < possible_outputs; j++) {
            delta_weights3[i*possible_outputs + j] = last_activations[i] * output_difference[j];
        }
    }
}
*/

__kernel
void get_delta_matrix_weights3(__global TYPE* delta_weights3,
                               __global TYPE* output_difference,
                               __global TYPE* last_activations,
                               int possible_outputs) {
    int i = get_global_id(0);
    int j = 0;
    for( j = 0; j < possible_outputs; j++) {
        delta_weights3[i*possible_outputs + j] = last_activations[i] * output_difference[j];
    }
}

