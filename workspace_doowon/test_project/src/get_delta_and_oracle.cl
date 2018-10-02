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

void get_oracle_activations2(TYPE weights3[nodes_per_layer*possible_outputs], 
                             TYPE output_differences[possible_outputs], 
                             TYPE oracle_activations[nodes_per_layer],
                             TYPE dactivations[nodes_per_layer]) {
    int i, j;
    for( i = 0; i < nodes_per_layer; i++) {
        oracle_activations[i] = (TYPE)0.0;
        for( j = 0; j < possible_outputs; j++) {
            oracle_activations[i] += output_differences[j] * weights3[i*possible_outputs + j];
        }
        oracle_activations[i] = oracle_activations[i] * dactivations[i];
    }
}

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

void get_oracle_activations1(TYPE weights2[nodes_per_layer*nodes_per_layer], 
                             TYPE output_differences[nodes_per_layer], 
                             TYPE oracle_activations[nodes_per_layer],
                             TYPE dactivations[nodes_per_layer]) {
    int i, j;
    for( i = 0; i < nodes_per_layer; i++) {
        oracle_activations[i] = (TYPE)0.0;
        for( j = 0; j < nodes_per_layer; j++) {
            oracle_activations[i] += output_differences[j] * weights2[i*nodes_per_layer + j];
        }
        oracle_activations[i] = oracle_activations[i] * dactivations[i];
    }
}

*/

__global
void get_delta_and_oracle(__global TYPE* output_difference,
                          __global TYPE* activations1,
                          __global TYPE* activations2,
                          __global TYPE* weights2,
                          __global TYPE* weights3,
                          __global TYPE* dactivations1,
                          __global TYPE* delta_weights3,
                          __global TYPE* delta_weights2,
                          __global TYPE* oracle_activations2,
                          __global TYPE* oracle_activations1,
                          int possible_outputs,
                          int nodes_per_layer)
{
    int i = get_global_id(0), j;
    for( j = 0; j < possible_outputs; j++) {
        delta_weights3[i*possible_outputs + j] = activations2[i] * output_difference[j];
    }

    oracle_activations2[i] = (TYPE)0.0;
    for( j = 0; j < possible_outputs; j++) {
        oracle_activations2[i] += output_differences[j] * weights3[i*possible_outputs + j];
    }
    oracle_activations2[i] = oracle_activations[i] * dactivations[i];

    for( j = 0; j < nodes_per_layer; j++) {
        delta_weights2[i*nodes_per_layer + j] = activation1[i] * output_difference[j];
    }

    oracle_activations1[i] = (TYPE)0.0;
    for( j = 0; j < nodes_per_layer; j++) {
        oracle_activations1[i] += output_differences[j] * weights2[i*nodes_per_layer + j];
    }
    oracle_activations1[i] = oracle_activations[i] * dactivations1[i];
}