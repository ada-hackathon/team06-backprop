__kernel
void softmax_and_take_difference(__global const TYPE *activations, 
                                 __global const TYPE *dactivations,
                                 __global const TYPE *solutions,
                                 __global const TYPE *output_difference,
                                 int possible_outputs) {
    int i = get_global_id(0);

    TYPE sum = (TYPE)0.0;
    int j;
    for(j = 0; j < possible_outputs; j++) {
        sum += exp(-activations[j]);
    }

    TYPE net_output = exp(-activations[i]) / sum;
    output_difference[i] = (solutions[i] - net_output) * dactivations[i];
}

