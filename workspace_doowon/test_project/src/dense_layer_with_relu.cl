__kernel
void dense_layer_with_relu(__global const TYPE *biases, 
                           __global const TYPE *weights, 
                           __global TYPE *out_activations, 
                           __global TYPE *dactivations, 
                           __global const TYPE *in_activations, 
                           int in_size) {
    int i = get_global_id(0);
    TYPE out = biases[i];

    int j;
    for (j = 0; j < in_size; ++j) {
        out += weights[i * in_size + j] * in_activations[j];
    }

    dactivations[i] = out * (1.0 - out);
    out_activations[i] = 1.0 / (1.0 + exp(-out));
}

