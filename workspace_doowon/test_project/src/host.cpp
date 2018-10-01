/**********
Copyright (c) 2018, Xilinx, Inc.
All rights reserved.

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice,
this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
this list of conditions and the following disclaimer in the documentation
and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors
may be used to endorse or promote products derived from this software
without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
**********/
//OpenCL utility layer include
#include "xcl2.hpp"
#include <iostream>
#include <vector>
#include "backprop.h"
#include <string.h>


#define DATA_SIZE 256
#define COLS 16
#define EPSILON (1.0e-6)

///// TYPE macros
// Macro trick to automatically expand TYPE into the appropriate function
// (S)et (T)ype (A)nd (C)oncatenate
#define __STAC_EXPANDED(f_pfx,t,f_sfx) f_pfx##t##f_sfx
#define STAC(f_pfx,t,f_sfx) __STAC_EXPANDED(f_pfx,t,f_sfx)
// Invoke like this:
//   #define TYPE int32_t
//   STAC(write_,TYPE,_array)(fd, array, n);
// where array is of type (TYPE *)
// This translates to:
//   write_int32_t_array(fd, array, n);

int INPUT_SIZE = sizeof(struct bench_args_t);

void run_benchmark( void *vargs ) {
  struct bench_args_t *args = (struct bench_args_t *)vargs;
  backprop( args->weights1, args->weights2, args->weights3,
            args->biases1,  args->biases2,  args->biases3,
            args->training_data, args->training_targets );
}

/* Input format:
%% Section 1
TYPE[row_size*col_size]: input matrix
%% Section 2
TYPE[f_size]: filter coefficients
*/

void input_to_data(int fd, void *vdata) {
  struct bench_args_t *data = (struct bench_args_t *)vdata;
  char *p, *s;
  // Zero-out everything.
  memset(vdata,0,sizeof(struct bench_args_t));

  // Load input string
  p = readfile(fd);

  s = find_section_start(p,1);
  STAC(parse_,TYPE,_array)(s, data->weights1, input_dimension*nodes_per_layer);

  s = find_section_start(p,2);
  STAC(parse_,TYPE,_array)(s, data->weights2, nodes_per_layer*nodes_per_layer);

  s = find_section_start(p,3);
  STAC(parse_,TYPE,_array)(s, data->weights3, nodes_per_layer*possible_outputs);

  s = find_section_start(p,4);
  STAC(parse_,TYPE,_array)(s, data->biases1, nodes_per_layer);

  s = find_section_start(p,5);
  STAC(parse_,TYPE,_array)(s, data->biases2, nodes_per_layer);

  s = find_section_start(p,6);
  STAC(parse_,TYPE,_array)(s, data->biases3, possible_outputs);

  s = find_section_start(p,7);
  STAC(parse_,TYPE,_array)(s, data->training_data, training_sets*input_dimension);

  s = find_section_start(p,8);
  STAC(parse_,TYPE,_array)(s, data->training_targets, training_sets*possible_outputs);
  free(p);
}

void data_to_input(int fd, void *vdata) {
  struct bench_args_t *data = (struct bench_args_t *)vdata;

  write_section_header(fd);
  STAC(write_,TYPE,_array)(fd, data->weights1, input_dimension*nodes_per_layer);

  write_section_header(fd);
  STAC(write_,TYPE,_array)(fd, data->weights2, nodes_per_layer*nodes_per_layer);

  write_section_header(fd);
  STAC(write_,TYPE,_array)(fd, data->weights3, nodes_per_layer*possible_outputs);

  write_section_header(fd);
  STAC(write_,TYPE,_array)(fd, data->biases1, nodes_per_layer);

  write_section_header(fd);
  STAC(write_,TYPE,_array)(fd, data->biases2, nodes_per_layer);

  write_section_header(fd);
  STAC(write_,TYPE,_array)(fd, data->biases3, possible_outputs);

  write_section_header(fd);
  STAC(write_,TYPE,_array)(fd, data->training_data, training_sets*input_dimension);

  write_section_header(fd);
  STAC(write_,TYPE,_array)(fd, data->training_targets, training_sets*possible_outputs);
}

/* Output format:
%% Section 1
TYPE[row_size*col_size]: solution matrix
*/

void output_to_data(int fd, void *vdata) {
  struct bench_args_t *data = (struct bench_args_t *)vdata;
  char *p, *s;
  // Zero-out everything.
  memset(vdata,0,sizeof(struct bench_args_t));
  // Load input string
  p = readfile(fd);

  s = find_section_start(p,1);
  STAC(parse_,TYPE,_array)(s, data->weights1, input_dimension*nodes_per_layer);

  s = find_section_start(p,2);
  STAC(parse_,TYPE,_array)(s, data->weights2, nodes_per_layer*nodes_per_layer);

  s = find_section_start(p,3);
  STAC(parse_,TYPE,_array)(s, data->weights3, nodes_per_layer*possible_outputs);

  s = find_section_start(p,4);
  STAC(parse_,TYPE,_array)(s, data->biases1, nodes_per_layer);

  s = find_section_start(p,5);
  STAC(parse_,TYPE,_array)(s, data->biases2, nodes_per_layer);

  s = find_section_start(p,6);
  STAC(parse_,TYPE,_array)(s, data->biases3, possible_outputs);
  free(p);

}

void data_to_output(int fd, void *vdata) {
  struct bench_args_t *data = (struct bench_args_t *)vdata;

  write_section_header(fd);
  STAC(write_,TYPE,_array)(fd, data->weights1, input_dimension*nodes_per_layer);

  write_section_header(fd);
  STAC(write_,TYPE,_array)(fd, data->weights2, nodes_per_layer*nodes_per_layer);

  write_section_header(fd);
  STAC(write_,TYPE,_array)(fd, data->weights3, nodes_per_layer*possible_outputs);

  write_section_header(fd);
  STAC(write_,TYPE,_array)(fd, data->biases1, nodes_per_layer);

  write_section_header(fd);
  STAC(write_,TYPE,_array)(fd, data->biases2, nodes_per_layer);

  write_section_header(fd);
  STAC(write_,TYPE,_array)(fd, data->biases3, possible_outputs);

}

int check_data( void *vdata, void *vref ) {
  struct bench_args_t *data = (struct bench_args_t *)vdata;
  struct bench_args_t *ref = (struct bench_args_t *)vref;
  int has_errors = 0;
  int i, j;
  TYPE diff;

  for(i=0; i<input_dimension; i++) {
    for(j=0; j<nodes_per_layer; j++) {
      diff = data->weights1[i*nodes_per_layer + j] - ref->weights1[i*nodes_per_layer + j];
      has_errors |= (diff<-EPSILON) || (EPSILON<diff);
    }
  }
  for(i=0; i<nodes_per_layer; i++) {
    for(j=0; j<nodes_per_layer; j++) {
      diff = data->weights2[i*nodes_per_layer + j] - ref->weights2[i*nodes_per_layer + j];
      has_errors |= (diff<-EPSILON) || (EPSILON<diff);
    }
  }
  for(i=0; i<nodes_per_layer; i++) {
    for(j=0; j<possible_outputs; j++) {
      diff = data->weights3[i*possible_outputs + j] - ref->weights3[i*possible_outputs + j];
      has_errors |= (diff<-EPSILON) || (EPSILON<diff);
    }
  }
  for(i=0; i<nodes_per_layer; i++) {
    diff = data->biases1[i] - ref->biases1[i];
    has_errors |= (diff<-EPSILON) || (EPSILON<diff);
  }
  for(i=0; i<nodes_per_layer; i++) {
    diff = data->biases2[i] - ref->biases2[i];
    has_errors |= (diff<-EPSILON) || (EPSILON<diff);
  }
  for(i=0; i<possible_outputs; i++) {
    diff = data->biases3[i] - ref->biases3[i];
    has_errors |= (diff<-EPSILON) || (EPSILON<diff);
  }
  // Return true if it's correct.
  return !has_errors;
}
using namespace std;
int main(int argc, char** argv)
{
    // doowon: copied from main() in harness.cpp
    // Parse command line.
    char *in_file;
    char *check_file;
    assert( argc<4 && "Usage: ./benchmark <input_file> <check_file>" );
    in_file = "input.data";
    check_file = "check.data";
    if( argc>1 )
      in_file = argv[1];
    if( argc>2 )
      check_file = argv[2];
  
    // Load input data
    int in_fd;
    char *data;
    data = malloc(INPUT_SIZE);
    assert( data!=NULL && "Out of memory" );
    in_fd = open( in_file, O_RDONLY );
    assert( in_fd>0 && "Couldn't open input data file");
    input_to_data(in_fd, data);
  
    // Unpack and call
    run_benchmark( data );
  
    int out_fd;
    out_fd = open("output.data", O_WRONLY|O_CREAT|O_TRUNC, S_IRUSR|S_IWUSR|S_IRGRP|S_IWGRP|S_IROTH|S_IWOTH);
    assert( out_fd>0 && "Couldn't open output data file" );
    data_to_output(out_fd, data);
    close(out_fd);
  
    // Load check data
    int check_fd;
    char *ref;
    ref = malloc(INPUT_SIZE);
    assert( ref!=NULL && "Out of memory" );
    check_fd = open( check_file, O_RDONLY );
    assert( check_fd>0 && "Couldn't open check data file");
    output_to_data(check_fd, ref);
  
    // Validate benchmark results
    if( !check_data(data, ref) ) {
      fprintf(stderr, "Benchmark results are incorrect\n");
      return -1;
    }
    free(data);
    free(ref);
  
    printf("Success.\n");
    // end of main() in harness.cpp

    return 0;

    //Allocate Memory in Host Memory
    size_t vector_size_bytes = sizeof(float) * DATA_SIZE;

    //Initialize inputs
    std::vector<float,aligned_allocator<float>> source_input1     (DATA_SIZE);
    std::vector<float,aligned_allocator<float>> source_input2     (DATA_SIZE);
    std::vector<float,aligned_allocator<float>> source_hw_results(DATA_SIZE);
    std::vector<float,aligned_allocator<float>> source_sw_results(DATA_SIZE);

    // Create the test data and Software Result 
    for(int i = 0 ; i < DATA_SIZE ; i++){
        source_input1[i] = i;
        source_input2[i] = i;
        source_hw_results[i] = 0;
    }

    //software matrix multiplier

    for(int i =0; i< DATA_SIZE/COLS ;i++)
    	 for(int j = 0;j<COLS;j++)
    	 {   int temp =0;
    		 for(int k = 0;k< COLS;k++)
    		 temp = temp + source_input1[i*COLS + k] * source_input2[k*COLS +j];
    		 source_sw_results[i*COLS +j] = temp;

    	 }

  //  for(int i= 0;i< DATA_SIZE;i++)
   // 	cout<<source_sw_results[i]<<endl;
//OPENCL HOST CODE AREA START
    std::vector<cl::Device> devices = xcl::get_xil_devices();
    cl::Device device = devices[0];

    cl::Context context(device);
    cl::CommandQueue q(context, device, CL_QUEUE_PROFILING_ENABLE);
    std::string device_name = device.getInfo<CL_DEVICE_NAME>(); 

    //Create Program and Kernel
    std::string binaryFile = xcl::find_binary_file(device_name,"mult");
    cl::Program::Binaries bins = xcl::import_binary_file(binaryFile);
    devices.resize(1);
    cl::Program program(context, devices, bins);
    cl::Kernel krnl_mult(program,"mult");

    //Allocate Buffer in Global Memory
    cl::Buffer buffer_input1 (context, CL_MEM_READ_ONLY,
                        vector_size_bytes);
    cl::Buffer buffer_input2 (context, CL_MEM_READ_ONLY,
                           vector_size_bytes);
    cl::Buffer buffer_output(context, CL_MEM_WRITE_ONLY, 
                            vector_size_bytes);

    //Copy input data to device global memory
    q.enqueueWriteBuffer(buffer_input1, CL_TRUE, 0, vector_size_bytes, source_input1.data());
    q.enqueueWriteBuffer(buffer_input2, CL_TRUE, 0, vector_size_bytes, source_input2.data());

   // int inc = INCR_VALUE;
    int size = DATA_SIZE;
    int cols = COLS;
    //Set the Kernel Arguments
    int narg=0;
    krnl_mult.setArg(narg++,buffer_input1);
    krnl_mult.setArg(narg++,buffer_input2);
    krnl_mult.setArg(narg++,buffer_output);
    krnl_mult.setArg(narg++,cols);

    //Launch the Kernel
    q.enqueueNDRangeKernel(krnl_mult,cl::NullRange,cl::NDRange(cols,size/cols),cl::NullRange);

    //Copy Result from Device Global Memory to Host Local Memory
    q.enqueueReadBuffer(buffer_output, CL_TRUE, 0, vector_size_bytes, source_hw_results.data());

    q.finish();

//OPENCL HOST CODE AREA END
    
    // Compare the results of the Device to the simulation
    bool match = true;
    for (int i = 0 ; i < DATA_SIZE ; i++){
        if (source_hw_results[i] != source_sw_results[i]){
            std::cout << "Error: Result mismatch" << std::endl;
            std::cout << "i = " << i << " CPU result = " << source_sw_results[i]
                << " Device result = " << source_hw_results[i] << std::endl;
            match = false;
          break;
       }
    }

    std::cout << "TEST " << (match ? "PASSED" : "FAILED") << std::endl; 
    return (match ? EXIT_SUCCESS :  EXIT_FAILURE);
}
