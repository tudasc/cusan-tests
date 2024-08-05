// clang-format off
// RUN: %cusan-mpicxx %tsan-compile-flags -O2 -g %s -x cuda -gencode arch=compute_70,code=sm_70 -o %cutests_test_dir/%basename_t.exe
// RUN: %tsan-options %cutests_test_dir/%basename_t.exe 2>&1 | %filecheck %s
// clang-format on

// CHECK-DAG: ThreadSanitizer: data race
// CHECK-DAG: Thread T{{[0-9]+}} 'cuda_stream'
// CHECK-DAG: [Error]

#include "../support/gpu_mpi.h"

#include <unistd.h>

__global__ void write_kernel_delay(int* arr, const int N, int value, const unsigned int delay) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
#if __CUDA_ARCH__ >= 700
  for (int i = 0; i < tid; i++) {
    __nanosleep(delay);
  }
#else
  printf(">>> __CUDA_ARCH__ !\n");
#endif
  if (tid < N) {
    arr[tid] = value;
  }
}

int main(int argc, char* argv[]) {
  cudaStream_t stream1;
  cudaStream_t stream2;
  cudaStreamCreate(&stream1);
  cudaStreamCreate(&stream2);

  const int size            = 512;
  const int threadsPerBlock = size;
  const int blocksPerGrid   = (size + threadsPerBlock - 1) / threadsPerBlock;

  int* managed_data;
  cudaMallocManaged(&managed_data, size * sizeof(int));
  cudaMemset(managed_data, 0, size * sizeof(int));

  int* d_data2;
  cudaMalloc(&d_data2, size * sizeof(int));
  cudaDeviceSynchronize();

  // write access to managed
  write_kernel_delay<<<blocksPerGrid, threadsPerBlock, 0, stream1>>>(managed_data, size, 128, 9999999);
  // this kernel runs on default as such it implicitly waits for the previous
  // kernel to finish
  write_kernel_delay<<<blocksPerGrid, threadsPerBlock, 0, 0>>>(d_data2, size, 0, 1);
  // and also blocks this next kernel from starting until it is finished.
  write_kernel_delay<<<blocksPerGrid, threadsPerBlock, 0, stream2>>>(d_data2, size, 128, 1);
  // Then if we sync on the last kernel we also know that the previous 2 must
  // also be finished
  // cudaStreamSynchronize(stream2);

  for (int i = 0; i < size; i++) {
    if (managed_data[i] == 0) {
      printf("[Error] sync %i %i\n", managed_data[i], i);
      break;
    }
  }

  cudaFree(managed_data);
  cudaFree(d_data2);
  cudaStreamDestroy(stream1);
  cudaStreamDestroy(stream2);
  return 0;
}
