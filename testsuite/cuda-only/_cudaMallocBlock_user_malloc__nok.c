// clang-format off
// RUN: %cusan-mpicxx %tsan-compile-flags -O2 -g -x cuda -gencode arch=compute_70,code=sm_70 %s  -o %cutests_test_dir/%basename_t.exe
// RUN: %tsan-options %cutests_test_dir/%basename_t.exe 2>&1 | %filecheck %s

// clang-format on

// CHECK-DAG: ThreadSanitizer: data race
// CHECK-DAG: [Error] sync

#include <cstdio>
#include <cuda_runtime.h>

__global__ void write_kernel_delay(int* arr, const int N, const unsigned int delay) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
#if __CUDA_ARCH__ >= 700
  for (int i = 0; i < tid; i++) {
    __nanosleep(delay);
  }
#else
  printf(">>> __CUDA_ARCH__ !\n");
#endif
  if (tid < N) {
    arr[tid] = (tid + 1);
  }
}

int main() {
  const int size            = 256;
  const int threadsPerBlock = size;
  const int blocksPerGrid   = (size + threadsPerBlock - 1) / threadsPerBlock;

  cudaStream_t stream1;
  cudaStream_t stream2;
  cudaStreamCreate(&stream1);
  cudaStreamCreate(&stream2);

  int* data;
  int* data2;
  int* h_data = (int*)malloc(size * sizeof(int));
  cudaMalloc(&data, size * sizeof(int));
  cudaMemset(data, 0, size * sizeof(int));

  cudaDeviceSynchronize();

  // write on data
  write_kernel_delay<<<blocksPerGrid, threadsPerBlock, 0, stream1>>>(data, size, 1316134912);
  // malloc does not block so we get a race
  cudaMalloc(&data2, size * sizeof(int));

  cudaMemcpyAsync(h_data, data, size * sizeof(int), cudaMemcpyDefault, stream2);
  cudaStreamSynchronize(stream2);
  for (int i = 0; i < size; i++) {
    if (h_data[i] == 0) {
      printf("[Error] sync %i\n", h_data[i]);
      break;
    }
  }

  cudaFree(data);
  free(h_data);
  cudaStreamDestroy(stream1);
  cudaStreamDestroy(stream2);

  return 0;
}
