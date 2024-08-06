// clang-format off
// RUN: %cusan-mpicxx %tsan-compile-flags -O1 -g %s -x cuda -gencode arch=compute_70,code=sm_70 -o %cutests_test_dir/%basename_t.exe
// RUN: %tsan-options %cutests_test_dir/%basename_t.exe 2>&1 | %filecheck %s

// clang-format on

// HECK-DAG: data race
// CHECK-DAG: [Error] sync

#include <cstdio>
#include <cuda_runtime.h>
#include <unistd.h>

__global__ void write_kernel_delay(int* arr, const int N, const int value, const unsigned int delay) {
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
  write_kernel_delay<<<blocksPerGrid, threadsPerBlock, 0, stream1>>>(data, size, 0, 1316134912);
  // hostalloc does not sync so we get a race
  cudaHostAlloc(&data2, 1 * sizeof(int), cudaHostAllocDefault);
  write_kernel_delay<<<blocksPerGrid, threadsPerBlock, 0, stream2>>>(data, size, 255, 1);

  // sleep 1 second to allow the first kernel to overwrite some data
  sleep(1);

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
