// clang-format off
// RUN: %cusan-mpicxx %tsan-compile-flags -O2 -g %s -x cuda -gencode arch=compute_70,code=sm_70 -o %cutests_test_dir/%basename_t.exe
// RUN: %cutests_test_dir/%basename_t.exe 2>&1 | %filecheck %s
// clang-format on

// CHECK-NOT: ThreadSanitizer: data race
// CHECK-NOT: Thread T{{[0-9]+}} 'cuda_stream'
// CHECK-NOT: [Error] sync

#include "../support/gpu_mpi.h"

__global__ void kernel(int* arr, const int N) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid < N) {
#if __CUDA_ARCH__ >= 700
    for (int i = 0; i < tid; i++) {
      __nanosleep(1000000U);
    }
#else
    printf("[Error] __CUDA_ARCH__ !\n");
#endif
    arr[tid] = (tid + 1);
  }
}

int main(int argc, char* argv[]) {
  const int size            = 512;
  const int threadsPerBlock = size;
  const int blocksPerGrid   = (size + threadsPerBlock - 1) / threadsPerBlock;

  int* m_data;
  cudaMallocManaged(&m_data, size * sizeof(int));
  cudaMemset(m_data, 0, size * sizeof(int));
  cudaDeviceSynchronize();

  cudaStream_t stream1;
  cudaStreamCreate(&stream1);
  int* h_data = (int*)malloc(size * sizeof(int));

  kernel<<<blocksPerGrid, threadsPerBlock, 0>>>(m_data, size);
  // https://docs.nvidia.com/cuda/cuda-runtime-api/api-sync-behavior.html#api-sync-behavior__memcpy-async
  // "For transfers from any host memory to any host memory, the function is
  // fully synchronous with respect to the host."
  cudaMemcpyAsync(h_data, h_data, 1 * sizeof(int), cudaMemcpyHostToHost, stream1);

  for (int i = 0; i < size; i++) {
    const int buf_v = m_data[i];
    if (buf_v == 0) {
      printf("[Error] sync\n");
      break;
    }
  }

  cudaStreamDestroy(stream1);
  cudaDeviceSynchronize();
  cudaFree(m_data);
  return 0;
}
