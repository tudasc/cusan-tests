// clang-format off
// RUN: %cusan-mpicxx %tsan-compile-flags -O2 -g %s -x cuda -gencode arch=compute_70,code=sm_70 -o %cutests_test_dir/%basename_t.exe
// RUN: %tsan-options %cutests_test_dir/%basename_t.exe 2>&1 | %filecheck %s
// clang-format on

// CHECK-DAG: ThreadSanitizer: data race
// CHECK-DAG: Thread T{{[0-9]+}} 'cuda_stream'
// CHECK-DAG: [Error]

// REQUIRES: mca-rules

#include "../support/gpu_mpi.h"

__global__ void kernel(int* arr, const int N) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid < N) {
#if __CUDA_ARCH__ >= 700
    for (int i = 0; i < tid; i++) {
      __nanosleep(100000000U);
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

  int *d_data, *m_data;
  cudaMalloc(&d_data, size * sizeof(int));
  cudaMallocManaged(&m_data, size * sizeof(int));
  cudaMemset(m_data, 0, size * sizeof(int));
  cudaDeviceSynchronize();

  kernel<<<blocksPerGrid, threadsPerBlock>>>(m_data, size);
  // https://docs.nvidia.com/cuda/cuda-driver-api/api-sync-behavior.html
  // 4. For transfers from device memory to device memory, no host-side
  // synchronization is performed.
  cudaMemcpy(d_data, d_data, 1, cudaMemcpyDeviceToDevice);

  for (int i = 0; i < size; i++) {
    const int buf_v = m_data[i];
    if (buf_v == 0) {
      printf("[Error] sync\n");
      break;
    }
  }

  cudaDeviceSynchronize();
  cudaFree(d_data);
  cudaFree(m_data);
  return 0;
}
