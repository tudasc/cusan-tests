// clang-format off
// RUN: %cusan-mpicxx %tsan-compile-flags -O2 -g %s -x cuda -gencode arch=compute_70,code=sm_70 -o %cutests_test_dir/%basename_t.exe
// RUN: %cusan-mpiexec -n 2 %cutests_test_dir/%basename_t.exe 2>&1 | %filecheck %s
// clang-format on

// CHECK-NOT: ThreadSanitizer: data race
// CHECK-NOT: Thread T{{[0-9]+}} 'cuda_stream'

// REQUIRES: mca-rules

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
  if (!has_gpu_aware_mpi()) {
    printf("[Error] This example is designed for CUDA-aware MPI. Exiting.\n");
    return 1;
  }

  const int size            = 512;
  const int threadsPerBlock = size;
  const int blocksPerGrid   = (size + threadsPerBlock - 1) / threadsPerBlock;

  MPI_Init(&argc, &argv);
  int world_size, world_rank;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

  if (world_size != 2) {
    printf("[Error] This example is designed for 2 MPI processes. Exiting.\n");
    MPI_Finalize();
    return 1;
  }

  int* d_data;
  cudaMalloc(&d_data, size * sizeof(int));
  cudaMemset(d_data, 0, size * sizeof(int));
  cudaDeviceSynchronize();

  if (world_rank == 0) {
    cudaStream_t stream1;
    cudaStreamCreate(&stream1);
    int* h_data = (int*)malloc(size * sizeof(int));

    kernel<<<blocksPerGrid, threadsPerBlock, 0, stream1>>>(d_data, size);
    // https://docs.nvidia.com/cuda/cuda-runtime-api/api-sync-behavior.html#api-sync-behavior__memcpy-async
    // "For transfers between device memory and pageable host memory, the
    // function might be synchronous with respect to host."
    cudaMemcpyAsync(h_data, d_data, size * sizeof(int), cudaMemcpyDeviceToHost, stream1);
    cudaStreamSynchronize(stream1);  // TEST FIX
    MPI_Send(d_data, size, MPI_INT, 1, 0, MPI_COMM_WORLD);

    cudaStreamSynchronize(stream1);
    cudaStreamDestroy(stream1);
    free(h_data);
  } else if (world_rank == 1) {
    MPI_Recv(d_data, size, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  }

  if (world_rank == 1) {
    int* h_data = (int*)malloc(size * sizeof(int));
    cudaMemcpy(h_data, d_data, size * sizeof(int), cudaMemcpyDeviceToHost);
    for (int i = 0; i < size; i++) {
      const int buf_v = h_data[i];
      if (buf_v == 0) {
        printf("[Error] sync\n");
        break;
      }
    }
    free(h_data);
  }

  cudaDeviceSynchronize();
  cudaFree(d_data);
  MPI_Finalize();
  return 0;
}
