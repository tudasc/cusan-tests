// clang-format off
// RUN: %cusan-mpicxx %tsan-compile-flags -O2 -g -x cuda -gencode arch=compute_70,code=sm_70 %s  -o %cutests_test_dir/%basename_t.exe
// RUN: %cusan-mpiexec -n 2 %cutests_test_dir/%basename_t.exe 2>&1 | %filecheck %s
// clang-format on

// CHECK-DAG: ThreadSanitizer: data race
// CHECK-DAG: Thread T{{[0-9]+}} 'cuda_stream{{( [0-9]+)?}}'

#include "../support/gpu_mpi.h"

__global__ void kernel_init(int* arr, const int N) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid < N) {
    arr[tid] = -1;
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

  if (world_rank == 0) {
    cudaMemset(d_data, 0, size * sizeof(int));
    cudaDeviceSynchronize();
  }

  MPI_Barrier(MPI_COMM_WORLD);

  MPI_Request request;
  if (world_rank == 0) {
    MPI_Isend(d_data, size, MPI_INT, 1, 0, MPI_COMM_WORLD, &request);
    // requires wait
    kernel_init<<<blocksPerGrid, threadsPerBlock>>>(d_data, size);
    MPI_Wait(&request, MPI_STATUS_IGNORE);
  } else if (world_rank == 1) {
    MPI_Irecv(d_data, size, MPI_INT, 0, 0, MPI_COMM_WORLD, &request);
    MPI_Wait(&request, MPI_STATUS_IGNORE);
  }

  if (world_rank == 1) {
    int* h_data = (int*)malloc(size * sizeof(int));
    cudaMemcpy(h_data, d_data, size * sizeof(int), cudaMemcpyDeviceToHost);
    for (int i = 0; i < size; i++) {
      const int buf_v = h_data[i];
      // Expect: all values should be 0, given the p_0 sends them (before
      // kernel call)
      if (buf_v != 0) {
        printf("[Error] sync\n");
        break;
      }
    }
    free(h_data);
  }

  cudaFree(d_data);
  MPI_Finalize();
  return 0;
}
