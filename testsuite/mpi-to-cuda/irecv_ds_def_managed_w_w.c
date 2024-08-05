// clang-format off
// RUN: %cusan-mpicxx %tsan-compile-flags -O2 -g %s -x cuda -gencode arch=compute_70,code=sm_70 -o %cutests_test_dir/%basename_t.exe
// RUN: %cusan-mpiexec -n 2 %cutests_test_dir/%basename_t.exe 2>&1 | %filecheck %s
// clang-format on

// CHECK-NOT: ThreadSanitizer: data race
// CHECK-NOT: Thread T{{[0-9]+}} 'cuda_stream'
// CHECK-NOT: [Error]

#include "../support/gpu_mpi.h"

__global__ void kernel_init(int* arr, const int N) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid < N) {
    arr[tid] = -(tid + 1);
  }
}

__global__ void kernel(int* arr, const int N) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid < N) {
#if __CUDA_ARCH__ >= 700
    for (int i = 0; i < tid; i++) {
      __nanosleep(100U);
    }
#else
    printf(">>> __CUDA_ARCH__ !\n");
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
  cudaMallocManaged(&d_data, size * sizeof(int));

  if (world_rank == 0) {
    kernel_init<<<blocksPerGrid, threadsPerBlock>>>(d_data, size);
    cudaDeviceSynchronize();
  }

  MPI_Barrier(MPI_COMM_WORLD);

  if (world_rank == 0) {
    MPI_Send(d_data, size, MPI_INT, 1, 0, MPI_COMM_WORLD);
  } else if (world_rank == 1) {
    MPI_Request request;
    // Recv all negative numbers:
    MPI_Irecv(d_data, size, MPI_INT, 0, 0, MPI_COMM_WORLD, &request);
    MPI_Wait(&request, MPI_STATUS_IGNORE);  // TEST FIX
    kernel<<<blocksPerGrid, threadsPerBlock>>>(d_data, size);
  }

  if (world_rank == 1) {
    cudaDeviceSynchronize();
    for (int i = 0; i < size; i++) {
      const int buf_v = d_data[i];
      // Expect: all values should be positive, given the p_1 kernel sets them
      // to tid.
      if (buf_v < 1) {
        printf("[Error] sync\n");
        break;
      }
    }
  }

  cudaFree(d_data);
  MPI_Finalize();
  return 0;
}
