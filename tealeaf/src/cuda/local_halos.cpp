#include "chunk.h"
#include "cuknl_shared.h"
#include "shared.h"

__global__ void update_bottom(const int x, const int y, const int halo_depth, const int depth, double *buffer) {
  const int gid = threadIdx.x + blockIdx.x * blockDim.x;
  if (gid >= x * depth) return;

  const int lines = gid / x;
  const int offset = x * halo_depth;
  const int from_index = offset + gid;
  const int to_index = from_index - (1 + lines * 2) * x;
  buffer[to_index] = buffer[from_index];
}

__global__ void update_top(const int x, const int y, const int halo_depth, const int depth, double *buffer) {
  const int gid = threadIdx.x + blockIdx.x * blockDim.x;
  if (gid >= x * depth) return;

  const int lines = gid / x;
  const int offset = x * (y - halo_depth);
  const int to_index = offset + gid;
  const int from_index = to_index - (1 + lines * 2) * x;
  buffer[to_index] = buffer[from_index];
}

__global__ void update_left(const int x, const int y, const int halo_depth, const int depth, double *buffer) {
  const int gid = threadIdx.x + blockDim.x * blockIdx.x;
  if (gid >= y * depth) return;

  const int flip = gid % depth;
  const int lines = gid / depth;
  const int offset = halo_depth + lines * (x - depth);
  const int from_index = offset + gid;
  const int to_index = from_index - (1 + flip * 2);

  buffer[to_index] = buffer[from_index];
}

__global__ void update_right(const int x, const int y, const int halo_depth, const int depth, double *buffer) {
  const int gid = threadIdx.x + blockDim.x * blockIdx.x;
  if (gid >= y * depth) return;

  const int flip = gid % depth;
  const int lines = gid / depth;
  const int offset = x - halo_depth + lines * (x - depth);
  const int to_index = offset + gid;
  const int from_index = to_index - (1 + flip * 2);

  buffer[to_index] = buffer[from_index];
}

// Updates faces in turn.
void update_face(const int x, const int y, const int halo_depth, const int *chunk_neighbours, const int depth, double *buffer) {
  int num_blocks = std::ceil((x * depth) / (double)BLOCK_SIZE);
  if (chunk_neighbours[CHUNK_TOP] == EXTERNAL_FACE) {
    update_top<<<num_blocks, BLOCK_SIZE>>>(x, y, halo_depth, depth, buffer);
    check_errors(__LINE__, __FILE__);
  }
  if (chunk_neighbours[CHUNK_BOTTOM] == EXTERNAL_FACE) {
    update_bottom<<<num_blocks, BLOCK_SIZE>>>(x, y, halo_depth, depth, buffer);
    check_errors(__LINE__, __FILE__);
  }

  num_blocks = std::ceil((y * depth) / (float)BLOCK_SIZE);
  if (chunk_neighbours[CHUNK_RIGHT] == EXTERNAL_FACE) {
    update_right<<<num_blocks, BLOCK_SIZE>>>(x, y, halo_depth, depth, buffer);
    check_errors(__LINE__, __FILE__);
  }
  if (chunk_neighbours[CHUNK_LEFT] == EXTERNAL_FACE) {
    update_left<<<num_blocks, BLOCK_SIZE>>>(x, y, halo_depth, depth, buffer);
    check_errors(__LINE__, __FILE__);
  }
}

// The kernel for updating halos locally
void local_halos(const int x, const int y, const int halo_depth, const int depth, const int *chunk_neighbours,
                 const bool *fields_to_exchange, double *density, double *energy0, double *energy, double *u, double *p, double *sd) {
  if (fields_to_exchange[FIELD_DENSITY]) {
    update_face(x, y, halo_depth, chunk_neighbours, depth, density);
  }
  if (fields_to_exchange[FIELD_P]) {
    update_face(x, y, halo_depth, chunk_neighbours, depth, p);
  }
  if (fields_to_exchange[FIELD_ENERGY0]) {
    update_face(x, y, halo_depth, chunk_neighbours, depth, energy0);
  }
  if (fields_to_exchange[FIELD_ENERGY1]) {
    update_face(x, y, halo_depth, chunk_neighbours, depth, energy);
  }
  if (fields_to_exchange[FIELD_U]) {
    update_face(x, y, halo_depth, chunk_neighbours, depth, u);
  }
  if (fields_to_exchange[FIELD_SD]) {
    update_face(x, y, halo_depth, chunk_neighbours, depth, sd);
  }
}

// Solver-wide kernels
void run_local_halos(Chunk *chunk, Settings &settings, int depth) {
  START_PROFILING(settings.kernel_profile);

  local_halos(chunk->x, chunk->y, settings.halo_depth, depth, chunk->neighbours, settings.fields_to_exchange, chunk->density,
              chunk->energy0, chunk->energy, chunk->u, chunk->p, chunk->sd);

  STOP_PROFILING(settings.kernel_profile, __func__);
}
