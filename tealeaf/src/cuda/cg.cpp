#include "chunk.h"
#include "cuknl_shared.h"

__global__ void cg_init_u(const int x, const int y, const int coefficient, const double *density, const double *energy1, double *u,
                          double *p, double *r, double *w) {
  const int gid = threadIdx.x + blockIdx.x * blockDim.x;
  if (gid >= x * y) return;

  p[gid] = 0.0;
  r[gid] = 0.0;
  u[gid] = energy1[gid] * density[gid];

  w[gid] = (coefficient == CONDUCTIVITY) ? density[gid] : 1.0 / density[gid];
}

__global__ void cg_init_k(const int x_inner, const int y_inner, const int halo_depth, const double *w, double *kx, double *ky, double rx,
                          double ry) {
  const int gid = threadIdx.x + blockIdx.x * blockDim.x;
  if (gid >= x_inner * y_inner) return;

  const int x = x_inner + 2 * halo_depth - 1;
  const int col = gid % x_inner;
  const int row = gid / x_inner;
  const int off0 = halo_depth * (x + 1);
  const int index = off0 + col + row * x;

  kx[index] = rx * (w[index - 1] + w[index]) / (2.0 * w[index - 1] * w[index]);
  ky[index] = ry * (w[index - x] + w[index]) / (2.0 * w[index - x] * w[index]);
}

__global__ void cg_init_others(const int x_inner, const int y_inner, const int halo_depth, const double *u, const double *kx,
                               const double *ky, double *p, double *r, double *w, double *mi, double *rro) {
  const int gid = threadIdx.x + blockIdx.x * blockDim.x;
  __shared__ double rro_shared[BLOCK_SIZE];
  rro_shared[threadIdx.x] = 0.0;

  if (gid < x_inner * y_inner) {
    const int x = x_inner + 2 * halo_depth;
    const int col = gid % x_inner;
    const int row = gid / x_inner;
    const int off0 = halo_depth * (x + 1);
    const int index = off0 + col + row * x;

    const double smvp = tealeaf_SMVP(u);

    w[index] = smvp;
    r[index] = u[index] - w[index];
    p[index] = r[index];
    rro_shared[threadIdx.x] = r[index] * r[index];
  }

  reduce<double, BLOCK_SIZE / 2>::run(rro_shared, rro, SUM);
}

__global__ void cg_calc_w(const int x_inner, const int y_inner, const int halo_depth, const double *kx, const double *ky, const double *p,
                          double *w, double *pw) {
  const int gid = threadIdx.x + blockIdx.x * blockDim.x;
  __shared__ double pw_shared[BLOCK_SIZE];
  pw_shared[threadIdx.x] = 0.0;

  if (gid < x_inner * y_inner) {
    const int x = x_inner + 2 * halo_depth;
    const int col = gid % x_inner;
    const int row = gid / x_inner;
    const int off0 = halo_depth * (x + 1);
    const int index = off0 + col + row * x;

    const double smvp = tealeaf_SMVP(p);
    w[index] = smvp;
    pw_shared[threadIdx.x] = w[index] * p[index];
  }

  reduce<double, BLOCK_SIZE / 2>::run(pw_shared, pw, SUM);
}

__global__ void cg_calc_ur(const int x_inner, const int y_inner, const int halo_depth, const double alpha, const double *p, const double *w,
                           double *u, double *r, double *rrn) {
  const int gid = threadIdx.x + blockIdx.x * blockDim.x;
  __shared__ double rrn_shared[BLOCK_SIZE];
  rrn_shared[threadIdx.x] = 0.0;

  if (gid < x_inner * y_inner) {
    const int x = x_inner + 2 * halo_depth;
    const int col = gid % x_inner;
    const int row = gid / x_inner;
    const int off0 = halo_depth * (x + 1);
    const int index = off0 + col + row * x;

    u[index] += alpha * p[index];
    r[index] -= alpha * w[index];
    rrn_shared[threadIdx.x] = r[index] * r[index];
  }

  reduce<double, BLOCK_SIZE / 2>::run(rrn_shared, rrn, SUM);
}

__global__ void cg_calc_p(const int x_inner, const int y_inner, const int halo_depth, const double beta, const double *r, double *p) {
  const int gid = threadIdx.x + blockIdx.x * blockDim.x;
  if (gid >= x_inner * y_inner) return;

  const int x = x_inner + 2 * halo_depth;
  const int col = gid % x_inner;
  const int row = gid / x_inner;
  const int off0 = halo_depth * (x + 1);
  const int index = off0 + col + row * x;

  p[index] = r[index] + beta * p[index];
}

// CG solver kernels
void run_cg_init(Chunk *chunk, Settings &settings, double rx, double ry, double *rro) {
  START_PROFILING(settings.kernel_profile);

  int num_blocks = ceil((double)(chunk->x * chunk->y) / (double)BLOCK_SIZE);
  cg_init_u<<<num_blocks, BLOCK_SIZE>>>(chunk->x, chunk->y, settings.coefficient, chunk->density, chunk->energy, chunk->u, chunk->p,
                                        chunk->r, chunk->w);

  int x_inner = chunk->x - (2 * settings.halo_depth - 1);
  int y_inner = chunk->y - (2 * settings.halo_depth - 1);
  num_blocks = ceil((double)(x_inner * y_inner) / (double)BLOCK_SIZE);

  cg_init_k<<<num_blocks, BLOCK_SIZE>>>(x_inner, y_inner, settings.halo_depth, chunk->w, chunk->kx, chunk->ky, rx, ry);

  x_inner = chunk->x - 2 * settings.halo_depth;
  y_inner = chunk->y - 2 * settings.halo_depth;
  num_blocks = ceil((double)(x_inner * y_inner) / (double)BLOCK_SIZE);

  cg_init_others<<<num_blocks, BLOCK_SIZE>>>(x_inner, y_inner, settings.halo_depth, chunk->u, chunk->kx, chunk->ky, chunk->p, chunk->r,
                                             chunk->w, chunk->mi, chunk->ext->d_reduce_buffer);

  sum_reduce_buffer(chunk->ext->d_reduce_buffer, rro, num_blocks);

  KERNELS_END();
}

void run_cg_calc_w(Chunk *chunk, Settings &settings, double *pw) {
  KERNELS_START(2 * settings.halo_depth);

  cg_calc_w<<<num_blocks, BLOCK_SIZE>>>(x_inner, y_inner, settings.halo_depth, chunk->kx, chunk->ky, chunk->p, chunk->w,
                                        chunk->ext->d_reduce_buffer);

  sum_reduce_buffer(chunk->ext->d_reduce_buffer, pw, num_blocks);

  KERNELS_END();
}

void run_cg_calc_ur(Chunk *chunk, Settings &settings, double alpha, double *rrn) {
  KERNELS_START(2 * settings.halo_depth);

  cg_calc_ur<<<num_blocks, BLOCK_SIZE>>>(x_inner, y_inner, settings.halo_depth, alpha, chunk->p, chunk->w, chunk->u, chunk->r,
                                         chunk->ext->d_reduce_buffer);

  sum_reduce_buffer(chunk->ext->d_reduce_buffer, rrn, num_blocks);

  KERNELS_END();
}

void run_cg_calc_p(Chunk *chunk, Settings &settings, double beta) {
  KERNELS_START(2 * settings.halo_depth);

  cg_calc_p<<<num_blocks, BLOCK_SIZE>>>(x_inner, y_inner, settings.halo_depth, beta, chunk->r, chunk->p);

  KERNELS_END();
}
