#pragma once

using FieldBufferType = double *;
using StagingBufferType = double *;

struct ChunkExtension {
  double *d_reduce_buffer;
  double *d_reduce_buffer2;
  double *d_reduce_buffer3;
  double *d_reduce_buffer4;
};
