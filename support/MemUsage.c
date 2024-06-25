#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/resource.h>

#ifndef PMPIPREFIX
#define PMPIPREFIX PMPI
#endif

#define PMPIZE_T(f, p) p##f
#define PMPIZE_H(f, p) PMPIZE_T(f, p)
#define PMPIZE(f) PMPIZE_H(f, PMPIPREFIX)

int PMPIZE(_Finalize)();
int MPI_Finalize() {
  int rank, all = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (all) {
    struct rusage end;
    getrusage(RUSAGE_SELF, &end);
    printf("%i: MAX RSS[KB] during execution: %ld\n", rank, end.ru_maxrss);
  }
  return PMPIZE(_Finalize)();
}
