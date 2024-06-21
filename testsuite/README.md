# Test Suite

## Test Naming

General: MPI call,Synchronization Calls,Stream Semantics,Memory Allocation,CUDA Data access,MPI data access,Has Datarace (NOK)

#### Synchronization (Explicit)
- ds (DeviceSynchronize)
- ss (StreamSynchronize)
- es (EventSynchronize)
- sq (StreamQuery)
- eq (EventQuery)

#### Stream Semantics
- def (Default stream)
- def_user (Default stream and user defined stream)
- user (User defined stream)*for stopm