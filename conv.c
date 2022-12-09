// conv.c
// Name: Tanay Agarwal, Nirmal Krishnan
// JHED: tagarwa2, nkrishn9

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <papi.h>
#include <mpi.h>
#include <pthread.h>
#include <immintrin.h>
#include "./include/convutils.h"

#define DEFAULT_ITERATIONS 1
#define DEFAULT_THREADS 2
#define GRID_FILE_PATH "./io-files/grids/haring.bin"
#define KERNEL_FILE_PATH "./io-files/kernels/gblur.bin"
#define VEC_SIZE 4
#define DEBUG 1                                       // True to save result in textual and binary mode
#define WORKER_FACTOR 1                               // Used to calc how many rows must be computed before polling the neighbours
#define MEAN_VALUE 0                                  // Temporary solution (due to normalization)
#define REQS_OFFSET(POS) ((POS) * SIM_REQS / 2)       // Used to calc index offset in MPI requests array
#define RECV_OFFSET(POS) (REQS_OFFSET(POS) + 2)       // Used to calc MPI_Irecv index in MPI requests array

void *worker_thread(void*);
void *grid_io_thread(void*);
void thread_polling(enum POSITION, uint, struct worker_data*, long_long**);
void remote_polling(enum POSITION, uint, struct worker_data*, long_long**);
void load_balancing(int, uint8_t, struct worker_data* worker, long_long**);
void load_balancing_custom(uint, uint*, long_long**);
void conv_subgrid(float*, float*, int, int);
void get_process_additional_row(struct proc_info*);
void initialize_thread_coordinates(struct thread_handler*);

float *kernel;                        // Kernel used for convolution
float *new_grid;                      // Input/Result grid, swapped at every iteration
float *old_grid;                      // Input/Result grid, swapped at every iteration
float kern_dot_sum;                   // Used for normalization, its value is equal to: sum(dot(kernel, kernel))
uint8_t affinity;                     // If thread affinity (cpu pinning) should be set
uint32_t kern_width;                  // Number of elements in one kernel matrix row
uint32_t grid_width;                  // Number of elements in one grid matrix row
uint pad_nrows;                       // Number of rows that should be shared with other processes
uint pad_elems;                       // Number of elements in the pad section of the grid matrix
uint rbf_elems;                       // Number of elements in Rows to compute Before Polling the neighbours
uint kern_elems;                      // Number of elements in whole kernel matrix
uint grid_elems;                      // Number of elements in whole grid matrix
uint proc_nrows;                      // Number of rows assigned to a process
uint workers_nrows;                   // Number of rows to distribute to worker threads except the ones having mpi dependencies and load_balancer
uint nrows_per_thread;                // Number of rows assigned to a worker having no MPI dependencies 
uint bordering_thread_nrows;          // Number of rows assigned to a worker having MPI dependencies (those threads are offloaded)
uint max_lb_amount;                   // Number of additional rows that a woker with MPI dependencies should do in lb set of rows 
uint8_t num_bordering_threads;        // Number of threads having MPI dependencies
int num_procs;                        // Number of MPI processes in the communicator
int num_threads;                      // Number of threads (main included) for every MPI process
int num_iterations;                   // Number of convolution iterations
int rank;                             // MPI process identifier
int reqs_completed[SIM_REQS];         // Log of the completed mpi requests
MPI_Request requests[SIM_REQS];       // There are at most two "Isend" and one "Irecv" not completed at the same time per worker_thread, hence six per process
pthread_mutex_t mutex_mpi;            // To call MPI routines in mutual exclusion (will be used only by top and bottom thread)
struct io_thread_args io_args;        // Used by io, worker, and main threads to synchronize about grid read
struct load_balancer lb;              // Structure containing all synchronization variables used during load balancing 
long_long** thread_measures;          // To collect thread measures
__m128i last_mask;                    // Used by PSIMD instructions to discard last elements in contiguous load

int main(int argc, char** argv) {
  int provided;                       // MPI thread level supported
  int rc;                             // Return code used in error handling
  long_long time_start, time_stop;    // To measure execution time
  FILE *fp_grid = NULL;               // I/O files for grid and kernel matrices
  FILE *fp_kernel = NULL;

  // Fetch from arguments how many convolution iterations do and the number of threads
  num_iterations = (argc > 1) ? atoi(argv[1]) : DEFAULT_ITERATIONS;
  num_threads = (argc > 2) ? atoi(argv[2]) : DEFAULT_THREADS;
  affinity = (argc > 3) ? atoi(argv[3]) : 1;
  if(num_iterations < DEFAULT_ITERATIONS) {
    fprintf(stderr, "Invalid number of convolution iterations (first argument), value inserted: %d\n", num_iterations);
    exit(-1);
  }
  if(num_threads < DEFAULT_THREADS) {
    fprintf(stderr, "Invalid number of threads (second argument), value inserted: %d\n", num_threads);
    exit(-1);
  }

  // PAPI setup
  if((rc = PAPI_library_init(PAPI_VER_CURRENT)) != PAPI_VER_CURRENT)
    handle_PAPI_error(rc, "Error in library init.");

  // MPI setup
  if((rc = MPI_Init_thread(&argc, &argv, MPI_THREAD_SERIALIZED, &provided)) != MPI_SUCCESS) {
    fprintf(stderr, "MPI_Init error. Return code: %d\n", rc);
    exit(-1);
  } 
  if(provided < MPI_THREAD_SERIALIZED) {
    fprintf(stderr, "Minimum MPI threading level requested: %d (provided: %d)\n", MPI_THREAD_SERIALIZED, provided);
    exit(-1);
  }
  MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  time_start = PAPI_get_real_usec();

  struct proc_info procs_info[(rank) ? 1 : num_procs];        // Which process must compute an additional row
  MPI_Request info_reqs[num_procs-1];                         // Used by rank 0 to ditribute initial informations (matrices size)
  MPI_Request kern_reqs[num_procs-1];                         // Used by rank 0 to ditribute kernel matrix
  MPI_Request grid_reqs[num_procs-1];                         // Used by rank 0 to ditribute grid matrix
  pthread_t threads[num_threads-1];                           // "-1" because the main thread will be reused as worker thread
  pthread_t io_thread;                                        // Used to read grid from disk while main thread works on something else
  io_args.requests = grid_reqs;
  memset(io_args.flags, 0, sizeof(uint8_t) * (SEND_INFO+1));
  pthread_mutex_init(&(io_args.mutex), NULL);                 // Used by io_thread and main thread to synchronize
  pthread_cond_init(&(io_args.cond), NULL);

  if(!rank) {
    // Opening input files
    if((fp_grid = fopen(GRID_FILE_PATH, "rb")) == NULL) {
      perror("Error while opening grid file:");
      exit(-1);
    }
    if((fp_kernel = fopen(KERNEL_FILE_PATH, "rb")) == NULL) {
      perror("Error while opening kernel file:");
      exit(-1);
    }

    // First token represent matrix dimension
    if(fread(&grid_width, sizeof(uint32_t), 1, fp_grid) != 1 || fread(&kern_width, sizeof(uint32_t), 1, fp_kernel) != 1) {
      fprintf(stderr, "Error in file reading: first element should be the row (or column) length of a square matrix\n");
      exit(-1);
    }

    // Exchange initial information 
    if(num_procs > 1) {
      const uint32_t to_send[] = {grid_width, kern_width};
      for (int p = 1; p < num_procs; p++)
        MPI_Isend(to_send, 2, MPI_UINT32_T, p, p, MPI_COMM_WORLD, &info_reqs[p-1]);
    }

    // Read and exchange kernel
    kern_elems = kern_width * kern_width;
    kernel = malloc(sizeof(float) * kern_elems);
    read_float_matrix(fp_kernel, kernel, kern_elems);

    if(num_procs > 1) {
      for (int p = 1; p < num_procs; p++)
        MPI_Isend(kernel, kern_elems, MPI_FLOAT, p, p, MPI_COMM_WORLD, &kern_reqs[p-1]);
    }

    // Start grid read. Rank 0 has the whole file in memory, other ranks have only the part they are interested in
    grid_elems = grid_width * grid_width;
    pad_nrows = (kern_width - 1) / 2;
    pad_elems = grid_width * pad_nrows;
    io_args.fp_grid = fp_grid;
    io_args.procs_info = procs_info;
    old_grid = malloc(sizeof(float) * (grid_elems + pad_elems*2));
    pthread_mutex_init(&mutex_mpi, NULL);

    if ((rc = pthread_create(&io_thread, NULL, grid_io_thread, NULL))) { 
      fprintf(stderr, "Error while creating pthread 'io_thread'; Return code: %d\n", rc);
      exit(-1);
    }
  } else {
    uint32_t to_recv[2];
    MPI_Recv(to_recv, 2, MPI_UINT32_T, 0, rank, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    kern_width = to_recv[1];
    kern_elems = kern_width * kern_width;
    kernel = malloc(sizeof(float) * kern_elems);
    MPI_Irecv(kernel, kern_elems, MPI_FLOAT, 0, rank, MPI_COMM_WORLD, &(io_args.requests[0]));

    grid_width = to_recv[0];
    grid_elems = grid_width * grid_width;
    pad_nrows = (kern_width - 1) / 2;
    pad_elems = grid_width * pad_nrows;
  }

  // Added a synchronization point, otherwise the 3rd node will wait too much before receiving the dimension informations 
  if(!rank && num_procs > 1) {
    MPI_Status statuses[num_procs-1];
    pthread_mutex_lock(&mutex_mpi);
    MPI_Waitall(num_procs-1, info_reqs, statuses);
    pthread_mutex_unlock(&mutex_mpi);
  }

  // Checking if input is big enough for work division between threads and load balancing 
  if(grid_width/(num_procs * num_threads + num_procs) < pad_nrows * 3) {
    fprintf(stderr, "Threading is oversized compared to input matrix. Threads: %dx%d | Input number of rows: %d\n", num_procs, num_threads, grid_width);
    exit(-1);
  }

  rbf_elems = pad_elems + grid_width * WORKER_FACTOR;                 // Rows before polling number of elements

  // Rank 0 get info about which processes must compute an additional row, other ranks get info only about themself
  get_process_additional_row(procs_info); 
  const uint min_nrows = (grid_width / num_procs);                    // Minimum amout of rows distributed to each process
  proc_nrows = min_nrows + procs_info[0].has_additional_row;          // Number of rows assigned to current process
  const uint proc_elems = proc_nrows * grid_width;                    // Number of elements assigned to a process

  if(!rank) {
    new_grid = malloc(sizeof(float) * (grid_elems + pad_elems*2));
  } else {
    new_grid = malloc(sizeof(float) * (proc_elems + pad_elems*2));
    old_grid = malloc(sizeof(float) * (proc_elems + pad_elems*2));    // Ranks different from 0 has a smaller grid to alloc
  }

  // Rank 0 prepares send/recv info (synch point with main thread), other ranks receives grid data. 
  if(!rank){
    io_args.procs_info[0].sstart = 0;
    io_args.procs_info[0].ssize = (proc_nrows + pad_nrows*2) * grid_width;
    if(num_procs > 1) {
      // Info about data scattering. Pads are included (to avoid an MPI exchange in the first iteration)
      int offset = io_args.procs_info[0].has_additional_row;
      for(int i = 1; i < num_procs; i++) {
        io_args.procs_info[i].sstart = (min_nrows * i + offset) * grid_width;
        io_args.procs_info[i].ssize = (min_nrows + pad_nrows*2 + io_args.procs_info[i].has_additional_row) * grid_width;
        if(i == num_procs-1) io_args.procs_info[i].ssize -= pad_elems;
        offset += io_args.procs_info[i].has_additional_row;
      }
      pthread_mutex_lock(&(io_args.mutex));
      io_args.flags[SEND_INFO] = 1;
      pthread_cond_signal(&(io_args.cond));
      pthread_mutex_unlock(&(io_args.mutex));
      
      // Info about result gathering from previous scattering data. Pads are excluded
      for(int p = 1; p < num_procs; p++) {
        io_args.procs_info[p].gstart = io_args.procs_info[p].sstart + pad_elems;
        io_args.procs_info[p].gsize = io_args.procs_info[p].ssize - pad_elems*2;
        if(p == num_procs-1) io_args.procs_info[p].gsize += pad_elems;
      }
    }
  } else {
    // Receive grid data
    uint recv_size = proc_elems + pad_elems * 2;
    if(rank == num_procs-1) recv_size -= pad_elems;
    MPI_Irecv(old_grid, recv_size, MPI_FLOAT, 0, rank, MPI_COMM_WORLD, &(io_args.requests[1]));
  }

  // Set a zero-pad for lowest and highest process 
  if(!rank){
    memset(new_grid, 0, pad_elems * sizeof(float));
    memset(old_grid, 0, pad_elems * sizeof(float));
  }
  if(rank == num_procs-1) {
    memset(&new_grid[proc_elems + pad_elems], 0, pad_elems * sizeof(float));
    memset(&old_grid[proc_elems + pad_elems], 0, pad_elems * sizeof(float));
  }

  for(int i = 0; i < SIM_REQS; i++) requests[i] = MPI_REQUEST_NULL;
  memset(reqs_completed, 0, sizeof(int) * SIM_REQS);
  
  // Computation of "last_mask"
  uint32_t rem = kern_width % VEC_SIZE;
  uint32_t to_load[VEC_SIZE];
  memset(to_load, 0, VEC_SIZE * sizeof(uint32_t));
  for(int i = 0; i < rem; i++) to_load[i] = UINT32_MAX;        // UINT32_MAX = -1
  last_mask = _mm_loadu_si128((__m128i*) to_load);

  // PThreads arguments initialization 
  struct thread_handler* handlers = calloc(num_threads, sizeof(struct thread_handler));
  for(int i = 0; i < num_threads; i++) {
    handlers[i].tid = i;
    handlers[i].neighbour[TOP] = (i > 0) ? &handlers[i-1] : NULL;
    handlers[i].neighbour[BOTTOM] = (i < num_threads-1) ? &handlers[i+1] : NULL;
    pthread_mutex_init(&handlers[i].mutex, NULL);
    pthread_cond_init(&handlers[i].pad_ready, NULL);
  }

  // Initializing load balancer (not an active thread) and its neighbours
  memset(&lb, 0, sizeof(struct load_balancer));
  lb.handler = &(struct thread_handler){0};
  lb.handler->tid = -1;
  lb.handler->neighbour[TOP] = &handlers[num_threads/2-1];       // Connections with other threads
  lb.handler->neighbour[BOTTOM] = &handlers[num_threads/2];
  handlers[num_threads/2-1].neighbour[BOTTOM] = lb.handler;
  handlers[num_threads/2].neighbour[TOP] = lb.handler;

  initialize_thread_coordinates(lb.handler);
  lb.curr_start = lb.handler->start + pad_elems;                 // Start from subpart having no dependencies
  pthread_mutex_init(&lb.mutex, NULL);
  pthread_mutex_init(&(lb.handler->mutex), NULL);
  pthread_cond_init(&lb.iter_completed, NULL);
  pthread_cond_init(&(lb.handler->pad_ready), NULL);
  thread_measures = malloc(sizeof(long_long*) * num_threads);

  // PThreads creation
  for(int i = 0; i < num_threads-1; i++) {
    rc = pthread_create(&threads[i], NULL, worker_thread, (void*)&handlers[i]);
    if (rc) { 
      fprintf(stderr, "Error while creating pthread 'worker_thread[%d]'; Return code: %d\n", i, rc);
      exit(-1);
    }
  }
  worker_thread((void*) &handlers[num_threads-1]);   // Main thread is reused as bottom thread

  // Check if 'io_thread' has exited and start recv listeners to gather results
  MPI_Request res_gather_reqs[num_procs-1];
  float *res_grid = (num_iterations % 2) ? new_grid : old_grid;
  if(!rank) {
    if(num_procs > 1) {
      for(int p = 1; p < num_procs; p++) {
        MPI_Irecv(&res_grid[procs_info[p].gstart], procs_info[p].gsize, MPI_FLOAT, p, p, MPI_COMM_WORLD, &res_gather_reqs[p-1]);
      }
    }

    if(pthread_join(io_thread, (void*) &rc)) 
      fprintf(stderr, "Join error, io_thread exited with: %d", rc);
  }

  // Wait workers termination
  for(int i = 0; i < num_threads-1; i++) {
    if(pthread_join(threads[i], (void*) &rc)) 
      fprintf(stderr, "Join error, worker_thread[%d] exited with: %d", i, rc);
  }

  if(rank) {
    MPI_Send(&res_grid[pad_elems], proc_elems, MPI_FLOAT, 0, rank, MPI_COMM_WORLD);
  } else if(num_procs > 1) {
    MPI_Status statuses[num_procs-1];
    MPI_Waitall(num_procs-1, res_gather_reqs, statuses);
  }
  
  // Print measures
  time_stop = PAPI_get_real_usec();
  printf("Rank[%d] | Elapsed time: %lld us\n", rank, (time_stop - time_start));

  long_long* curr_meas;
  for(int i = 0; i < num_threads; i++) {
    curr_meas = thread_measures[i];
    printf("Thread[%d][%d]: Elapsed: %llu us | Condition WT: %llu us | Handlers mutex WT: %llu us | MPI mutex WT: %llu us | LB mutex WT: %llu us | Total L2 cache misses: %lld\n", 
      rank, i, curr_meas[0], curr_meas[1], curr_meas[2], curr_meas[3], curr_meas[4], curr_meas[5]);
    free(curr_meas);
  }

  // Store computed matrix for debug purposes. Rank 0 has the whole matrix file in memory, other ranks have only the part they are interested in 
  if (DEBUG && !rank) {
    save_bin(res_grid);
    save_txt(res_grid);
  }

  // Destroy pthread objects and free all used resources
  pthread_mutex_destroy(&mutex_mpi);
  pthread_mutex_destroy(&lb.mutex);
  pthread_mutex_destroy(&io_args.mutex);
  pthread_cond_destroy(&io_args.cond);
  pthread_cond_destroy(&lb.iter_completed);
  for(int i = 0; i < num_threads; i++) {
    pthread_mutex_destroy(&handlers[i].mutex);
    pthread_cond_destroy(&handlers[i].pad_ready);
  }

  MPI_Finalize();
  if(!rank) {
    fclose(fp_kernel);
    fclose(fp_grid);
  }
  free(thread_measures);
  free(handlers);
  free(new_grid);
  free(old_grid);
  free(kernel);
  exit(0);
}

/* This is executed only by rank 0 */
void* grid_io_thread(void* args) {
  read_float_matrix(io_args.fp_grid, &old_grid[pad_elems], grid_elems);

  // Check if scattering info are ready
  if(num_procs > 1) {
    pthread_mutex_lock(&(io_args.mutex));
    while(!io_args.flags[SEND_INFO]) 
      pthread_cond_wait(&(io_args.cond), &(io_args.mutex));
    pthread_mutex_unlock(&(io_args.mutex));

    pthread_mutex_lock(&mutex_mpi);
    for(int p = 1; p < num_procs; p++)
      MPI_Isend(&old_grid[io_args.procs_info[p].sstart], io_args.procs_info[p].ssize, MPI_FLOAT, p, p, MPI_COMM_WORLD, &io_args.requests[p-1]);
    pthread_mutex_unlock(&mutex_mpi);
  }

  // Signal that grid read have been completed
  pthread_mutex_lock(&(io_args.mutex));
  io_args.flags[GRID] = 1;
  pthread_cond_broadcast(&(io_args.cond));
  pthread_mutex_unlock(&(io_args.mutex));

  pthread_exit(0);
}

void* worker_thread(void* args) {
  struct thread_handler *handler = (struct thread_handler*)args;
  float *my_old_grid = old_grid;
  float *my_new_grid = new_grid;
  float *temp;                                      // Used only for grid swap
  int center_start;                                 // Center elements are completed a bunch of rows at a time
  uint tot_rows_computed;                           // How many rows of load balancer set of rows have been computed by this thread      
  uint8_t changes;                                  // To track if something changed during polling
  const uint8_t mpi_needed[2] = {                   // If remote polling is necessary
    (num_procs > 1) && rank && !handler->tid,
    (num_procs > 1) && (rank < num_procs-1) && (handler->tid == num_threads-1)
  };

  // PAPI setup
  int rc, event_set = PAPI_NULL;
  if((rc = PAPI_thread_init(pthread_self)) != PAPI_OK)
    handle_PAPI_error(rc, "Error in PAPI thread init.");
  if((rc = PAPI_create_eventset(&event_set)) != PAPI_OK)
    handle_PAPI_error(rc, "Error while creating the PAPI eventset.");
  if((rc = PAPI_add_event(event_set, PAPI_L2_TCM)) != PAPI_OK)
    handle_PAPI_error(rc, "Error while adding L2 total cache miss event.");
  if((rc = PAPI_start(event_set)) != PAPI_OK) 
    handle_PAPI_error(rc, "Error in PAPI_start().");

  long_long t, cond_wait_time = 0, handler_mutex_wait_time = 0, mpi_mutex_wait_time = 0, lb_mutex_wait_time = 0;
  long_long* measures[4] = {&cond_wait_time, &handler_mutex_wait_time, &mpi_mutex_wait_time, &lb_mutex_wait_time};
  long_long* meas_to_return = malloc(sizeof(long_long) * 6);
  long_long time_start = PAPI_get_real_usec();

  // Thread setup
  if(affinity && stick_this_thread_to_core(handler->tid)) {
    fprintf(stderr, "Error occurred while setting thread affinity on core: %d\n", handler->tid);
    exit(-1);
  }
  initialize_thread_coordinates(handler);

  struct worker_data *worker = &(struct worker_data){0};
  worker->self = handler;
  worker->mpi = &(struct mpi_data){0};
  if(mpi_needed[TOP] || mpi_needed[BOTTOM]) {
    if(handler->tid == 0) {
      worker->mpi->send_position = handler->start;
      worker->mpi->recv_position = 0;
      worker->mpi->neighbour = rank - 1;
    } else {
      worker->mpi->send_position = (handler->end - pad_elems);
      worker->mpi->recv_position = handler->end;
      worker->mpi->neighbour = rank + 1;
    }
  } else worker->mpi = NULL;

  // Complete kernel receive and/or compute "sum(dot(kernel, kernel))"
  if(!handler->tid) {
    if(rank) MPI_Wait(&(io_args.requests[0]), MPI_STATUS_IGNORE);
    for(int pos = 0; pos < kern_elems; pos++) {
      kern_dot_sum += kernel[pos] * kernel[pos];
    }
    pthread_mutex_lock(&(io_args.mutex));
    io_args.flags[KERNEL] = 1;
    pthread_cond_broadcast(&(io_args.cond));
    pthread_mutex_unlock(&(io_args.mutex));
  }

  // Synchronization point, check if grid data is ready and start convolution
  if(!handler->tid && rank) {
    MPI_Wait(&(io_args.requests[1]), MPI_STATUS_IGNORE);
    pthread_mutex_lock(&(io_args.mutex));
    io_args.flags[GRID] = 1;
    pthread_cond_broadcast(&(io_args.cond));
    pthread_mutex_unlock(&(io_args.mutex));
  } else {
    pthread_mutex_lock(&(io_args.mutex));
    while(!io_args.flags[GRID] || !io_args.flags[KERNEL])
      pthread_cond_wait(&(io_args.cond), &(io_args.mutex));
    pthread_mutex_unlock(&(io_args.mutex));
  }

  // First convolution iteration (starting with top and bottom rows)
  conv_subgrid(my_old_grid, my_new_grid, handler->start, (handler->start + pad_elems));
  conv_subgrid(my_old_grid, my_new_grid, (handler->end - pad_elems), handler->end);

  t = PAPI_get_real_usec();
  pthread_mutex_lock(&(handler->mutex));
  handler_mutex_wait_time += PAPI_get_real_usec() - t;
  handler->rows_done[TOP][0] = 1;
  handler->rows_done[BOTTOM][0] = 1;
  pthread_cond_broadcast(&(handler->pad_ready));
  pthread_mutex_unlock(&(handler->mutex));

  // Send top or bottom rows
  if(mpi_needed[TOP] || mpi_needed[BOTTOM]) {
    const uint8_t offset = (!handler->tid) ? REQS_OFFSET(TOP) : REQS_OFFSET(BOTTOM);
    t = PAPI_get_real_usec();
    pthread_mutex_lock(&mutex_mpi);
    mpi_mutex_wait_time += PAPI_get_real_usec() - t;
    MPI_Isend(&my_new_grid[worker->mpi->send_position], pad_elems, MPI_FLOAT, worker->mpi->neighbour, 0, MPI_COMM_WORLD, &requests[0 + offset]);
    MPI_Irecv(&my_new_grid[worker->mpi->recv_position], pad_elems, MPI_FLOAT, worker->mpi->neighbour, 0, MPI_COMM_WORLD, &requests[2 + offset]);
    pthread_mutex_unlock(&mutex_mpi);
  }

  // Complete the first convolution iteration by computing central elements
  conv_subgrid(my_old_grid, my_new_grid, (handler->start + pad_elems), (handler->end - pad_elems));
  if(!lb.iter) load_balancing(0, 0, NULL, measures);

  // Second or higher convolution iterations
  int* completed = worker->completed;
  for(uint iter = 1; iter < num_iterations; iter++) {
    temp = my_old_grid;
    my_old_grid = my_new_grid;
    my_new_grid = temp;
    center_start = handler->start + pad_elems;
    tot_rows_computed = 0;

    while(!completed[TOP] || !completed[BOTTOM] || !completed[CENTER]) {
      changes = 0;

      if(!completed[TOP]) {
        if(mpi_needed[TOP]) remote_polling(TOP, iter, worker, measures);
        else thread_polling(TOP, iter, worker, measures);
        changes |= completed[TOP];
      }

      if(!completed[BOTTOM]) {
        if(mpi_needed[BOTTOM]) remote_polling(BOTTOM, iter, worker, measures);
        else thread_polling(BOTTOM, iter, worker, measures);
        changes |= completed[BOTTOM];
      }

      // Computing a bunch of central rows if top and bottom rows are incomplete
      if(!completed[CENTER]) {
        if(mpi_needed[TOP] || mpi_needed[BOTTOM]) {
          if(center_start != handler->end - pad_elems) {
            // Threads having MPI dependencies have a small amount of CENTER rows
            conv_subgrid(my_old_grid, my_new_grid, center_start, handler->end - pad_elems);
            center_start = handler->end - pad_elems;
          } else if (iter < lb.iter) { 
            // Load balancer set of rows have been already completed (this thread is late)
            completed[CENTER] = 1;
            continue;
          } else {
            // Compute a specic amount of load balancer rows
            uint rows_to_compute = rbf_elems/grid_width;
            load_balancing_custom(iter, &rows_to_compute, measures);          
            tot_rows_computed += rows_to_compute;
            if(!rows_to_compute || tot_rows_computed >= max_lb_amount || (completed[TOP] && completed[BOTTOM])) 
              completed[CENTER] = 1;
            continue;
          }
        } else {
          // Threads having no MPI dependencies are in this branch
          uint center_end;
          uint actual_end = handler->end - pad_elems;
          if (completed[TOP] && completed[BOTTOM]) {
            center_end = actual_end;
            completed[CENTER] = 1;
          } else {
            center_end = center_start + rbf_elems;
            if(center_end > actual_end) center_end = actual_end;
          }

          conv_subgrid(my_old_grid, my_new_grid, center_start, center_end);
          if(center_end == actual_end) completed[CENTER] = 1;
          else center_start += rbf_elems;
        }
        changes = 1;
      }

      if(!changes) load_balancing(iter, 1, NULL, measures);
    }

    // During load balancing
    memset(completed, 0, sizeof(int) * 3);

    // Load balancing if this thread ended current iteration earlier
    if(iter >= lb.iter) load_balancing(iter, 0, worker, measures);
  }

  // Retrieving execution info
  long_long num_cache_miss;
  if ((rc = PAPI_stop(event_set, &num_cache_miss)) != PAPI_OK)
    handle_PAPI_error(rc, "Error in PAPI_stop().");  
 
  meas_to_return[1] = cond_wait_time; 
  meas_to_return[2] = handler_mutex_wait_time; 
  meas_to_return[3] = mpi_mutex_wait_time;
  meas_to_return[4] = lb_mutex_wait_time;
  meas_to_return[5] = num_cache_miss;
  meas_to_return[0] = (PAPI_get_real_usec() - time_start);
  thread_measures[handler->tid] = meas_to_return;

  if(handler->tid != num_threads-1) pthread_exit(0);
  return 0;
}

/* Threads that ended their iteration earlier will compute a shared portion of the matrix */
void load_balancing(int iter, uint8_t single_work, struct worker_data* worker, long_long** meas){
  const uint8_t iter_even = (!iter) ? 1 : !(iter % 2);               // If current iteration is even or odd
  uint start = 0, end = 0;
  long_long t;
  long_long *handlers_mutex_wait_time = meas[1];
  long_long *lb_mutex_wait_time = meas[3];

  float *my_new_grid, *my_old_grid;
  if(!iter_even) {
    my_new_grid = old_grid;
    my_old_grid = new_grid;
  } else {
    my_new_grid = new_grid;
    my_old_grid = old_grid;
  }

  // If there are MPI dependencies, make a node polling every once in a while  
  enum POSITION node_pos = 0;
  int count = -1;
  if(!single_work && worker != NULL && worker->mpi != NULL) {
    count = 0;
    node_pos = (!worker->mpi->recv_position) ? TOP: BOTTOM;
  }

  // Compute a row of the load_balancer shared work 
  while(iter >= lb.iter) {
    // If there are MPI dependencies, MPI messages could be ready while computing load balancing set of rows
    if(!count && !worker->completed[node_pos]) {
      count = pad_nrows+1;
      remote_polling(node_pos, iter+1, worker, meas);
    } else count--;

    start = 0;
    t = PAPI_get_real_usec();
    pthread_mutex_lock(&lb.mutex);
    *lb_mutex_wait_time += PAPI_get_real_usec() - t;
    while(iter > lb.iter)
      pthread_cond_wait(&lb.iter_completed, &lb.mutex);         // Wait if lb work of previous iteration is not completed yet
    if(iter == lb.iter) {
      start = lb.curr_start;
      lb.curr_start += grid_width;
    }
    pthread_mutex_unlock(&lb.mutex);
    end = start + grid_width;
    if(!start || end > lb.handler->end + pad_elems) return;  // All shared works have been reserved, return to private work

    // If my shared work is in the middle, no dependencies
    if(start >= (lb.handler->start + pad_elems) && end <= (lb.handler->end - pad_elems)) {
      conv_subgrid(my_old_grid, my_new_grid, start, end);
      t = PAPI_get_real_usec();
      pthread_mutex_lock(&lb.mutex);
      *lb_mutex_wait_time += PAPI_get_real_usec() - t;
      lb.rows_completed++;                                   // Track the already computed row
      if(lb.rows_completed == lb.nrows) {                    // All shared works have been completed
        lb.rows_completed = 0;
        lb.curr_start = lb.handler->start + pad_elems;
        lb.iter++;
        pthread_cond_broadcast(&lb.iter_completed);
      }
      pthread_mutex_unlock(&lb.mutex);
      if(single_work) return;
      continue;
    }

    // If only the top matrix portion is left
    if(end > lb.handler->end) {
      start -= lb.size;
      end -= lb.size;
    }

    // Dependencies handling
    const enum POSITION pos = (end <= lb.handler->start + pad_elems) ? TOP : BOTTOM;
    struct thread_handler* neigh_handler = NULL;
    uint8_t *rows_to_wait, *rows_to_assert;
    uint* pad_counter;
    if(lb.handler->neighbour[pos] != NULL) {
      neigh_handler = lb.handler->neighbour[pos];
      rows_to_wait = lb.handler->neighbour[pos]->rows_done[!pos];
      rows_to_assert = lb.handler->rows_done[pos];
      pad_counter = &lb.nrwos_pad_completed[pos];
    }

    // Wait if neighbours are late
    if(iter > 0 && neigh_handler != NULL) {
      t = PAPI_get_real_usec();
      pthread_mutex_lock(&(neigh_handler->mutex));
      *handlers_mutex_wait_time += PAPI_get_real_usec() - t;
      while(!rows_to_wait[iter_even])
        pthread_cond_wait(&(neigh_handler->pad_ready), &(neigh_handler->mutex));
      pthread_mutex_unlock(&(neigh_handler->mutex));
    }
    conv_subgrid(my_old_grid, my_new_grid, start, end);

    // Track pad completion and signal neighbour thread
    if(iter+1 < num_iterations && neigh_handler != NULL) {
      t = PAPI_get_real_usec();
      pthread_mutex_lock(&(lb.handler->mutex));
      *handlers_mutex_wait_time += PAPI_get_real_usec() - t;
      (*pad_counter)++;
      if(*pad_counter == pad_nrows) {
        rows_to_wait[iter_even] = 0;
        rows_to_assert[!iter_even] = 1;
        *pad_counter = 0;
        pthread_cond_broadcast(&(lb.handler->pad_ready));
      }
      pthread_mutex_unlock(&(lb.handler->mutex));
    }

    // Track row completion
    t = PAPI_get_real_usec();
    pthread_mutex_lock(&lb.mutex);
    *lb_mutex_wait_time += PAPI_get_real_usec() - t;
    lb.rows_completed++;                                      // Track the already computed row
    if(lb.rows_completed == lb.nrows) {                       // All shared works have been completed
      lb.rows_completed = 0;
      lb.curr_start = lb.handler->start + pad_elems;
      lb.iter++;
      pthread_cond_broadcast(&lb.iter_completed);
    }
    pthread_mutex_unlock(&lb.mutex);
    if(single_work) return;
  }
}

/* 
 * Compute a custom number of rows of a subset of load balancing rows (the ones having no dependencies). If 
 * the portion with no dependencies is completed, then variable "work_nrows" will assume value 0. 
*/
void load_balancing_custom(uint iter, uint* work_nrows, long_long** meas){
  uint work_elems = (*work_nrows) * grid_width;
  long_long tmp, *lb_mutex_wait_time = meas[3];

  float *my_new_grid, *my_old_grid;
  if(iter % 2) {
    my_new_grid = old_grid;
    my_old_grid = new_grid;
  } else {
    my_new_grid = new_grid;
    my_old_grid = old_grid;
  }

  // Reserve a *work_nrows number of rows to convolute
  uint start = 0, end = 0;
  tmp = PAPI_get_real_usec();
  pthread_mutex_lock(&lb.mutex);
  lb_mutex_wait_time += PAPI_get_real_usec() - tmp;
  if(iter == lb.iter && (lb.curr_start >= lb.handler->start + pad_elems)) {
    start = lb.curr_start;
    if(lb.curr_start <= (lb.handler->end - pad_elems - work_elems)) 
      lb.curr_start += work_elems;
    else if(lb.curr_start < (lb.handler->end - pad_elems))
      lb.curr_start = lb.handler->end - pad_elems;
    end = lb.curr_start;
  }
  pthread_mutex_unlock(&lb.mutex);

  // If all middle rows have been already computed 
  if(!start || start >= end) {
    *work_nrows = 0;
    return;
  }

  // Compute convolution and signal rows completion
  conv_subgrid(my_old_grid, my_new_grid, start, end);
  if(end-start != work_elems) *work_nrows = (end-start)/grid_width;

  tmp = PAPI_get_real_usec();
  pthread_mutex_lock(&lb.mutex);
  lb_mutex_wait_time += PAPI_get_real_usec() - tmp;
  lb.rows_completed += (*work_nrows);                     // Track the already computed row
  if(lb.rows_completed == lb.nrows) {                     // All shared works have been completed
    lb.rows_completed = 0;
    lb.curr_start = lb.handler->start + pad_elems;
    lb.iter++;
    pthread_cond_broadcast(&lb.iter_completed);
  }
  pthread_mutex_unlock(&lb.mutex);

  // If all middle rows have been computed   
  if(end == lb.handler->end - pad_elems) *work_nrows = 0;
}

/* Test shared memory dependencies, if possible convolute bordering rows and signal their completion */
void thread_polling(enum POSITION pos, uint iter, struct worker_data* worker, long_long** meas) {
  const uint8_t iter_even = !(iter % 2);                            // If current iteration is even or odd
  uint8_t *rtw, *rta;                                               // Rows to wait and to assert
  struct thread_handler* neighbour = worker->self->neighbour[pos];
  if(neighbour)
    rtw = &neighbour->rows_done[!pos][iter_even];            // pos: TOP or BOTTOM; iter_even: previous iteration
  rta = &worker->self->rows_done[pos][!iter_even];           // pos: TOP or BOTTOM; !iter_even: current iteration

  int* completed = worker->completed;
  long_long *condition_wait_time = meas[0];
  long_long *handler_mutex_wait_time = meas[1];
  long_long tmp;

  if(neighbour == NULL) {
    // No upper or lower shared memory dependencies
    completed[pos] = 1;
  } else if((!completed[CENTER] || !completed[!pos])) { 
    // If there is still other work to do, trylock insted of lock
    if(!pthread_mutex_trylock(&(neighbour->mutex))) {
      if(*rtw) {
        *rtw = 0;
        completed[pos] = 1;
      }
      pthread_mutex_unlock(&(neighbour->mutex));
    }
  } else {
    // Lock and make a single check without waiting
    tmp = PAPI_get_real_usec();
    pthread_mutex_lock(&(neighbour->mutex));
    *handler_mutex_wait_time += PAPI_get_real_usec() - tmp;
    if(*rtw) {
      *rtw = 0;
      completed[pos] = 1;
    }
    pthread_mutex_unlock(&(neighbour->mutex));

    while(!completed[pos]) {
      if(iter >= lb.iter){
        // If neighbour part still not ready, compute a single row of load balancing set of rows
        load_balancing(iter, 1, NULL, meas);
        tmp = PAPI_get_real_usec();
        pthread_mutex_lock(&(neighbour->mutex));
        *handler_mutex_wait_time += PAPI_get_real_usec() - tmp;
        if(*rtw) {
          *rtw = 0;
          completed[pos] = 1;
        }
        pthread_mutex_unlock(&(neighbour->mutex));
      } else {
        // No more work available, wait
        tmp = PAPI_get_real_usec();
        pthread_mutex_lock(&(neighbour->mutex));
        *handler_mutex_wait_time += PAPI_get_real_usec() - tmp;

        tmp = PAPI_get_real_usec();
        while(!(*rtw))
          pthread_cond_wait(&(neighbour->pad_ready), &(neighbour->mutex));
        *condition_wait_time += PAPI_get_real_usec() - tmp;
        *rtw = 0;
        pthread_mutex_unlock(&(neighbour->mutex));
        completed[pos] = 1;
      }
    }
  }
  
  // If test was successful, compute convolution of the tested part
  if(!completed[pos]) return;
  float *my_new_grid, *my_old_grid;
  if(!iter_even) {
    my_new_grid = old_grid;
    my_old_grid = new_grid;
  } else {
    my_new_grid = new_grid;
    my_old_grid = old_grid;
  }
  
  const uint start = (pos == TOP) ? worker->self->start : (worker->self->end - pad_elems);
  conv_subgrid(my_old_grid, my_new_grid, start, (start + pad_elems));

  // Signal/Send pad completion if a next convolution iteration exists
  if(iter+1 == num_iterations) return; 
  tmp = PAPI_get_real_usec();
  pthread_mutex_lock(&(worker->self->mutex));
  handler_mutex_wait_time += PAPI_get_real_usec() - tmp;
  *rta = 1;
  pthread_cond_broadcast(&(worker->self->pad_ready));
  pthread_mutex_unlock(&(worker->self->mutex));
}

/* Test distributed memory dependencies, if possible convolute bordering rows and exchange them with MPI */
void remote_polling(enum POSITION pos, uint iter, struct worker_data* worker, long_long** meas) {
  const uint8_t iter_even = !(iter % 2);                   // If current iteration is even or odd
  const uint8_t recv_offset = RECV_OFFSET(pos);            // Index of MPI_Irecv request in requests array
  int* completed = worker->completed;
  long_long tmp;

  // Checking distributed memory dependencies
  int outcount, indexes[SIM_REQS];
  MPI_Status statuses[SIM_REQS];
  long_long *mpi_mutex_wait_time = meas[2];


  // If there is still other work to do, trylock and test
  if(!reqs_completed[recv_offset] && !pthread_mutex_trylock(&mutex_mpi)){
    MPI_Testsome(SIM_REQS, requests, &outcount, indexes, statuses);
    pthread_mutex_unlock(&mutex_mpi);
    for(int i = 0; i < outcount; i++) reqs_completed[indexes[i]] = 1;
  }
  if(reqs_completed[recv_offset]) {
    completed[pos] = 1;
    reqs_completed[recv_offset] = 0;
  }

  // Else, do one row of the load balancer set of rows and wait
  while(!completed[pos] && completed[!pos] && completed[CENTER]) {
    if(iter >= lb.iter)
      load_balancing(iter, 1, NULL, meas);
    
    if(!reqs_completed[recv_offset]) {
      tmp = PAPI_get_real_usec();

      // If there are no more load balancing work to do lock and wait, else try lock (continue if trylock fail)       
      if(iter < lb.iter) pthread_mutex_lock(&mutex_mpi);
      else if(pthread_mutex_trylock(&mutex_mpi)) continue;

      *mpi_mutex_wait_time += PAPI_get_real_usec() - tmp;
      MPI_Waitsome(SIM_REQS, requests, &outcount, indexes, statuses);
      pthread_mutex_unlock(&mutex_mpi);
      for(int i = 0; i < outcount; i++) reqs_completed[indexes[i]] = 1;
    }
    
    if(reqs_completed[recv_offset]) {
      completed[pos] = 1;
      reqs_completed[recv_offset] = 0;
    }
  }

  
  // If test was successful, compute convolution of the tested part
  if(!completed[pos]) return;
  float *my_new_grid, *my_old_grid;
  if(!iter_even) {
    my_new_grid = old_grid;
    my_old_grid = new_grid;
  } else {
    my_new_grid = new_grid;
    my_old_grid = old_grid;
  }

  conv_subgrid(my_old_grid, my_new_grid, worker->mpi->send_position, (worker->mpi->send_position + pad_elems));

  // Signal/Send pad completion if a next convolution iteration exists
  if(iter+1 == num_iterations) return;
  const uint8_t send_offset = REQS_OFFSET(pos) + !iter_even;
  tmp = PAPI_get_real_usec();
  pthread_mutex_lock(&mutex_mpi);
  mpi_mutex_wait_time += PAPI_get_real_usec() - tmp;
  MPI_Isend(&my_new_grid[worker->mpi->send_position], pad_elems, MPI_FLOAT, worker->mpi->neighbour, 0, MPI_COMM_WORLD, &requests[send_offset]);
  MPI_Irecv(&my_new_grid[worker->mpi->recv_position], pad_elems, MPI_FLOAT, worker->mpi->neighbour, 0, MPI_COMM_WORLD, &requests[recv_offset]);
  pthread_mutex_unlock(&mutex_mpi);
}

/* Compute convolution of "sub_grid" in the specified range. Save the result in "new_grid" */
void conv_subgrid(float *sub_grid, float *new_grid, int start_index, int end_index) {
  float result = 0;
  float matrix_dot_sum = 0;                // Used for normalization
  int col = start_index % grid_width;      // Index of current sub_grid column
  int row_start = start_index - col;       // Index of the first element in current sub_grid row

  int offset;                              // How far is current element from its closest border
  int grid_index;                          // Current position in grid matrix
  int kern_index;                          // Current position in kernel matrix
  int kern_end;                            // Describes when it's time to change row
  int iterations;                          // How many iterations are necessary to calc a single value of "new_grid"
  __m128 vec_grid, vec_kern, vec_temp;     // 4xF32 vector for grid and kernel (plus a temp vector)
  __m128 vec_mxds, vec_rslt;               // 4xF32 vector of matrix dot sum and result, will be reduced at the end

  for(int i = start_index; i < end_index; i++) {
    // Setting indexes for current element
    if(col < pad_nrows) {
      for(offset = 0; i-offset > row_start && offset <= pad_nrows; offset++);
      grid_index = i-offset-pad_elems;
      kern_index = (kern_width / 2) - offset;
      kern_end = kern_width-kern_index;
      iterations = (pad_nrows+col+1) *kern_width;
    } else if (col > grid_width-1-pad_nrows){
      int row_end = row_start + grid_width - 1;
      for(offset = 0; i+offset <= row_end && offset <= pad_nrows; offset++);
      grid_index = i-pad_nrows-pad_elems;
      kern_index = 0;
      kern_end = kern_width-offset;
      iterations = (pad_nrows + grid_width-1-col) * kern_width;
    } else {
      grid_index = i-pad_nrows-pad_elems;
      kern_index = 0;
      kern_end = kern_width;
      iterations = kern_elems;
    }

    // Convolution
    if(iterations == kern_elems) {
      // Packed SIMD instructions for center elements
      vec_rslt = _mm_setzero_ps();
      vec_mxds = _mm_setzero_ps();
      for(int kern_row = 0; kern_row < kern_width; kern_row++) {       // For every kernel row
        for(offset = 0; offset < kern_width; offset += VEC_SIZE) {     // For every ps_vector in a kernel (and grid) row
          if(offset + VEC_SIZE < kern_width) {                         // If this isn't the final iteration of this loop, load a full vector
            vec_grid = _mm_loadu_ps(&sub_grid[grid_index+offset]);
            vec_kern = _mm_loadu_ps(&kernel[kern_index+offset]);
          } else {
            vec_grid = _mm_maskload_ps(&sub_grid[grid_index+offset], last_mask);
            vec_kern = _mm_maskload_ps(&kernel[kern_index+offset], last_mask);
          }
          vec_temp = _mm_mul_ps(vec_grid, vec_kern);
          vec_rslt = _mm_add_ps(vec_rslt, vec_temp);
          vec_temp = _mm_mul_ps(vec_grid, vec_grid);
          vec_mxds = _mm_add_ps(vec_mxds, vec_temp);
        }
        grid_index += grid_width;
        kern_index += kern_width;
      }
      vec_rslt = _mm_hadd_ps(vec_rslt, vec_rslt);  // Sum reduction with two horizontal sum
      vec_rslt = _mm_hadd_ps(vec_rslt, vec_rslt);
      result = vec_rslt[0];
      vec_mxds = _mm_hadd_ps(vec_mxds, vec_mxds);  // Sum reduction with two horizontal sum
      vec_mxds = _mm_hadd_ps(vec_mxds, vec_mxds);
      matrix_dot_sum = vec_mxds[0];
    } else {
      // Standard convolution
      result = 0; matrix_dot_sum = 0; offset = 0;
      for (int iter=0; iter < iterations; iter++) {
        result += sub_grid[grid_index+offset] * kernel[kern_index+offset];
        matrix_dot_sum += sub_grid[grid_index+offset] * sub_grid[grid_index+offset];
        if (offset != kern_end-1) 
          offset++;
        else { 
          grid_index += grid_width;
          kern_index += kern_width;
          offset = 0;
        }
      }
    }

    // Normalization (avoid NaN results by assigning the mean value if needed)
    new_grid[i] = (!matrix_dot_sum) ? MEAN_VALUE : (result / sqrt(matrix_dot_sum * kern_dot_sum));

    // Setting row and col indexes for next element
    if (col != grid_width-1)
      col++;
    else{
      row_start += grid_width;
      col = 0;
    }
  }
}

/* 
 * Return if this process should compute an additional row. This happens if the number of processes is
 * not a divider of the total number of rows of the input matrix. If the calling rank is 0, it also 
 * returns info about other processes.
 * 
 * For example, for 8 processes the assignment order of the additional rows would be: 
 * Rank : 0, 1, 2, 3, 4, 5, 6, 7
 * Order: 0, 2, 4, 6, 7, 5, 3, 1
*/
void get_process_additional_row(struct proc_info* procs_info) {
  // Additional rows that need to be distributed between processes
  const int addrows = grid_width % num_procs;
  int offset_from_last_rank = num_procs - 1 - rank;
  const uint8_t closer_to_final_rank = rank > offset_from_last_rank;

  // This var assume value 0 for first rank, 1 for last, 2 for second, 3 for penultimate, ...
  int order = (closer_to_final_rank) ? (1 + offset_from_last_rank * 2) : (rank * 2);

  // This var assume a logical true value if this rank should compute one additional row
  const uint8_t proc_additional_row = addrows > order;
  procs_info[0].has_additional_row = proc_additional_row;
  if(rank) return;

  // Rank 0 needs info about other ranks
  for(int i = 1; i < num_procs; i++) {
    offset_from_last_rank = num_procs - 1 - i;
    order = (i > offset_from_last_rank) ? (1 + offset_from_last_rank * 2) : (i * 2);
    procs_info[i].has_additional_row =  addrows > order;
  }
}

/* 
 * Initialize the coordinates (starting and ending position) of the submatrix that should be computed by
 * a thread. If the number of threads is not a divider of the number of rows assigned to this process, 
 * distribute those additional rows in the load_balancer submatrix. First execution of this procedure has 
 * to be with the load balancer handler as argument. 
*/
void initialize_thread_coordinates(struct thread_handler* handler) {
  int submatrix_id;

  if(handler->tid != -1) {
    // If this thread works in an area after the load balancing set of rows, compute a different offset
    submatrix_id = handler->tid;
    if(handler->tid >= num_threads/2) submatrix_id++;
  } else {
    // Load balancer (tid == -1) it's placed in the center of the submatrix
    submatrix_id = num_threads/2;
    
    // Initial amount of rows assigned to load balancer
    lb.nrows = num_threads;
    if(lb.nrows < pad_nrows * 2) lb.nrows += (pad_nrows * 2);

    // Initial amount of rows assigned to worker threads, remainder rows assigned to load balancer
    nrows_per_thread = (proc_nrows - lb.nrows) / num_threads;
    uint rem_rows = (proc_nrows - lb.nrows) % num_threads;
    lb.nrows += rem_rows;

    // Amount of rows for threads having mpi dependencies (minimum is: pad_nrows*2 + 1)
    bordering_thread_nrows = pad_nrows * 3;
    num_bordering_threads = 0;
    if(num_procs > 1 && num_threads > 2) {
      num_bordering_threads++;
      if(rank > 0 && rank < num_procs-1) 
        num_bordering_threads++;
    }

    // Add to load balancer set of rows the work offload of bordering threads and recompute worker thread rows
    max_lb_amount = nrows_per_thread - bordering_thread_nrows;
    lb.nrows += (nrows_per_thread - bordering_thread_nrows) * num_bordering_threads;
    workers_nrows = proc_nrows - lb.nrows - num_bordering_threads * bordering_thread_nrows;
    nrows_per_thread = workers_nrows / (num_threads - num_bordering_threads);
    rem_rows = workers_nrows % (num_threads - num_bordering_threads);

    workers_nrows -= rem_rows;
    lb.nrows += rem_rows;
    lb.size = lb.nrows * grid_width;
  }

  // Initialize coordinates
  const uint thread_elems = nrows_per_thread * grid_width;
  const uint bordering_thread_elems = (!num_bordering_threads) ? thread_elems : bordering_thread_nrows * grid_width;
  uint actual_size;
  uint offset = pad_elems;

  if(!handler->tid && rank) {
    // Upper thread has mpi dependency
    actual_size = bordering_thread_elems;
    handler->start = offset;
    handler->end = handler->start + actual_size;
  } else if (num_procs > 1 && handler->tid == num_threads-1 && rank < num_procs-1) {
    // Lower thread has mpi dependency
    actual_size = bordering_thread_elems;
    handler->end = (proc_nrows * grid_width) + offset;
    handler->start = handler->end - actual_size;
  } else {
    if(rank) {
      // If upper thread have mpi dependency compute a different offset
      offset += bordering_thread_elems;
      handler->start = submatrix_id - 1;
    } else handler->start = submatrix_id;
    
    handler->start = handler->start * thread_elems + offset;
    if (handler->tid == -1) {
      actual_size = lb.size;
    } else {
      actual_size = thread_elems;
      // If this thread works in an area after the load balancing set of rows, compute a different offset
      if(submatrix_id > num_threads/2) 
        handler->start = handler->start - thread_elems + lb.size;
    }
    handler->end = handler->start + actual_size;
  }
}
