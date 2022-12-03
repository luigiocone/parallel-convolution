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
#define MEAN_VALUE 0
#define WORKER_FACTOR 1               // Used to calc how many rows must be computed from worker before polling the neighbours
#define COORDINATOR_FACTOR 1          // Used to calc how many rows must be computed from coordinator before checking MPI dependencies
#define DEBUG 0                       // True to save result in textual and binary mode
#define SEND_OFFSET(POS) ((POS) * SIM_RECV + SIM_RECV)

void *coordinator_thread(void*);
void *worker_thread(void*);
void *setup_thread(void*);
void thread_polling(uint8_t, uint, struct local_data*, long_long**);
void node_polling(uint8_t, uint, struct local_data*, struct node_data*, long_long**);
void wait_protocol(uint, struct local_data*, struct node_data*, long_long**);
void load_balancing(int, uint8_t, long_long**);
void load_balancing_custom(int, uint*, uint8_t, long_long**);
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
uint rbf_elems;                       // Rows before polling float elements, how many rows must be computed from worker before polling the neighbours
uint kern_elems;                      // Number of elements in whole kernel matrix
uint grid_elems;                      // Number of elements in whole grid matrix
uint proc_nrows;                      // Number of rows assigned to a process
uint static_work_nrows;               // Number of rows assigned to a process and distributed to all workers (hence not owned by load_balancer)
uint coordinator_nrows;               // Subset of load_balancer set of rows that coordinator should compute
uint coordinator_work_elems;          // Number of elements in a single coordinator work  
struct setup_args setup;              // Used by setup, worker, and main threads to synchronize about grid read
struct load_balancer lb;              // Structure containing all synchronization variables used during load balancing 
long_long** thread_measures;          // To collect thread measures
int num_procs;                        // Number of MPI processes in the communicator
int num_threads;                      // Number of threads (main included) for every MPI process
int num_iterations;                   // Number of convolution iterations
int rank;                             // MPI process identifier
__m128i last_mask;                    // Used by PSIMD instructions to discard last elements in contiguous load

int main(int argc, char** argv) {
  int provided;                       // MPI thread level supported
  int rc;                             // Return code used in error handling
  long_long time_start, time_stop;    // To measure execution time
  long_long grid_read_time;           // Measuring disk read time of grid to convolute
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
  if((rc = MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided)) != MPI_SUCCESS) {
    fprintf(stderr, "MPI_Init error. Return code: %d\n", rc);
    exit(-1);
  } 
  if(provided < MPI_THREAD_FUNNELED) {
    fprintf(stderr, "Minimum MPI threading level requested: %d (provided: %d)\n", MPI_THREAD_FUNNELED, provided);
    exit(-1);
  }
  MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  time_start = PAPI_get_real_usec();
  
  struct proc_info procs_info[(rank) ? 1 : num_procs];
  MPI_Request info_reqs[num_procs-1];                         // Used by rank 0 to ditribute initial informations (matrices size)
  MPI_Request kern_reqs[num_procs-1];                         // Used by rank 0 to ditribute kernel matrix
  MPI_Request grid_reqs[num_procs-1];                         // Used by rank 0 to ditribute grid matrix
  pthread_t threads[num_threads];                             // Worker threads except the middle one (num_threads/2) used as setup thread while main is reading from disk
  lb = (struct load_balancer){0};                             // Load balancer is not an active thread. This structure is used by all workers
  lb.handler = &(struct thread_handler){0};                   // Used to synchronize with neighbour threads
  setup.threads = threads;
  setup.procs_info = procs_info;
  memset(setup.flags, 0, sizeof(uint8_t) * (SEND_INFO+1));    // Used as flag, will be different from 0 only after the grid is ready to be convoluted
  pthread_mutex_init(&(setup.mutex), NULL);                   // Used by io_thread and main thread to synchronize
  pthread_cond_init(&(setup.cond), NULL);

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
    old_grid = malloc((grid_width + pad_nrows*2) * grid_width * sizeof(float));
    if ((rc = pthread_create(&threads[num_threads/2], NULL, setup_thread, NULL))) { 
      fprintf(stderr, "Error while creating the setup thread; Return code: %d\n", rc);
      exit(-1);
    }

    // Read grid and send its partitions to the MPI nodes
    grid_read_time = PAPI_get_real_usec();  
    read_float_matrix(fp_grid, &old_grid[pad_elems], grid_elems);
    grid_read_time = PAPI_get_real_usec() - grid_read_time;

    // Signal that grid read have been completed and check if processes info are ready
    pthread_mutex_lock(&(setup.mutex));
    setup.flags[GRID] = 1;
    pthread_cond_broadcast(&(setup.cond));
    if(num_procs > 1) {
      while(!setup.flags[SEND_INFO]) pthread_cond_wait(&(setup.cond), &(setup.mutex));
    }
    pthread_mutex_unlock(&(setup.mutex));

    // Grid distribution
    if(num_procs > 1) {
      for(int p = 1; p < num_procs; p++)
        MPI_Isend(&old_grid[setup.procs_info[p].sstart], setup.procs_info[p].ssize, MPI_FLOAT, p, p, MPI_COMM_WORLD, &grid_reqs[p-1]);
    }
  } else {
    uint32_t to_recv[2];
    MPI_Recv(to_recv, 2, MPI_UINT32_T, 0, rank, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    grid_width = to_recv[0];
    kern_width = to_recv[1];
    kern_elems = kern_width * kern_width;
    grid_elems = grid_width * grid_width;
    pad_nrows = (kern_width - 1) / 2;
    pad_elems = grid_width * pad_nrows;
    kernel = malloc(sizeof(float) * kern_elems);
    MPI_Irecv(kernel, kern_elems, MPI_FLOAT, 0, rank, MPI_COMM_WORLD, &(setup.requests[0]));
    setup_thread(NULL);
  }

  pthread_mutex_lock(&(setup.mutex));
  while(!setup.flags[HANDLERS]) 
    pthread_cond_wait(&(setup.cond), &(setup.mutex));
  pthread_mutex_unlock(&(setup.mutex));

  // If mpi is needed, main thread is placed around the center of the matrix and works with load_balancer rows
  if(num_procs > 1) 
    coordinator_thread((void*) &setup.handlers[num_threads/2]);
  else 
    worker_thread((void*) &setup.handlers[num_threads/2]);

  // Check if 'setup_thread' has exited and start recv listeners to gather results
  if(!rank && pthread_join(threads[num_threads/2], (void*) &rc)) {
    fprintf(stderr, "Join error, setup thread exited with: %d", rc);
  }

  MPI_Request res_gather_reqs[num_procs-1];
  float *res_grid = (num_iterations % 2) ? new_grid : old_grid;
  if(!rank && num_procs > 1) {
    for(int p = 1; p < num_procs; p++)
      MPI_Irecv(&res_grid[procs_info[p].gstart], procs_info[p].gsize, MPI_FLOAT, p, p, MPI_COMM_WORLD, &res_gather_reqs[p-1]);
  }

  // Wait workers termination and collect thread return values
  for(int i = 0; i < num_threads; i++) {
    if(i == num_threads/2) continue;
    if(pthread_join(threads[i], (void*) &rc)) 
      fprintf(stderr, "Join error, thread[%d] exited with: %d", i, rc);
  }

  if(rank) {
    MPI_Send(&res_grid[pad_elems], proc_nrows*grid_width, MPI_FLOAT, 0, rank, MPI_COMM_WORLD);
  } else if(num_procs > 1) {
    MPI_Status statuses[num_procs-1];
    MPI_Waitall(num_procs-1, res_gather_reqs, statuses);
  }

  // Print measures
  time_stop = PAPI_get_real_usec();
  printf("Rank[%d] | Elapsed time: %lld us\n", rank, (time_stop - time_start));
  if(!rank) printf("Rank[0] | Grid read from disk time: %lld us\n", grid_read_time);

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
  pthread_mutex_destroy(&lb.mutex);
  pthread_mutex_destroy(&setup.mutex);
  pthread_cond_destroy(&setup.cond);
  pthread_cond_destroy(&lb.iter_completed);
  for(int i = 0; i < num_threads; i++) {
    pthread_mutex_destroy(&setup.handlers[i].mutex);
    pthread_cond_destroy(&setup.handlers[i].pad_ready);
  }

  MPI_Finalize();
  if(!rank) {
    fclose(fp_kernel);
    fclose(fp_grid);
  }
  free(thread_measures);
  free(setup.handlers);
  free(new_grid);
  free(old_grid);
  free(kernel);
  exit(0);
}

/* 
 * Executed by a thread while main thread is reading from disk or as a function call by ranks different from 0. 
 * Initialize global variables, setup data structure used during convolution, setup load balancer, create 
 * threads and setup their handler connections. Ranks different from 0 make also some MPI calls to start data
 * receiving. Some synchronization is needed if read from disk end earlier.
*/
void* setup_thread(void* args) { 
  // Checking if input is big enough for work division between threads and load balancing 
  if(grid_width/(num_procs * num_threads + num_procs) < pad_nrows * 3) {
    fprintf(stderr, "Threading is oversized compared to input matrix. Threads: %dx%d | Input number of rows: %d\n", num_procs, num_threads, grid_width);
    exit(-1);
  }

  const uint rbf = pad_nrows + WORKER_FACTOR;       // Rows before polling. How many rows must be computed before polling the neighbours
  rbf_elems = rbf * grid_width;                     // Rows before polling number of float elements

  // Rank 0 get info about which processes must compute an additional row, other ranks get info only about themself
  get_process_additional_row(setup.procs_info); 
  const uint min_nrows = (grid_width / num_procs);                         // Minimum amout of rows distributed to each process
  proc_nrows = min_nrows + setup.procs_info[0].has_additional_row;         // Number of rows assigned to current process
  const uint proc_nrows_size = proc_nrows * grid_width;                    // Number of elements assigned to a process

  if(!rank) {
    new_grid = malloc((grid_elems + pad_elems*2) * sizeof(float));
  } else {
    new_grid = malloc((proc_nrows_size + pad_elems*2) * sizeof(float));
    old_grid = malloc((proc_nrows_size + pad_elems*2) * sizeof(float));    // Ranks different from 0 has a smaller grid to alloc
  }

  // Rank 0 prepares send/recv info (synch point with main thread), other ranks receives grid data. 
  if(!rank){
    setup.procs_info[0].sstart = 0;
    setup.procs_info[0].ssize = (proc_nrows + pad_nrows*2) * grid_width;
    if(num_procs > 1) {
      // Info about data scattering. Pads are included (to avoid an MPI exchange in the first iteration)
      int offset = setup.procs_info[0].has_additional_row;
      for(int i = 1; i < num_procs; i++) {
        setup.procs_info[i].sstart = (min_nrows * i + offset) * grid_width;
        setup.procs_info[i].ssize = (min_nrows + pad_nrows*2 + setup.procs_info[i].has_additional_row) * grid_width;
        if(i == num_procs-1) setup.procs_info[i].ssize -= grid_width * pad_nrows;
        offset += setup.procs_info[i].has_additional_row;
      }
      pthread_mutex_lock(&(setup.mutex));
      setup.flags[SEND_INFO] = 1;
      pthread_cond_signal(&(setup.cond));
      pthread_mutex_unlock(&(setup.mutex));
      
      // Info about result gathering from previous scattering data. Pads are excluded
      for(int p = 1; p < num_procs; p++) {
        setup.procs_info[p].gstart = setup.procs_info[p].sstart + pad_nrows * grid_width;
        setup.procs_info[p].gsize = setup.procs_info[p].ssize - pad_nrows * 2 * grid_width;
        if(p == num_procs-1) setup.procs_info[p].gsize += grid_width * pad_nrows;
      }
    }
  } else {
    // Receive grid data
    uint recv_size = (proc_nrows + pad_nrows * 2) * grid_width;
    if(rank == num_procs-1) recv_size -= grid_width;
    MPI_Irecv(old_grid, recv_size, MPI_FLOAT, 0, rank, MPI_COMM_WORLD, &(setup.requests[1]));
  }

  // Set a zero-pad for lowest and highest process 
  if(!rank){
    memset(new_grid, 0, pad_elems * sizeof(float));
    memset(old_grid, 0, pad_elems * sizeof(float));
  }
  if(rank == num_procs-1) {
    memset(&new_grid[proc_nrows_size + pad_elems], 0, pad_elems * sizeof(float));
    memset(&old_grid[proc_nrows_size + pad_elems], 0, pad_elems * sizeof(float));
  }

  // PThreads arguments initialization 
  struct thread_handler* handlers = calloc(num_threads, sizeof(struct thread_handler));
  for(int i = 0; i < num_threads; i++) {
    handlers[i].tid = i;
    handlers[i].top = (i > 0) ? &handlers[i-1] : NULL;
    handlers[i].bottom = (i < num_threads-1) ? &handlers[i+1] : NULL;
    pthread_mutex_init(&handlers[i].mutex, NULL);
    pthread_cond_init(&handlers[i].pad_ready, NULL);
  }

  // Managing lb handler and mt handler connections
  lb.handler->tid = -1;
  lb.handler->top = &handlers[num_threads/2-1];             // top neighbour is always the same
  handlers[num_threads/2-1].bottom = lb.handler;
  struct thread_handler* mth = &handlers[num_threads/2];    // main thread handler (tid == num_threads/2)

  if(num_procs == 1) {
    lb.handler->bottom = &handlers[num_threads/2];
    handlers[num_threads/2].top = lb.handler;
  } else {                                                 // bottom neighbour is different if num_procs > 1 
    if(num_threads > 2) {
      lb.handler->bottom = &handlers[num_threads/2+1];
      handlers[num_threads/2+1].top = lb.handler;
    }

    if(!rank){
      handlers[0].top = NULL;
      mth->bottom = NULL;
    } else {
      handlers[0].top = mth;
      mth->bottom = &handlers[0];
    }

    if(rank == num_procs-1) {
      handlers[num_threads-1].bottom = NULL;
      mth->top = NULL;
      if(num_threads == 2) {
        lb.handler->bottom = NULL;
        mth->bottom = &handlers[0];
      }
    } else {
      if(num_threads == 2){
        lb.handler->bottom = mth;
        mth->top = lb.handler;
      } else {
        handlers[num_threads-1].bottom = mth;
        mth->top = &handlers[num_threads-1];
      }
    }
  }

  pthread_mutex_init(&lb.mutex, NULL);
  pthread_cond_init(&lb.iter_completed, NULL);
  pthread_mutex_init(&(lb.handler->mutex), NULL);
  pthread_cond_init(&(lb.handler->pad_ready), NULL);
  initialize_thread_coordinates(lb.handler);
  lb.curr_start = lb.handler->start + pad_elems;               // Start from submatrix with no dependencies
  thread_measures = malloc(sizeof(long_long*) * num_threads);
  if(!rank && num_procs == 1) 
    initialize_thread_coordinates(&handlers[num_threads/2]);   // Initialize main thread coordinates while he is reading

  // Computation of "last_mask"
  uint32_t rem = kern_width % VEC_SIZE;
  uint32_t to_load[VEC_SIZE];
  memset(to_load, 0, VEC_SIZE * sizeof(uint32_t));
  for(int i = 0; i < rem; i++) to_load[i] = UINT32_MAX;        // UINT32_MAX = -1
  last_mask = _mm_loadu_si128((__m128i*) to_load);

  setup.handlers = handlers;
  pthread_mutex_lock(&(setup.mutex));
  setup.flags[HANDLERS] = 1;
  pthread_cond_signal(&(setup.cond));
  pthread_mutex_unlock(&(setup.mutex));

  // PThreads creation
  for(int i = 0; i < num_threads; i++) {
    if(i == num_threads/2) continue;
    int rc = pthread_create(&setup.threads[i], NULL, worker_thread, (void*)&handlers[i]);
    if(rc) { 
      fprintf(stderr, "Error while creating pthread 'worker_thread[%d]'; Return code: %d\n", i, rc);
      exit(-1);
    }
  }

  if(!rank) pthread_exit(0);
  else return 0;
}

/* 
 * Executed by main thread if number of processes is greater than 1, else main thread execute 'worker_thread' procedure. 
 * Differs from worker_thread procedure because it deals MPI dependencies and works only in the load_balancing area (because 
 * of the unpredictable amount of work associated with MPI)
*/
void* coordinator_thread(void* args) {
  struct thread_handler *handler = (struct thread_handler*)args;
  float *my_old_grid = old_grid;
  float *my_new_grid = new_grid;
  float *tmp;                                       // Used only for grid swap
  uint8_t changes = 0;                              // If no changes have been made (or cannot be made), main thread will wait
  struct local_data local = {0};
  struct node_data node = {0};
  const uint bottom_pad_start = proc_nrows * grid_width;
  const uint work_nrows = pad_nrows * COORDINATOR_FACTOR;
  coordinator_work_elems = pad_elems * COORDINATOR_FACTOR;

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

  // Data structure initialization
  for(int i = 0; i < SIM_REQS; i++) node.requests[i] = MPI_REQUEST_NULL;
  local.self = handler;

  if(handler->top == NULL) local.completed[TOP] = 1;
  else if(num_iterations > 1) {
    node.neighbour[TOP] = rank+1;
    node.send_position[TOP] = bottom_pad_start;
    node.recv_position[TOP] = bottom_pad_start + pad_elems;
    local.neigh[TOP] = handler->top;
    local.rows_to_wait[TOP] = handler->top->bot_rows_done;
    local.rows_to_assert[TOP] = handler->top_rows_done;
  }

  if(handler->bottom == NULL) local.completed[BOTTOM] = 1;
  else if(num_iterations > 1) {
    node.neighbour[BOTTOM] = rank-1;
    node.send_position[BOTTOM] = pad_elems;
    node.recv_position[BOTTOM] = 0;
    local.neigh[BOTTOM] = handler->bottom;
    local.rows_to_wait[BOTTOM] = handler->bottom->top_rows_done;
    local.rows_to_assert[BOTTOM] = handler->bot_rows_done;
  }

  // Complete kernel receive and/or compute "sum(dot(kernel, kernel))"
  if(rank) MPI_Wait(&(setup.requests[0]), MPI_STATUS_IGNORE);
  for(int i = 0; i < kern_elems; i++) {
    kern_dot_sum += kernel[i] * kernel[i];
  }
  pthread_mutex_lock(&(setup.mutex));
  setup.flags[KERNEL] = 1;
  pthread_cond_broadcast(&(setup.cond));
  pthread_mutex_unlock(&(setup.mutex));

  // Synchronization point, check if grid data is ready and start convolution
  if(rank) { 
    MPI_Wait(&(setup.requests[1]), MPI_STATUS_IGNORE);
    pthread_mutex_lock(&(setup.mutex));
    setup.flags[GRID] = 1;
    pthread_cond_broadcast(&(setup.cond));
    pthread_mutex_unlock(&(setup.mutex));
  } else {
    pthread_mutex_lock(&(setup.mutex));
    while(!setup.flags[GRID] || !setup.flags[KERNEL])
      pthread_cond_wait(&(setup.cond), &(setup.mutex));
    pthread_mutex_unlock(&(setup.mutex));
  }

  // First convolution iteration (starting with top and bottom mpi rows)
  if(handler->bottom != NULL) {
    conv_subgrid(my_old_grid, my_new_grid, pad_elems, pad_elems*2);
    if(num_iterations > 1) {
      MPI_Isend(&my_new_grid[node.send_position[BOTTOM]], pad_elems, MPI_FLOAT, rank-1, 0, MPI_COMM_WORLD, &node.requests[SEND_OFFSET(BOTTOM)]);
      MPI_Irecv(&my_new_grid[node.recv_position[BOTTOM]], pad_elems, MPI_FLOAT, rank-1, 0, MPI_COMM_WORLD, &node.requests[BOTTOM]);
    }
  }
  if(handler->top != NULL) {
    conv_subgrid(my_old_grid, my_new_grid, bottom_pad_start, bottom_pad_start + pad_elems);
    if(num_iterations > 1) {
      MPI_Isend(&my_new_grid[node.send_position[TOP]], pad_elems, MPI_FLOAT, rank+1, 0, MPI_COMM_WORLD, &node.requests[SEND_OFFSET(TOP)]);
      MPI_Irecv(&my_new_grid[node.recv_position[TOP]], pad_elems, MPI_FLOAT, rank+1, 0, MPI_COMM_WORLD, &node.requests[TOP]);
    }
  }

  if(num_iterations > 1) {
    t = PAPI_get_real_usec();
    pthread_mutex_lock(&(handler->mutex));
    handler_mutex_wait_time += PAPI_get_real_usec() - t;
    handler->top_rows_done[0] = 1;
    handler->bot_rows_done[0] = 1;
    pthread_cond_broadcast(&(handler->pad_ready));
    pthread_mutex_unlock(&(handler->mutex));
  }

  // Complete the first convolution iteration by computing central elements
  if(!lb.iter) load_balancing(0, 0, measures);

  // Second or higher convolution iterations
  int* completed = local.completed;
  for(int iter = 1; iter < num_iterations; iter++) {
    tmp = my_old_grid;
    my_old_grid = my_new_grid;
    my_new_grid = tmp;
    completed[CENTER] = 0;
    if(handler->top != NULL) completed[TOP] = 0;
    if(handler->bottom != NULL) completed[BOTTOM] = 0;

    while(!completed[TOP] || !completed[BOTTOM] || !completed[CENTER]) {
      changes = 0;
      if(!completed[BOTTOM]){
        node_polling(BOTTOM, iter, &local, &node, measures);
        changes |= completed[BOTTOM];
      }

      if(!completed[TOP]){
        node_polling(TOP, iter, &local, &node, measures);
        changes |= completed[TOP];
      }

      if(!completed[CENTER]) {
        if(iter >= lb.iter) { 
          completed[CENTER] = 1;
          continue;
        }
        uint rows_computed = work_nrows;
        load_balancing_custom(iter, &rows_computed, (completed[TOP] && completed[BOTTOM]), measures);
        if(!rows_computed) completed[CENTER] = 1;
        changes = 1;
      }

      if(!changes) wait_protocol(iter, &local, &node, measures);
    }

    // Load balancing if this thread ended current iteration earlier
    if(iter >= lb.iter) load_balancing(iter, 0, measures);
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

  return 0;
}

void* worker_thread(void* args) {  
  struct thread_handler *handler = (struct thread_handler*)args;
  float *my_old_grid = old_grid;
  float *my_new_grid = new_grid;
  float *tmp;                                       // Used only for grid swap
  int center_start;                                 // Center elements are completed one row at a time
  uint8_t changes;                                  // To track if something changed during polling
  struct local_data local = {0};

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

  // Thread setup. Coordinates of main thread could have been initialized by setup thread
  if(affinity && stick_this_thread_to_core(handler->tid)) {
    fprintf(stderr, "Error occurred while setting thread affinity on core: %d\n", handler->tid);
    exit(-1);
  }
  if(num_procs > 1 || handler->tid != num_threads/2)
    initialize_thread_coordinates(handler);
  const uint actual_end = handler->end-pad_elems;   // Where central part ends

  // Compute "sum(dot(kernel, kernel))" if there is only one process
  if(!handler->tid && num_procs == 1) {
    for(int pos = 0; pos < kern_elems; pos++) {
      kern_dot_sum += kernel[pos] * kernel[pos];
    }
    pthread_mutex_lock(&(setup.mutex));
    setup.flags[KERNEL] = 1;
    pthread_cond_broadcast(&(setup.cond));
    pthread_mutex_unlock(&(setup.mutex));
  }

  // Synchronization point, check if grid data is ready and start convolution
  pthread_mutex_lock(&(setup.mutex));
  while(!setup.flags[GRID] || !setup.flags[KERNEL])
    pthread_cond_wait(&(setup.cond), &(setup.mutex));
  pthread_mutex_unlock(&(setup.mutex));

  // First convolution iteration (starting with top and bottom rows)
  conv_subgrid(my_old_grid, my_new_grid, handler->start, (handler->start + pad_elems));
  conv_subgrid(my_old_grid, my_new_grid, (handler->end - pad_elems), handler->end);

  if(num_iterations > 1) {
    t = PAPI_get_real_usec();
    pthread_mutex_lock(&(handler->mutex));
    handler_mutex_wait_time += PAPI_get_real_usec() - t;
    handler->top_rows_done[0] = 1;
    handler->bot_rows_done[0] = 1;
    pthread_cond_broadcast(&(handler->pad_ready));
    pthread_mutex_unlock(&(handler->mutex));

    local.self = handler;
    local.neigh[TOP] = handler->top;
    local.neigh[BOTTOM] = handler->bottom;
    local.rows_to_assert[TOP] = handler->top_rows_done;
    local.rows_to_assert[BOTTOM] = handler->bot_rows_done;
    if(local.neigh[TOP]) local.rows_to_wait[TOP] = handler->top->bot_rows_done;
    if(local.neigh[BOTTOM]) local.rows_to_wait[BOTTOM] = handler->bottom->top_rows_done;
  }

  // Complete the first convolution iteration by computing central elements
  conv_subgrid(my_old_grid, my_new_grid, (handler->start + pad_elems), (handler->end - pad_elems));
  if(!lb.iter) load_balancing(0, 0, measures);

  // Second or higher convolution iterations
  int* completed = local.completed;
  for(int iter = 1; iter < num_iterations; iter++) {
    tmp = my_old_grid;
    my_old_grid = my_new_grid;
    my_new_grid = tmp;
    center_start = handler->start + pad_elems;
    memset(completed, 0, sizeof(int) * (CENTER+1));

    while(!completed[TOP] || !completed[BOTTOM] || !completed[CENTER]) {
      changes = 0;
      if(!completed[TOP]) {
        thread_polling(TOP, iter, &local, measures);
        changes |= completed[TOP];
      }

      if(!completed[BOTTOM]) {
        thread_polling(BOTTOM, iter, &local, measures);
        changes |= completed[BOTTOM];
      }

      // Computing central rows one at a time if top and bottom rows are incomplete
      if(!completed[CENTER]) {
        uint center_end;
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
        changes = 1;
      }

      if(!changes) load_balancing(iter, 1, measures);
    }

    // Load balancing if this thread ended current iteration earlier
    if(iter >= lb.iter) load_balancing(iter, 0, measures);
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

  if(handler->tid == num_threads/2) return 0;
  pthread_exit(0);
}

/* Threads that ended their iteration earlier will compute a shared portion of the matrix */
void load_balancing(int iter, uint8_t single_work, long_long** meas){
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

  // Compute a row of the load_balancer shared work 
  while(iter >= lb.iter) {
    start = 0;
    t = PAPI_get_real_usec();
    pthread_mutex_lock(&lb.mutex);
    *lb_mutex_wait_time += PAPI_get_real_usec() - t;
    while(iter > lb.iter)
      pthread_cond_wait(&lb.iter_completed, &lb.mutex);         // Wait if lb work of previous iteration is not completed yet
    if(iter == lb.iter) {
      start = lb.curr_start;
      lb.curr_start += grid_width;                              // From lb->start to lb->end
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
    struct thread_handler* neigh_handler = NULL;
    uint8_t *rows_to_wait, *rows_to_assert;
    uint* pad_counter;
    if(end <= lb.handler->start + pad_elems) {           // Top pad
      neigh_handler = lb.handler->top;
      rows_to_wait = lb.handler->top->bot_rows_done;
      rows_to_assert = lb.handler->top_rows_done;
      pad_counter = &lb.top_pad;
    } else if(lb.handler->bottom != NULL){               // Bottom pad
      neigh_handler = lb.handler->bottom;
      rows_to_wait = lb.handler->bottom->top_rows_done;
      rows_to_assert = lb.handler->bot_rows_done;
      pad_counter = &lb.bot_pad;
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
 * Compute a custom number of rows of the load balancing middle rows (the ones having no dependencies). The flag 
 * "check_for_more_work" is used by coordinator if it has completed all other works dealing with neighbours. If 
 * the coordinator portion is completed then variable "work_nrows" will assume value 0. 
*/
void load_balancing_custom(int iter, uint* work_nrows, uint8_t check_for_more_work, long_long** meas){
  uint start = 0, end = 0;
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

  // Reserve a number of rows 
  tmp = PAPI_get_real_usec();
  pthread_mutex_lock(&lb.mutex);
  lb_mutex_wait_time += PAPI_get_real_usec() - tmp;
  if(iter == lb.iter && (lb.curr_start >= lb.handler->start + pad_elems)) {
    if(check_for_more_work && (lb.rows_completed < coordinator_nrows))
      work_elems = (coordinator_nrows - lb.rows_completed) * grid_width;
    start = lb.curr_start;
    if(lb.curr_start <= (lb.handler->end - pad_elems - work_elems)) 
      lb.curr_start += work_elems;
    else
      lb.curr_start = lb.handler->end;
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
  if(end-start != coordinator_work_elems) *work_nrows = (end-start)/grid_width;
  tmp = PAPI_get_real_usec();
  pthread_mutex_lock(&lb.mutex);
  lb_mutex_wait_time += PAPI_get_real_usec() - tmp;
  lb.rows_completed += *work_nrows;                       // Track the already computed row
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

/* Test if pad rows are ready. If they are, compute their convolution and send/signal their completion */
void thread_polling(uint8_t pos, uint iter, struct local_data* ld, long_long** meas) {
  const uint8_t iter_even = !(iter % 2);               // If current iteration is even or odd
  uint8_t* rtw = &ld->rows_to_wait[pos][iter_even];    // pos: TOP or BOTTOM; iter_even: previous iteration
  uint8_t* rta = &ld->rows_to_assert[pos][!iter_even]; // pos: TOP or BOTTOM; !iter_even: current iteration
  long_long *condition_wait_time = meas[0];
  long_long tmp;

  if(ld->neigh[pos] == NULL) ld->completed[pos] = 1;          // No upper or lower shared memory dependencies
  else {
    if((!ld->completed[CENTER] || !ld->completed[!pos]) && !pthread_mutex_trylock(&(ld->neigh[pos]->mutex))) {
      // If there is still other work to do, trylock insted of lock 
      if(*rtw) {
        *rtw = 0;
        ld->completed[pos] = 1;
      }
      pthread_mutex_unlock(&(ld->neigh[pos]->mutex));
    } else {
      long_long *handler_mutex_wait_time = meas[1];
      
      // Make a first single check without waiting
      tmp = PAPI_get_real_usec();
      pthread_mutex_lock(&(ld->neigh[pos]->mutex));
      *handler_mutex_wait_time += PAPI_get_real_usec() - tmp;
      if(*rtw) {
        *rtw = 0;
        ld->completed[pos] = 1;
      }
      pthread_mutex_unlock(&(ld->neigh[pos]->mutex));
      
      // If neighbour part still not ready, make a single work of load balancing set of rows and wait
      while(!ld->completed[pos]) {
        if(iter >= lb.iter){
          load_balancing(iter, 1, meas);
          tmp = PAPI_get_real_usec();
          pthread_mutex_lock(&(ld->neigh[pos]->mutex));
          *handler_mutex_wait_time += PAPI_get_real_usec() - tmp;
          if(*rtw) {
            *rtw = 0;
            ld->completed[pos] = 1;
          }
          pthread_mutex_unlock(&(ld->neigh[pos]->mutex));
        } else {
          tmp = PAPI_get_real_usec();
          pthread_mutex_lock(&(ld->neigh[pos]->mutex));
          *handler_mutex_wait_time += PAPI_get_real_usec() - tmp;

          tmp = PAPI_get_real_usec();
          while(!(*rtw))
            pthread_cond_wait(&(ld->neigh[pos]->pad_ready), &(ld->neigh[pos]->mutex));
          *condition_wait_time += PAPI_get_real_usec() - tmp;
          *rtw = 0;
          pthread_mutex_unlock(&(ld->neigh[pos]->mutex));
          ld->completed[pos] = 1;
        }
      }
    }
  }
  
  // If test was successful, compute convolution of the tested part
  if(!ld->completed[pos]) return;
  long_long *handler_mutex_wait_time = meas[1];
  float *my_new_grid, *my_old_grid;
  if(!iter_even) {
    my_new_grid = old_grid;
    my_old_grid = new_grid;
  } else {
    my_new_grid = new_grid;
    my_old_grid = old_grid;
  }
  
  const int start = (pos == TOP) ? ld->self->start : (ld->self->end - pad_elems);
  conv_subgrid(my_old_grid, my_new_grid, start, (start + pad_elems));

  // Signal/Send pad completion if a next convolution iteration exists
  if(iter+1 == num_iterations) return;
  tmp = PAPI_get_real_usec();
  pthread_mutex_lock(&(ld->self->mutex));
  handler_mutex_wait_time += PAPI_get_real_usec() - tmp;
  *rta = 1;
  pthread_cond_broadcast(&(ld->self->pad_ready));
  pthread_mutex_unlock(&(ld->self->mutex));
}

/* Test if pad rows are ready. If they are, compute their convolution and send/signal their completion */
void node_polling(uint8_t pos, uint iter, struct local_data* ld, struct node_data* nd, long_long** meas) {
  const uint8_t iter_even = !(iter % 2);                // If current iteration is even or odd
  uint8_t* rtw = &ld->rows_to_wait[pos][iter_even];     // pos: TOP or BOTTOM; iter_even: previous iteration
  uint8_t* rta = &ld->rows_to_assert[pos][!iter_even];  // pos: TOP or BOTTOM; !iter_even: current iteration
  long_long *handler_mutex_wait_time = meas[1];
  long_long tmp;
  
  // Check if MPI receive requests have been completed
  if(!nd->recv_completed[pos]) {
    int indexes[SIM_RECV], outcount;
    MPI_Testsome(SIM_RECV, nd->requests, &outcount, indexes, nd->statuses);
    for(int i = 0; i < outcount; i++) nd->recv_completed[indexes[i]] = 1;
  }

  // If MPI requests have been completed, check if local neighbour has computed its bordering rows
  if(nd->recv_completed[pos]) {
    tmp = PAPI_get_real_usec();
    pthread_mutex_lock(&(ld->neigh[pos]->mutex));
    *handler_mutex_wait_time += PAPI_get_real_usec() - tmp;
    if(*rtw) {
      *rtw = 0;
      ld->completed[pos] = 1;
    }
    pthread_mutex_unlock(&(ld->neigh[pos]->mutex));
  }

  // Return if previous checks fails, else compute convolution and exchange pad
  if(!ld->completed[pos]) return; 

  float *my_new_grid, *my_old_grid;
  if(!iter_even) {
    my_new_grid = old_grid;
    my_old_grid = new_grid;
  } else {
    my_new_grid = new_grid;
    my_old_grid = old_grid;
  }
  conv_subgrid(my_old_grid, my_new_grid, nd->send_position[pos], nd->send_position[pos] + pad_elems);
  nd->recv_completed[pos] = 0;
  
  // Return if there isn't a next convolution iteration
  if(iter+1 == num_iterations) return;
  const uint8_t send_offset = SEND_OFFSET(pos) + !iter_even;
  tmp = PAPI_get_real_usec();
  pthread_mutex_lock(&(ld->self->mutex));
  *handler_mutex_wait_time += PAPI_get_real_usec() - tmp;
  *rta = 1;
  pthread_cond_broadcast(&(ld->self->pad_ready));
  pthread_mutex_unlock(&(ld->self->mutex));
  MPI_Isend(&my_new_grid[nd->send_position[pos]], pad_elems, MPI_FLOAT, nd->neighbour[pos], 0, MPI_COMM_WORLD, &nd->requests[send_offset]);
  MPI_Irecv(&my_new_grid[nd->recv_position[pos]], pad_elems, MPI_FLOAT, nd->neighbour[pos], 0, MPI_COMM_WORLD, &nd->requests[pos]);
}

/* Check what coordinator thread has to wait */
void wait_protocol(uint iter, struct local_data* ld, struct node_data* nd, long_long** meas) {
  const uint8_t iter_even = !(iter % 2);                // If current iteration is even or odd
  long_long *condition_wait_time = meas[0];
  long_long *handler_mutex_wait_time = meas[1];
  long_long tmp;
  float *my_new_grid, *my_old_grid;
  
  if(!iter_even) {
    my_new_grid = old_grid;
    my_old_grid = new_grid;
  } else {
    my_new_grid = new_grid;
    my_old_grid = old_grid;
  }
  
  for(uint8_t pos = TOP; pos <= BOTTOM && ld->neigh[pos]; pos++) {
    uint8_t* rtw = &ld->rows_to_wait[pos][iter_even];     // pos: TOP or BOTTOM; iter_even: previous iteration
    uint8_t* rta = &ld->rows_to_assert[pos][!iter_even];  // pos: TOP or BOTTOM; !iter_even: current iteration
    uint8_t send_offset = SEND_OFFSET(pos) + !iter_even;
    
    if(!nd->recv_completed[pos] || ld->completed[pos]) continue;
    
    tmp = PAPI_get_real_usec();
    pthread_mutex_lock(&(ld->neigh[pos]->mutex));
    *handler_mutex_wait_time += PAPI_get_real_usec() - tmp;

    tmp = PAPI_get_real_usec();
    while(!(*rtw))
      pthread_cond_wait(&(ld->neigh[pos]->pad_ready), &(ld->neigh[pos]->mutex));
    *condition_wait_time += PAPI_get_real_usec() - tmp;
    *rtw = 0;
    pthread_mutex_unlock(&(ld->neigh[pos]->mutex));
    ld->completed[pos] = 1;

    conv_subgrid(my_old_grid, my_new_grid, nd->send_position[pos], nd->send_position[pos] + pad_elems);
    nd->recv_completed[pos] = 0;

    if(iter+1 == num_iterations) continue;
    tmp = PAPI_get_real_usec();
    pthread_mutex_lock(&(ld->self->mutex));
    *handler_mutex_wait_time += PAPI_get_real_usec() - tmp;
    *rta = 1;
    pthread_cond_broadcast(&(ld->self->pad_ready));
    pthread_mutex_unlock(&(ld->self->mutex));
    MPI_Isend(&my_new_grid[nd->send_position[pos]], pad_elems, MPI_FLOAT, nd->neighbour[pos], 0, MPI_COMM_WORLD, &nd->requests[send_offset]);
    MPI_Irecv(&my_new_grid[nd->recv_position[pos]], pad_elems, MPI_FLOAT, nd->neighbour[pos], 0, MPI_COMM_WORLD, &nd->requests[pos]);
  }
  
  if(ld->completed[TOP] && ld->completed[BOTTOM]) return;
  int indexes[SIM_RECV], outcount;
  MPI_Waitsome(SIM_RECV, nd->requests, &outcount, indexes, nd->statuses);
  for(int i = 0; i < outcount; i++) nd->recv_completed[indexes[i]] = 1;
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

    /* Convolution
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
    } else {*/
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
   // }

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
 * distribute those additional rows in the load_balancer submatrix.
 * 
 * If the number of processes is greater than 1, the main thread must execute MPI call to exchange pads. 
 * Because of this, main thread will have a variable workload. Hence, main thread works in the LB area 
 * if number of procs is greater than one; else will have a static area like any other worker thraed. 
*/
void initialize_thread_coordinates(struct thread_handler* handler) {
  int submatrix_id;
  uint nrows_per_thread;
  const uint num_workers = num_threads - (num_procs > 1 ? 1 : 0);

  if(handler->tid != -1) {
    submatrix_id = handler->tid;
    if(num_procs == 1 && handler->tid >= num_threads/2) submatrix_id++;
    nrows_per_thread = static_work_nrows / num_workers;
  } else {
    // Load balancer (tid == -1) it's placed in the center of the process matrix
    submatrix_id = num_threads/2;
    lb.nrows = num_threads;
    if (lb.nrows < pad_nrows * 2) lb.nrows += (pad_nrows * 2);

    // Compute the amount of rows assigned to load balancer (including remainder rows and main thread)
    uint rem_rows = (proc_nrows - lb.nrows) % num_threads;
    nrows_per_thread = (proc_nrows - lb.nrows) / num_threads;
    lb.nrows += rem_rows;
    if(num_procs > 1) {
      const uint offload = (pad_nrows*2 > num_threads) ? pad_nrows*2 : 0;
      coordinator_nrows = nrows_per_thread - offload;
      lb.nrows += coordinator_nrows;
    }

    // Worker threads will not work (initially) at load balancer rows. Also, only main thread computes mpi pads
    static_work_nrows = proc_nrows - lb.nrows;
    if(num_procs > 1) { 
      static_work_nrows -= pad_nrows;
      if(rank > 0 && rank < num_procs-1) static_work_nrows -= pad_nrows;
    }
    nrows_per_thread = static_work_nrows / num_workers;
    rem_rows = static_work_nrows % num_workers;

    static_work_nrows -= rem_rows;
    lb.nrows += rem_rows;
    lb.size = lb.nrows * grid_width;
  }

  // Initialize coordinates
  const uint fixed_size = nrows_per_thread * grid_width;
  handler->start = pad_elems + submatrix_id * fixed_size;
  uint actual_size; 
  if (handler->tid == -1) {
    actual_size = lb.size;
  } else {
    actual_size = fixed_size;
    if(submatrix_id > num_threads/2) 
      handler->start = handler->start - fixed_size + lb.size;
  }

  if(rank) handler->start += pad_elems;
  handler->end = handler->start + actual_size;
}
