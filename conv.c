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
#define RESULT_FILE_PATH "./io-files/result.bin"
#define VEC_SIZE 4
#define MEAN_VALUE 0
#define SIM_REQS 6                    // Per-process simultaneous MPI requests
#define ROWS_BEFORE_POLLING 2         // How many rows must be computed before polling the neighbours
#define MAIN_THREAD_OFFLOAD 2         // Will be subtracted at main thread number of rows

void* main_thread(void*);
void* worker_thread(void*);
void* init_vars(void*);
void test_and_update(uint8_t, uint8_t, int*, struct mpi_args*, struct thread_handler*, long_long**);
void load_balancing(int, uint8_t, long_long**);
void conv_subgrid(float*, float*, int, int);
void read_kernel(FILE*);
void get_process_additional_row(struct proc_info*);
void initialize_thread_coordinates(struct thread_handler*);

float *kernel;                        // Kernel used for convolution
float *new_grid;                      // Input/Result grid, swapped at every iteration
float *old_grid;                      // Input/Result grid, swapped at every iteration
float kern_dot_sum;                   // Used for normalization, its value is equal to: sum(dot(kernel, kernel))
uint8_t affinity;                     // If thread affinity (cpu pinning) should be set
uint32_t kern_width;                  // Number of elements in one kernel matrix row
uint32_t grid_width;                  // Number of elements in one grid matrix row
uint num_pads;                        // Number of rows that should be shared with other processes
uint pad_elems;                       // Number of elements in the pad section of the grid matrix
uint kern_elems;                      // Number of elements in whole kernel matrix
uint grid_elems;                      // Number of elements in whole grid matrix
uint proc_num_rows;                   // Number of rows assigned to a process
uint proc_static_work_rows;           // Number of rows not owned by load_balancer
// Load balancing global variables
struct thread_handler* load_balancer; // Used by worker threads to do some additional work if they end earlier. Not an active thread 
pthread_mutex_t mutex_lb;             // Used to access at shared variable of the load balancing
pthread_cond_t lb_iter_completed;     // In some cases a thread could access to load_balancer while previous lb_iter was not completed
uint lb_iter;                         // Used in load balancing to track current iteration
uint lb_curr_start;                   // To track how many load balancer rows have been reserved (but not yet computed)
uint lb_rows_completed;               // To track how many load balancer rows have been computed
uint lb_top_pad, lb_bot_pad;          // To track how many load balancer pad rows have been computed
uint lb_size;                         // Number of elements in load balancer submatrix
uint lb_num_rows;                     // Number of rows in load balancer submatrix (i.e. handled dynamically for load balancing)
// MPI, Pthreads, and PSMID global variables
struct setup_args setup;              // Used by setup, worker, and main threads to synchronize about grid read
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
  long_long read_time;                // Measuring disk I/O times
  FILE *fp_grid, *fp_kernel;          // I/O files for grid and kernel matrices

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
  
  pthread_t threads[num_threads];                             // Worker threads except the middle one (num_threads/2) used as setup thread while main is reading from disk
  struct proc_info procs_info[(rank) ? 1 : num_procs];
  setup.threads = threads;
  setup.procs_info = procs_info;
  load_balancer = &(struct thread_handler){0};                // Load balancer is not an active thread, this structure is used by all workers
  memset(setup.flags, 0, sizeof(uint8_t) * (SEND_INFO+1));    // Used as flag, will be different from 0 only after the grid is ready to be convoluted
  pthread_mutex_init(&(setup.mutex), NULL);                   // Used by io_thread and main thread to synchronize
  pthread_cond_init(&(setup.cond), NULL);

  if(!rank) {
    // Opening input files
    if((fp_grid = fopen(GRID_FILE_PATH, "rb")) == NULL) {
      fprintf(stderr, "Error while opening grid file\n");
      exit(-1);
    }
    if((fp_kernel = fopen(KERNEL_FILE_PATH, "rb")) == NULL) {
      fprintf(stderr, "Error while opening kernel file\n");
      exit(-1);
    }

    // First token represent matrix dimension
    read_time = PAPI_get_real_usec();
    if(fread(&grid_width, sizeof(uint32_t), 1, fp_grid) != 1 || fread(&kern_width, sizeof(uint32_t), 1, fp_kernel) != 1) {
      fprintf(stderr, "Error in file reading: first element should be the row (or column) length of a square matrix\n");
      exit(-1);
    }
    read_time = PAPI_get_real_usec() - read_time;

    // Exchange initial information 
    if(num_procs > 1) {
      MPI_Request info_reqs[num_procs-1];
      uint32_t to_send[] = {grid_width, kern_width};
      for (int p = 1; p < num_procs; p++)
        MPI_Isend(to_send, 2, MPI_UINT32_T, p, p, MPI_COMM_WORLD, &info_reqs[p-1]);
    }

    // Exchange kernel
    kern_elems = kern_width * kern_width;
    long_long t = PAPI_get_real_usec();
    read_kernel(fp_kernel);
    read_time += PAPI_get_real_usec() - t;
    if(num_procs > 1) {
      MPI_Request kern_reqs[num_procs-1];
      for (int p = 1; p < num_procs; p++)
        MPI_Isend(kernel, kern_elems, MPI_FLOAT, p, p, MPI_COMM_WORLD, &kern_reqs[p-1]);
    }

    // Start grid read. Rank 0 has the whole file in memory, other ranks have only the part they are interested in
    num_pads = (kern_width - 1) / 2;
    old_grid = malloc((grid_width + num_pads*2) * grid_width * sizeof(float));
    if ((rc = pthread_create(&threads[num_threads/2], NULL, init_vars, NULL))) { 
      fprintf(stderr, "Error while creating the setup thread; Return code: %d\n", rc);
      exit(-1);
    }

    // Read grid and send its partitions to the MPI nodes
    int float_read = 0;
    read_time = PAPI_get_real_usec();  
    do float_read += fread(&old_grid[pad_elems], sizeof(float), grid_elems, fp_grid);
    while (!(feof(fp_grid) | ferror(fp_grid)));
    read_time = PAPI_get_real_usec() - read_time;
    if(float_read < grid_elems) {
      fprintf(stderr, "Error in file reading: number of float elements read (%d) is lower than the expected amount (%d)\n", float_read, grid_elems);
      exit(-1);
    }
    fclose(fp_grid);

    // Signal that grid read have been completed and check if processes info are ready
    pthread_mutex_lock(&(setup.mutex));
    setup.flags[GRID] = 1;
    pthread_cond_broadcast(&(setup.cond));
    while(!setup.flags[SEND_INFO])
      pthread_cond_wait(&(setup.cond), &(setup.mutex));
    pthread_mutex_unlock(&(setup.mutex));

    // Grid distribution
    if(num_procs > 1) {
      MPI_Request grid_scatter_reqs[num_procs-1];
      for(int p = 1; p < num_procs; p++) {
        MPI_Isend(&old_grid[setup.procs_info[p].start], setup.procs_info[p].size, MPI_FLOAT, p, p, MPI_COMM_WORLD, &grid_scatter_reqs[p-1]);
      }
    }

    // Info about result gathering from previous scattering data. Pads are excluded
    if(num_procs > 1) {
      for(int p = 1; p < num_procs; p++) {
        setup.procs_info[p].start += num_pads * grid_width;            // Starting position final recv
        setup.procs_info[p].size -= num_pads * 2 * grid_width;         // Payload size for final recv
        if(p == num_procs-1) setup.procs_info[p].size += grid_width * num_pads;
      }
    }
  } else {
    uint32_t to_recv[2];
    MPI_Recv(to_recv, 2, MPI_UINT32_T, 0, rank, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    grid_width = to_recv[0];
    kern_width = to_recv[1];
    kern_elems = kern_width * kern_width;
    kernel = malloc(sizeof(float) * kern_elems);
    MPI_Irecv(kernel, kern_elems, MPI_FLOAT, 0, rank, MPI_COMM_WORLD, &(setup.requests[0]));    
    init_vars(NULL);
  }

  pthread_mutex_lock(&(setup.mutex));
  while(!setup.flags[HANDLERS]) 
    pthread_cond_wait(&(setup.cond), &(setup.mutex));
  pthread_mutex_unlock(&(setup.mutex));

  // If mpi is needed, main thread is placed around the center of the matrix and works with load_balancer rows
  if(num_procs > 1) 
    main_thread((void*) &setup.handlers[num_threads/2]);
  else 
    worker_thread((void*) &setup.handlers[num_threads/2]);

  // Check if 'setup_thread' has exited and start recv listeners to gather results
  MPI_Request res_gather_reqs[num_procs-1];
  float *res_grid = (num_iterations % 2) ? new_grid : old_grid;
  if(!rank && num_procs > 1) {
    for(int p = 1; p < num_procs; p++)
      MPI_Irecv(&res_grid[procs_info[p].start], procs_info[p].size, MPI_FLOAT, p, p, MPI_COMM_WORLD, &res_gather_reqs[p-1]);
  }

  // Wait workers termination and collect thread return values
  if(!rank && pthread_join(threads[num_threads/2], (void*) &rc)) {
    fprintf(stderr, "Join error, setup thread exited with: %d", rc);
  }
  for(int i = 0; i < num_threads; i++) {
    if(i == num_threads/2) continue;
    if(pthread_join(threads[i], (void*) &rc)) 
      fprintf(stderr, "Join error, thread[%d] exited with: %d", i, rc);
  }

  if(rank) {
    MPI_Send(&res_grid[pad_elems], proc_num_rows*grid_width, MPI_FLOAT, 0, rank, MPI_COMM_WORLD);
  } else if(num_procs > 1) {
    MPI_Status statuses[num_procs-1];
    MPI_Waitall(num_procs-1, res_gather_reqs, statuses);
  }

  // Print measures
  time_stop = PAPI_get_real_usec();
  printf("Rank[%d] | Elapsed time: %lld us\n", rank, (time_stop - time_start));
  if(!rank) printf("Rank[0] | Elapsed time to read from disk: %lld us\n", read_time);

  long_long* curr_meas;
  for(int i = 0; i < num_threads; i++) {
    curr_meas = thread_measures[i];
    printf("Thread[%d][%d]: Elapsed: %llu us | Condition WT: %llu us | Handlers mutex WT: %llu us | MPI mutex WT: %llu us | LB mutex WT: %llu us | Total L2 cache misses: %lld\n", 
      rank, i, curr_meas[0], curr_meas[1], curr_meas[2], curr_meas[3], curr_meas[4], curr_meas[5]);
    free(curr_meas);
  }

  // Store computed matrix for debug purposes. Rank 0 has the whole matrix file in memory, other ranks have only the part they are interested in 
  FILE *fp_result = NULL;
  if (!rank && (fp_result = fopen(RESULT_FILE_PATH, "wb")) != NULL) {
    int float_written = fwrite(&res_grid[pad_elems], sizeof(float), grid_elems, fp_result);
    if(float_written != grid_elems) {
      fprintf(stderr, "Error in file writing: number of float grid elements written (%d) is different from the expected amount (%d)\n", float_written, grid_elems);
      exit(-1);
    }
    fclose(fp_result);
    save_txt(res_grid);
  }

  // Destroy pthread objects and free all used resources
  pthread_mutex_destroy(&mutex_lb);
  pthread_mutex_destroy(&setup.mutex);
  pthread_cond_destroy(&setup.cond);
  pthread_cond_destroy(&lb_iter_completed);
  for(int i = 0; i < num_threads; i++) {
    pthread_mutex_destroy(&setup.handlers[i].mutex);
    pthread_cond_destroy(&setup.handlers[i].pad_ready);
  }

  MPI_Finalize();
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
void* init_vars(void* args) {
  grid_elems = grid_width * grid_width;
  num_pads = (kern_width - 1) / 2;
  pad_elems = grid_width * num_pads;

  // Rank 0 get info about which processes must compute an additional row, other ranks get info only about themself
  get_process_additional_row(setup.procs_info); 
  const uint min_nrows = (grid_width / num_procs);                            // Minimum amout of rows distributed to each process
  proc_num_rows = min_nrows + setup.procs_info[0].has_additional_row;         // Number of rows assigned to current process
  const uint proc_num_rows_size = proc_num_rows * grid_width;                 // Number of elements assigned to a process

  if(!rank) {
    new_grid = malloc((grid_elems + pad_elems*2) * sizeof(float));
  } else {
    new_grid = malloc((proc_num_rows_size + pad_elems*2) * sizeof(float));
    old_grid = malloc((proc_num_rows_size + pad_elems*2) * sizeof(float));    // Ranks different from 0 has a smaller grid to alloc
  }

  // Rank 0 prepares send/recv info (synch point with io_thread), other ranks receives grid data. 
  if(!rank){
    setup.procs_info[0].start = 0;
    setup.procs_info[0].size = (proc_num_rows + num_pads*2) * grid_width;
    if(num_procs > 1) {
      // Info about data scattering. Pads are included (to avoid an MPI exchange in the first iteration)
      int offset = setup.procs_info[0].has_additional_row;
      for(int i = 1; i < num_procs; i++) {
        setup.procs_info[i].start = (min_nrows * i + offset) * grid_width;                                          // Starting position for Isend
        setup.procs_info[i].size = (min_nrows + num_pads*2 + setup.procs_info[i].has_additional_row) * grid_width;  // Payload size for Isend
        if(i == num_procs-1) setup.procs_info[i].size -= grid_width * num_pads;
        offset += setup.procs_info[i].has_additional_row;
      }
    }
    pthread_mutex_lock(&(setup.mutex));
    setup.flags[SEND_INFO] = 1;
    pthread_cond_signal(&(setup.cond));
    pthread_mutex_unlock(&(setup.mutex));
  } else {
    // Receive grid data
    int recv_size = (proc_num_rows + num_pads * 2) * grid_width;
    if(rank == num_procs-1) recv_size -= grid_width;
    MPI_Irecv(old_grid, recv_size, MPI_FLOAT, 0, rank, MPI_COMM_WORLD, &(setup.requests[1]));
  }

  // Set a zero-pad for lowest and highest process 
  if(!rank){
    memset(new_grid, 0, pad_elems * sizeof(float));
    memset(old_grid, 0, pad_elems * sizeof(float));
  }
  if(rank == num_procs-1) {
    memset(&new_grid[proc_num_rows_size + pad_elems], 0, pad_elems * sizeof(float));
    memset(&old_grid[proc_num_rows_size + pad_elems], 0, pad_elems * sizeof(float));
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
  load_balancer->tid = -1;
  load_balancer->top = &handlers[num_threads/2-1];          // top neighbour is always the same
  handlers[num_threads/2-1].bottom = load_balancer;
  struct thread_handler* mth = &handlers[num_threads/2];    // main thread handler (tid == num_threads/2)

  if(num_procs == 1) {
    load_balancer->bottom = &handlers[num_threads/2];
    handlers[num_threads/2].top = load_balancer;
  } else {                                                 // bottom neighbour is different if num_procs > 1 
    if(num_threads > 2) {
      load_balancer->bottom = &handlers[num_threads/2+1];
      handlers[num_threads/2+1].top = load_balancer;
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
        load_balancer->bottom = NULL;
        mth->bottom = &handlers[0];
      }
    } else {
      if(num_threads == 2){
        load_balancer->bottom = mth;
        mth->top = load_balancer;
      } else {
        handlers[num_threads-1].bottom = mth;
        mth->top = &handlers[num_threads-1];
      }
    }
  }

  /*printf("[%d] Connections and dependencies\n", rank);
  for(int i = 0; i < num_threads; i++) {
    char tid_bot[4]; char tid_top[4];
    char lb_bot[4]; char lb_top[4];
    if(handlers[i].bottom) {
      if(handlers[i].bottom->tid == -1) {tid_bot[0] = 'L'; tid_bot[1] = 'B'; tid_bot[2] = '\0';}
      else sprintf(tid_bot, "%d", handlers[i].bottom->tid);
    }
    if(handlers[i].top) {
      if(handlers[i].top->tid == -1) {tid_top[0] = 'L'; tid_top[1] = 'B'; tid_top[2] = '\0';}
      else sprintf(tid_top, "%d", handlers[i].top->tid);
    }
    if(load_balancer->top) sprintf(lb_top, "%d", load_balancer->top->tid);
    if(load_balancer->bottom) sprintf(lb_bot, "%d", load_balancer->bottom->tid);
    
    printf("[%d][%d] TOP: [%s] | BOTTOM: [%s] | lb->top: %s | lb->bot: %s\n", rank, handlers[i].tid, (handlers[i].top ? tid_top : "NULL"), (handlers[i].bottom ? tid_bot : "NULL"), (load_balancer->top ? lb_top : "NULL"), ( load_balancer->bottom ? lb_bot : "NULL"));
  } 
  const uint8_t mpi_not_needed = (num_procs > 1) & (rank == num_procs-1) & (num_threads == 2); */

  pthread_mutex_init(&mutex_lb, NULL);
  pthread_cond_init(&lb_iter_completed, NULL);
  pthread_mutex_init(&(load_balancer->mutex), NULL);
  pthread_cond_init(&(load_balancer->pad_ready), NULL);
  initialize_thread_coordinates(load_balancer);
  lb_curr_start = load_balancer->start + pad_elems;                 // Start from submatrix with no dependencies
  lb_iter = 0; lb_rows_completed = 0;
  lb_top_pad = 0; lb_bot_pad = 0;
  thread_measures = malloc(sizeof(long_long*) * num_threads);

  // Computation of "last_mask"
  uint32_t rem = kern_width % VEC_SIZE;
  uint32_t to_load[VEC_SIZE];
  memset(to_load, 0, VEC_SIZE * sizeof(uint32_t));
  for(int i = 0; i < rem; i++) to_load[i] = UINT32_MAX;
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
void* main_thread(void* args) {
  struct thread_handler *handler = (struct thread_handler*)args;
  float *my_old_grid = old_grid;
  float *my_new_grid = new_grid;
  float *tmp;                                       // Used only for grid swap
  int completed[CENTER+1];                          // If my top, bottom, or central rows have been completed
  MPI_Request requests[SIM_REQS];                   // There are at most two "Isend" and one "Irecv" not completed at the same time per worker_thread, hence six per process
  int requests_completed[SIM_REQS];                 // Log of the completed mpi requests
  uint8_t prev_odd = 1;                             // If previous iteration index was odd
  //uint8_t changed = 0;
  const int bottom_pad_start = proc_num_rows * grid_width;

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

  for(int i = 0; i < SIM_REQS; i++) requests[i] = MPI_REQUEST_NULL;
  memset(requests_completed, 0, sizeof(int) * SIM_REQS);
  if(handler->top == NULL) completed[TOP] = 1;
  if(handler->bottom == NULL) completed[BOTTOM] = 1;

  // Complete kernel receive and/or compute "sum(dot(kernel, kernel))"
  if(rank) MPI_Wait(&(setup.requests[0]), MPI_STATUS_IGNORE);
  for(int pos = 0; pos < kern_elems; pos++) {
    kern_dot_sum += kernel[pos] * kernel[pos];
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
    MPI_Isend(&my_new_grid[pad_elems], pad_elems, MPI_FLOAT, rank-1, 0, MPI_COMM_WORLD, &requests[0]);
    MPI_Irecv(&my_new_grid[0],         pad_elems, MPI_FLOAT, rank-1, 0, MPI_COMM_WORLD, &requests[2]);
  }
  if(handler->top != NULL) {
    conv_subgrid(my_old_grid, my_new_grid, bottom_pad_start, bottom_pad_start + pad_elems);
    MPI_Isend(&my_new_grid[bottom_pad_start],             pad_elems, MPI_FLOAT, rank+1, 0, MPI_COMM_WORLD, &requests[0 + SIM_REQS/2]);
    MPI_Irecv(&my_new_grid[bottom_pad_start + pad_elems], pad_elems, MPI_FLOAT, rank+1, 0, MPI_COMM_WORLD, &requests[2 + SIM_REQS/2]);
  }

  t = PAPI_get_real_usec();
  pthread_mutex_lock(&(handler->mutex));
  handler_mutex_wait_time += PAPI_get_real_usec() - t;
  handler->top_rows_done[0] = 1;
  handler->bot_rows_done[0] = 1;
  pthread_cond_broadcast(&(handler->pad_ready));
  pthread_mutex_unlock(&(handler->mutex));

  // Complete the first convolution iteration by computing central elements
  if(!lb_iter) load_balancing(0, 0, measures);

  // Second or higher convolution iterations
  int cnt_work, outcount, indexes[SIM_REQS];
  MPI_Status statuses[SIM_REQS];
  for(int iter = 1; iter < num_iterations; iter++) {
    prev_odd = !prev_odd;
    tmp = my_old_grid;
    my_old_grid = my_new_grid;
    my_new_grid = tmp;
    cnt_work = 0;
    completed[CENTER] = 0;
    if(handler->top != NULL) completed[TOP] = 0;
    if(handler->bottom != NULL) completed[BOTTOM] = 0;

    while(!completed[TOP] || !completed[BOTTOM] || !completed[CENTER]) {
      if(!completed[BOTTOM] && handler->bottom != NULL){
        if(!requests_completed[prev_odd] || !requests_completed[2]) {
          MPI_Waitsome(SIM_REQS, requests, &outcount, indexes, statuses);
          if(outcount) for(int i = 0; i < outcount; i++) requests_completed[indexes[i]] = 1;
        }

        if(requests_completed[prev_odd] && requests_completed[2]) {        
          pthread_mutex_lock(&(handler->bottom->mutex));
          while(!handler->bottom->top_rows_done[prev_odd])
            pthread_cond_wait(&(handler->bottom->pad_ready), &(handler->bottom->mutex));
          handler->bottom->top_rows_done[prev_odd] = 0;
          pthread_mutex_unlock(&(handler->bottom->mutex));

          conv_subgrid(my_old_grid, my_new_grid, pad_elems,  pad_elems*2);
          requests_completed[prev_odd] = 0;
          requests_completed[2] = 0;
          completed[BOTTOM] = 1;
          
          if(iter+1 < num_iterations) {
            pthread_mutex_lock(&(handler->mutex));
            handler->bot_rows_done[!prev_odd] = 1;
            pthread_cond_broadcast(&(handler->pad_ready));
            pthread_mutex_unlock(&(handler->mutex));

            MPI_Isend(&my_new_grid[pad_elems], pad_elems, MPI_FLOAT, rank-1, 0, MPI_COMM_WORLD, &requests[!prev_odd]);
            MPI_Irecv(&my_new_grid[0],         pad_elems, MPI_FLOAT, rank-1, 0, MPI_COMM_WORLD, &requests[2]);
          }
        }
      }

      if(!completed[TOP] && handler->top != NULL){
        if(!requests_completed[prev_odd + SIM_REQS/2] || !requests_completed[2 + SIM_REQS/2]) {
          MPI_Waitsome(SIM_REQS, requests, &outcount, indexes, statuses);
          if(outcount) for(int i = 0; i < outcount; i++) requests_completed[indexes[i]] = 1;
        }
        
        if(requests_completed[prev_odd + SIM_REQS/2] && requests_completed[2 + SIM_REQS/2]) {
          pthread_mutex_lock(&(handler->top->mutex));
          while(!handler->top->bot_rows_done[prev_odd])
            pthread_cond_wait(&(handler->top->pad_ready), &(handler->top->mutex));
          handler->top->bot_rows_done[prev_odd] = 0;
          pthread_mutex_unlock(&(handler->top->mutex));

          conv_subgrid(my_old_grid, my_new_grid, bottom_pad_start, bottom_pad_start + pad_elems);
          requests_completed[prev_odd + SIM_REQS/2] = 0;
          requests_completed[2 + SIM_REQS/2] = 0;
          completed[TOP] = 1;
          
          if(iter+1 < num_iterations) {
            pthread_mutex_lock(&(handler->mutex));
            handler->top_rows_done[!prev_odd] = 1;
            pthread_cond_broadcast(&(handler->pad_ready));
            pthread_mutex_unlock(&(handler->mutex));

            MPI_Isend(&my_new_grid[bottom_pad_start],             pad_elems, MPI_FLOAT, rank+1, 0, MPI_COMM_WORLD, &requests[!prev_odd + SIM_REQS/2]);
            MPI_Irecv(&my_new_grid[bottom_pad_start + pad_elems], pad_elems, MPI_FLOAT, rank+1, 0, MPI_COMM_WORLD, &requests[2 + SIM_REQS/2]);
          }
        }
      }

      if(!completed[CENTER]) {
        //temporary solution
        if(iter >= lb_iter) {
          load_balancing(iter, 1, measures);
          cnt_work++;
          if(cnt_work > lb_num_rows/2) completed[CENTER] = 1;
        }
        else completed[CENTER] = 1;
      }
    }

    // Load balancing if this thread ended current iteration earlier
    if(iter >= lb_iter) load_balancing(iter, 0, measures);
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
  int completed[CENTER+1];                          // If my top, bottom, or central rows have been completed
  int center_start;                                 // Center elements are completed one row at a time
  struct mpi_args *margs = &(struct mpi_args){0};   // Pointer to an empty struct, initialized later if this thread needs MPI

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

  t = PAPI_get_real_usec();
  pthread_mutex_lock(&(handler->mutex));
  handler_mutex_wait_time += PAPI_get_real_usec() - t;
  handler->top_rows_done[0] = 1;
  handler->bot_rows_done[0] = 1;
  pthread_cond_broadcast(&(handler->pad_ready));
  pthread_mutex_unlock(&(handler->mutex));

  // Complete the first convolution iteration by computing central elements
  conv_subgrid(my_old_grid, my_new_grid, (handler->start + pad_elems), (handler->end - pad_elems));
  if(!lb_iter) load_balancing(0, 0, measures);

  // Second or higher convolution iterations
  for(int iter = 1; iter < num_iterations; iter++) {
    tmp = my_old_grid;
    my_old_grid = my_new_grid;
    my_new_grid = tmp;
    center_start = handler->start + pad_elems;
    memset(completed, 0, sizeof(int) * (CENTER+1));

    while(!completed[TOP] || !completed[BOTTOM] || !completed[CENTER]) {
      if(!completed[TOP]) {
        test_and_update(TOP, iter, completed, margs, handler, measures);
      }

      if(!completed[BOTTOM]) {
        test_and_update(BOTTOM, iter, completed, margs, handler, measures);
      }

      // Computing central rows one at a time if top and bottom rows are incomplete
      if(!completed[CENTER]) {
        uint center_end;
        uint actual_end = handler->end - pad_elems;
        if (completed[TOP] && completed[BOTTOM]) {
          center_end = actual_end;
          completed[CENTER] = 1;
        } else {
          center_end = center_start + grid_width * ROWS_BEFORE_POLLING;
          if(center_end > actual_end) center_end = actual_end;
        }

        conv_subgrid(my_old_grid, my_new_grid, center_start, center_end);
        if(center_end == actual_end) completed[CENTER] = 1;
        else center_start += (grid_width * ROWS_BEFORE_POLLING);
      }
    }

    // Load balancing if this thread ended current iteration earlier
    if(iter >= lb_iter) load_balancing(iter, 0, measures);
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
  const uint8_t prev_odd = (!iter) ? 1 : (iter-1) % 2;
  uint start = 0, end = 0;
  long_long t;
  long_long *handlers_mutex_wait_time = meas[1];
  long_long *lb_mutex_wait_time = meas[3];

  float *my_new_grid, *my_old_grid;
  if(!prev_odd) {
    my_new_grid = old_grid;
    my_old_grid = new_grid;
  } else {
    my_new_grid = new_grid;
    my_old_grid = old_grid;
  }

  // Compute a row of the load_balancer shared work 
  while(iter >= lb_iter) {
    start = 0;
    t = PAPI_get_real_usec();
    pthread_mutex_lock(&mutex_lb);
    *lb_mutex_wait_time += PAPI_get_real_usec() - t;
    while(iter > lb_iter)
      pthread_cond_wait(&lb_iter_completed, &mutex_lb);         // Wait if lb work of previous iteration is not completed yet
    if(iter == lb_iter) {
      start = lb_curr_start;
      lb_curr_start += grid_width;                              // From lb->start to lb->end
    }
    pthread_mutex_unlock(&mutex_lb);
    end = start + grid_width;
    if(!start || end > load_balancer->end + pad_elems) return;  // All shared works have been reserved, return to private work

    // If my shared work is in the middle, no dependencies
    if(start >= (load_balancer->start + pad_elems) && end <= (load_balancer->end - pad_elems)) {
      conv_subgrid(my_old_grid, my_new_grid, start, end);
      t = PAPI_get_real_usec();
      pthread_mutex_lock(&mutex_lb);
      *lb_mutex_wait_time += PAPI_get_real_usec() - t;
      lb_rows_completed++;                                      // Track the already computed row
      if(lb_rows_completed == lb_num_rows) {                    // All shared works have been completed
        lb_rows_completed = 0;
        lb_curr_start = load_balancer->start + pad_elems;
        lb_iter++;
        pthread_cond_broadcast(&lb_iter_completed);
      }
      pthread_mutex_unlock(&mutex_lb);
      if(single_work) return;
      continue;
    }

    // If only the top matrix portion is left
    if(end > load_balancer->end) {
      start -= lb_size;
      end -= lb_size;
    }

    // Dependencies handling
    struct thread_handler* neigh_handler = NULL;
    uint8_t *rows_to_wait, *rows_to_assert;
    uint* pad_counter;
    if(end <= load_balancer->start + pad_elems) {           // Top pad
      neigh_handler = load_balancer->top;
      rows_to_wait = load_balancer->top->bot_rows_done;
      rows_to_assert = load_balancer->top_rows_done;
      pad_counter = &lb_top_pad;
    } else if(load_balancer->bottom != NULL){               // Bottom pad
      neigh_handler = load_balancer->bottom;
      rows_to_wait = load_balancer->bottom->top_rows_done;
      rows_to_assert = load_balancer->bot_rows_done;
      pad_counter = &lb_bot_pad;
    }

    // Wait if neighbours are late
    if(iter > 0 && neigh_handler != NULL) {
      t = PAPI_get_real_usec();
      pthread_mutex_lock(&(neigh_handler->mutex));
      *handlers_mutex_wait_time += PAPI_get_real_usec() - t;
      while(!rows_to_wait[prev_odd])
        pthread_cond_wait(&(neigh_handler->pad_ready), &(neigh_handler->mutex));
      pthread_mutex_unlock(&(neigh_handler->mutex));
    }
    conv_subgrid(my_old_grid, my_new_grid, start, end);

    // Track pad completion and signal neighbour thread
    if(iter+1 < num_iterations && neigh_handler != NULL) {
      t = PAPI_get_real_usec();
      pthread_mutex_lock(&(load_balancer->mutex));
      *handlers_mutex_wait_time += PAPI_get_real_usec() - t;
      (*pad_counter)++;
      if(*pad_counter == num_pads) {
        rows_to_wait[prev_odd] = 0;
        rows_to_assert[!prev_odd] = 1;
        *pad_counter = 0;
        pthread_cond_broadcast(&(load_balancer->pad_ready));
      }
      pthread_mutex_unlock(&(load_balancer->mutex));
    }

    // Track row completion
    t = PAPI_get_real_usec();
    pthread_mutex_lock(&mutex_lb);
    *lb_mutex_wait_time += PAPI_get_real_usec() - t;
    lb_rows_completed++;                                      // Track the already computed row
    if(lb_rows_completed == lb_num_rows) {                    // All shared works have been completed
      lb_rows_completed = 0;
      lb_curr_start = load_balancer->start + pad_elems;
      lb_iter++;
      pthread_cond_broadcast(&lb_iter_completed);
    }
    pthread_mutex_unlock(&mutex_lb);
    if(single_work) return;
  }
}

/* Test if pad rows are ready. If they are, compute their convolution and send/signal their completion */
void test_and_update(uint8_t position, uint8_t iter, int* completed, struct mpi_args* margs, struct thread_handler* handler, long_long** meas) {
  uint8_t prev_odd = (iter-1) % 2;                // If previous iteration was odd or even
  uint8_t *rows_to_wait, *rows_to_assert;         // Pointers of flags to wait and signal
  struct thread_handler* neigh_handler;           // Thread handler of the neighbour thread (tid +- 1)
  long_long *condition_wait_time = meas[0];
  long_long t;

  switch(position) {
    case TOP:
      neigh_handler = handler->top;
      rows_to_assert = handler->top_rows_done;
      if(neigh_handler == NULL) break;
      rows_to_wait = handler->top->bot_rows_done;
      break;

    case BOTTOM:
      neigh_handler = handler->bottom;
      rows_to_assert = handler->bot_rows_done;
      if(neigh_handler == NULL) break;
      rows_to_wait = handler->bottom->top_rows_done;
      break;

    default:
      return;
  }

  if(neigh_handler == NULL) {                                 // No upper or lower dependency
    completed[position] = 1;
  } else {                                                    // Upper or lower shared memory dependency
    if(!completed[CENTER] || !completed[!position]) {         // If there is still other work to do, trylock insted of lock
      if(!pthread_mutex_trylock(&(neigh_handler->mutex))) {
        if(rows_to_wait[prev_odd]) {
          rows_to_wait[prev_odd] = 0;
          completed[position] = 1;
        }
        pthread_mutex_unlock(&(neigh_handler->mutex));
      }
    } else {                                                  // Else, lock and wait
      long_long *handler_mutex_wait_time = meas[1];
      
      t = PAPI_get_real_usec();
      pthread_mutex_lock(&(neigh_handler->mutex));
      *handler_mutex_wait_time += PAPI_get_real_usec() - t;
      t = PAPI_get_real_usec();
      while(!rows_to_wait[prev_odd])
        pthread_cond_wait(&(neigh_handler->pad_ready), &(neigh_handler->mutex));
      *condition_wait_time += PAPI_get_real_usec() - t;
      rows_to_wait[prev_odd] = 0;
      pthread_mutex_unlock(&(neigh_handler->mutex));
      completed[position] = 1;
    }
  }

  
  // If test was successful, compute convolution of the tested part
  if(!completed[position]) return;
  long_long *handler_mutex_wait_time = meas[1];
  float *my_new_grid, *my_old_grid;
  if(!prev_odd) {
    my_new_grid = old_grid;
    my_old_grid = new_grid;
  } else {
    my_new_grid = new_grid;
    my_old_grid = old_grid;
  }
  
  int start = (position == TOP) ? handler->start : handler->end - pad_elems;
  conv_subgrid(my_old_grid, my_new_grid, start, (start + pad_elems));

  // Signal/Send pad completion if a next convolution iteration exists
  if(iter+1 == num_iterations) return;
  t = PAPI_get_real_usec();
  pthread_mutex_lock(&(handler->mutex));
  handler_mutex_wait_time += PAPI_get_real_usec() - t;
  rows_to_assert[!prev_odd] = 1;
  pthread_cond_broadcast(&(handler->pad_ready));
  pthread_mutex_unlock(&(handler->mutex));
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
  __m128 vec_grid, vec_kern, vec_temp;     // Floating point vector for grid and kernel (plus a temp vector)
  __m128 vec_rslt;                         // Floating point vector of result, will be reduced at the end
  __m128 vec_mxds;                         // Floating point vector of matrix dot sum, will be reduced at the end

  for(int i = start_index; i < end_index; i++) {
    // Setting indexes for current element
    if(col < num_pads) {
      for(offset = 0; i-offset > row_start && offset <= num_pads; offset++);
      grid_index = i-offset-pad_elems;
      kern_index = (kern_width / 2) - offset;
      kern_end = kern_width-kern_index;
      iterations = (num_pads+col+1) *kern_width;
    } else if (col > grid_width-1-num_pads){
      int row_end = row_start + grid_width - 1;
      for(offset = 0; i+offset <= row_end && offset <= num_pads; offset++);
      grid_index = i-num_pads-pad_elems;
      kern_index = 0;
      kern_end = kern_width-offset;
      iterations = (num_pads + grid_width-1-col) * kern_width;
    } else {
      grid_index = i-num_pads-pad_elems;
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
          } else {
            vec_grid = _mm_maskload_ps(&sub_grid[grid_index+offset], last_mask);
          }
          vec_kern = _mm_loadu_ps(&kernel[kern_index+offset]);
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

/* Read kernel from file and compute the chars-to-floats conversion */
void read_kernel(FILE *fp_kernel){
  kernel = malloc(sizeof(float) * kern_elems);
  const int float_read = fread(kernel, sizeof(float), kern_elems, fp_kernel);
  fclose(fp_kernel);

  if(float_read < kern_elems) {
    fprintf(stderr, "Error in file reading: number of float kernel elements read (%d) is lower than the expected amount (%d)\n", float_read, kern_elems);
    exit(-1);
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
  int rows_per_thread;
  const int num_workers = num_threads - (num_procs > 1 ? 1 : 0);

  if(handler->tid != -1) {
    submatrix_id = handler->tid;
    if(num_procs == 1 && handler->tid >= num_threads/2) submatrix_id++;
    rows_per_thread = proc_static_work_rows / num_workers;
  } else {
    // Load balancer (tid == -1) it's placed in the center of the process matrix
    submatrix_id = num_threads/2;
    lb_num_rows = num_threads;
    if (lb_num_rows < num_pads * 2) lb_num_rows += (num_pads * 2);

    // Compute the amount of rows assigned to load balancer (including remainder rows and main thread)
    int rem_rows = (proc_num_rows - lb_num_rows) % num_threads;
    rows_per_thread = (proc_num_rows - lb_num_rows) / num_threads;
    lb_num_rows += rem_rows;
    if(num_procs > 1) {
      int main_thread_rows = rows_per_thread - num_pads*2 - MAIN_THREAD_OFFLOAD;
      lb_num_rows += main_thread_rows;
    }

    // Worker threads will not work (initially) at load balancer rows. Also, only main thread computes mpi pads
    proc_static_work_rows = proc_num_rows - lb_num_rows;
    if(num_procs > 1) { 
      proc_static_work_rows -= num_pads;
      if(rank > 0 && rank < num_procs-1) proc_static_work_rows -= num_pads;
    }
    rows_per_thread = proc_static_work_rows / num_workers;
    rem_rows = proc_static_work_rows % num_workers;
    
    proc_static_work_rows -= rem_rows;
    lb_num_rows += rem_rows;
    lb_size = lb_num_rows * grid_width;
  }

  // Initialize coordinates
  const int fixed_size = rows_per_thread * grid_width;
  handler->start = pad_elems + submatrix_id * fixed_size;
  int actual_size; 
  if (handler->tid == -1) {
    actual_size = lb_size;
  } else {
    actual_size = fixed_size;
    if(submatrix_id > num_threads/2) 
      handler->start = handler->start - fixed_size + lb_size;
  }

  if(rank) handler->start += pad_elems;
  handler->end = handler->start + actual_size; 
}
