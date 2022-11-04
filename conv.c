// conv.c
// Name: Tanay Agarwal, Nirmal Krishnan
// JHED: tagarwa2, nkrishn9

#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <papi.h>
#include <mpi.h>
#include <pthread.h>
#include <unistd.h>
#include <immintrin.h>

#define DEFAULT_ITERATIONS 1
#define DEFAULT_THREADS 2
#define GRID_FILE_PATH "./io-files/grids/haring.bin"
#define KERNEL_FILE_PATH "./io-files/kernels/ridge.bin"
#define RESULT_FILE_PATH "./io-files/result.bin"
#define NSEC_WAIT (5*1000)
#define TOP 0
#define BOTTOM 1
#define CENTER 2
#define EXP_CHARS 13                  // Standard "%+e" format has this num of chars (e.g. -9.075626e+20)

struct thread_handler {
  int tid;                            // Custom thread ID, not the one returned by "pthread_self()"
  uint start, end;                    // Matrix area of ​​interest for this thread
  uint8_t top_rows_done[2];           // Flags for curr and next iteration. "True" if this thread has computed its top rows
  uint8_t bot_rows_done[2];           // Flags for curr and next iteration. "True" if this thread has computed its bottom rows
  struct thread_handler* top;         // To exchange information about pads with neighbour threads
  struct thread_handler* bottom;
  pthread_mutex_t mutex;              // Mutex to access this handler
  pthread_cond_t pad_ready;           // Thread will wait if neighbour's top and bottom rows (pads) aren't ready
};

struct proc_info {
  uint8_t has_additional_row;         // If this process must compute an additional row
  uint start, size;                   // Used for initial and final MPI send/recv. Grid start position and payload size
};

struct io_thread_args {
  FILE* fp_grid;                      // File containing input grid
  MPI_Request request;                // Used by rank different from 0 to wait the receiving of grid data
  uint8_t grid_ready;                 // True if the grid is in local memory
  pthread_mutex_t mutex;              // Mutex to access shared variables beetween main thread and io thread
  pthread_cond_t cond;                // Necessary one synchronization point beetween main thread and io thread
  struct proc_info* procs_info;       // Info used for MPI distribution of grid
  long_long* read_time;               // To measure read time of grid file
};

struct mpi_args {
  uint send_position;                 // Grid position of the payload fetched by MPI
  uint recv_position;                 // Grid position where the payload received (through MPI) should be stored
  MPI_Request request[3];             // There are at most two "Isend" and one "Irecv" not completed at the same time
  int neighbour;                      // MPI rank of the neighbour process
  int requests_completed[3];          // Log of the completed mpi requests
};

void* worker_thread(void*);
void* grid_input_thread(void*);
void test_and_update(uint8_t, uint8_t, int*, struct mpi_args*, struct thread_handler*, long_long**);
void conv_subgrid(float*, float*, int, int);
void read_kernel(FILE*);
int stick_this_thread_to_core(int);
void handle_PAPI_error(int, char*);
void get_process_additional_row(struct proc_info*);
void initialize_thread_coordinates(struct thread_handler*);
void parse_subgrid_and_sync(struct thread_handler*);

__m128i mask;                         // Used in "_mm_maskload_ps" to discards last element in contiguous load 
pthread_mutex_t mutex_mpi;            // To call MPI routines in mutual exclusion (will be used only by top and bottom thread)
uint8_t affinity;                     // If thread affinity should be set 
uint8_t num_pads;                     // Number of rows that should be shared with other processes
uint32_t kern_width;                  // Number of elements in one kernel matrix row
uint16_t pad_size;                    // Number of elements in the pad section of the grid matrix
uint32_t grid_width;                  // Number of elements in one grid matrix row
uint32_t kern_size;                   // Number of elements in whole kernel matrix
uint64_t grid_size;                   // Number of elements in whole grid matrix
int proc_assigned_rows;               // Number of rows assigned to a process
int proc_assigned_rows_size;          // Number of elements assigned to a process
int num_procs;                        // Number of MPI processes in the communicator
int num_threads;                      // Number of threads (main included) for every MPI process
int num_iterations;                   // Number of convolution iterations
int rank;                             // MPI process identifier
float kern_dot_sum;                   // Used for normalization, its value is equal to: sum(dot(kernel, kernel))
float *kernel;                        // Kernel buffer
float *grid;                          // Grid buffer
float *old_grid;                      // Old grid buffer
struct io_thread_args io_args;        // Used by io, worker, and main threads to synchronize about grid read
long_long** thread_measures;          // To collect thread measures
const struct timespec WAIT_TIME = {   // Wait time used by MPI threads
  .tv_sec = 0, 
  .tv_nsec = NSEC_WAIT
};

int main(int argc, char** argv) {
  int provided;                       // MPI thread level supported
  int rc;                             // Return code used in error handling
  long_long time_start, time_stop;    // To measure execution time
  long_long read_time, write_time, t; // Measuring disk I/O times
  FILE *fp_grid, *fp_kernel;          // I/O files for grid and kernel matrices
  FILE *fp_result = NULL;             // I/O file for result

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
  MPI_Request requests[num_procs * 2];
  pthread_t threads[num_threads-1];                       // "-1" because the main thread will be reused as worker thread
  pthread_t io_thread;                                    // Used to read grid from disk while main thread works on something else
  struct proc_info procs_info[(rank) ? 1 : num_procs];    // Which process must compute an additional row
  procs_info[num_procs-1].size = 0;                       // Used as flag, will be different from 0 only after the procs info are ready
  io_args.grid_ready = 0;                                 // Used as flag, will be different from 0 only after the grid is ready to be convoluted
  pthread_mutex_init(&(io_args.mutex), NULL);             // Used by io_thread and main thread to synchronize
  pthread_cond_init(&(io_args.cond), NULL);

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
      uint to_send[] = {grid_width, kern_width};
      for (int i = 1; i < num_procs; i++)
        MPI_Isend(to_send, 2, MPI_INT, i, i, MPI_COMM_WORLD, &requests[i]);
    }

    // Exchange kernel
    kern_size = kern_width * kern_width;
    t = PAPI_get_real_usec();
    read_kernel(fp_kernel);
    read_time += PAPI_get_real_usec() - t;
    if(num_procs > 1) {
      for (int i = 1; i < num_procs; i++)
        MPI_Isend(kernel, kern_size, MPI_FLOAT, i, i, MPI_COMM_WORLD, &requests[i + num_procs]);
    }

    // Start grid read. Rank 0 has the whole file in memory, other ranks have only the part they are interested in
    num_pads = (kern_width - 1) / 2;
    old_grid = malloc((grid_width + num_pads*2) * grid_width * sizeof(float));
    io_args.fp_grid = fp_grid;
    io_args.procs_info = procs_info;
    io_args.read_time = &read_time;
    if ((rc = pthread_create(&io_thread, NULL, grid_input_thread, (void*)&io_args))) { 
      fprintf(stderr, "Error while creating pthread 'io_thread'; Return code: %d\n", rc);
      exit(-1);
    }
  } else {
    uint to_recv[2];
    MPI_Recv(to_recv, 2, MPI_INT, 0, rank, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    grid_width = to_recv[0];
    kern_width = to_recv[1];
    kern_size = kern_width * kern_width;
    kernel = malloc(sizeof(float) * kern_size);
    MPI_Irecv(kernel, kern_size, MPI_FLOAT, 0, rank, MPI_COMM_WORLD, &requests[0]);
  }

  // Variable initialization and data splitting
  pthread_mutex_init(&mutex_mpi, NULL);
  thread_measures = malloc(sizeof(long_long*) * num_threads);
  mask = _mm_set_epi32(0, UINT32_MAX, UINT32_MAX, UINT32_MAX);
  grid_size = grid_width * grid_width;
  num_pads = (kern_width - 1) / 2;
  pad_size = grid_width * num_pads;

  // Rank 0 get info about which processes must compute an additional row, other ranks get info only about themself
  get_process_additional_row(procs_info); 

  const int fixed_rows_per_proc = (grid_width / num_procs);                          // Minimum amout of rows distributed to each process
  proc_assigned_rows = fixed_rows_per_proc + procs_info[0].has_additional_row;       // Number of rows assigned to current process
  proc_assigned_rows_size = proc_assigned_rows * grid_width;                         // Number of elements assigned to a process

  if(!rank) {
    grid = malloc((grid_size + pad_size*2) * sizeof(float));
  } else {
    grid = malloc((proc_assigned_rows_size + pad_size*2) * sizeof(float));
    old_grid = malloc((proc_assigned_rows_size + pad_size*2) * sizeof(float));       // Ranks different from 0 has a smaller grid to alloc
  }

  // Rank 0 prepares send/recv info (synch point with io_thread), other ranks receives grid data. 
  if(!rank){
    pthread_mutex_lock(&(io_args.mutex));
    procs_info[0].start = 0;
    procs_info[0].size = (proc_assigned_rows + num_pads*2) * grid_width;
    if(num_procs > 1) {
      // Info about data scattering. Pads are included (to avoid an MPI exchange in the first iteration)
      int offset = procs_info[0].has_additional_row;
      for(int i = 1; i < num_procs; i++) {
        procs_info[i].start = (fixed_rows_per_proc * i + offset) * grid_width;                                    // Starting position for Isend
        procs_info[i].size = (fixed_rows_per_proc + num_pads*2 + procs_info[i].has_additional_row) * grid_width;  // Payload size for Isend
        if(i == num_procs-1) procs_info[i].size -= grid_width * num_pads;
        offset += procs_info[i].has_additional_row;
      }
    }
    pthread_cond_signal(&(io_args.cond));
    pthread_mutex_unlock(&(io_args.mutex));
  } else {
    // Receive grid data
    int recv_size = (proc_assigned_rows + num_pads * 2) * grid_width;
    if(rank == num_procs-1) recv_size -= grid_width;
    MPI_Irecv(old_grid, recv_size, MPI_FLOAT, 0, rank, MPI_COMM_WORLD, &(io_args.request));
  }

  // Prepare output file (used later) while io_thread is blocked
  if(!rank && (fp_result = fopen(RESULT_FILE_PATH, "wb")) == NULL) {
    fprintf(stderr, "Error while creating and/or opening result file\n");
    exit(-1);
  }

  // Set a zero-pad for lowest and highest process 
  if(!rank){
    memset(grid, 0, pad_size * sizeof(float));
    memset(old_grid, 0, pad_size * sizeof(float));
  }
  if(rank == num_procs-1) {
    memset(&grid[proc_assigned_rows_size + pad_size], 0, pad_size * sizeof(float));
    memset(&old_grid[proc_assigned_rows_size + pad_size], 0, pad_size * sizeof(float));
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

  // Complete kernel receive and compute "sum(dot(kernel, kernel))"
  if(num_procs > 1 && rank) 
    MPI_Wait(requests, MPI_STATUS_IGNORE);
  for(int pos = 0; pos < kern_size; pos++) {
    kern_dot_sum += kernel[pos] * kernel[pos];
  }

  // PThreads creation
  for(int i = 0; i < num_threads-1; i++) {
    rc = pthread_create(&threads[i], NULL, worker_thread, (void*)&handlers[i]);
    if (rc) { 
      fprintf(stderr, "Error while creating pthread 'worker_thread[%d]'; Return code: %d\n", i, rc);
      exit(-1);
    }
  }
  worker_thread((void*) &handlers[num_threads-1]);   // Main thread is the bottom thread

  // Gather results and terminate MPI execution environment
  float *res_grid = (num_iterations % 2) ? grid : old_grid;
  if(!rank && num_procs > 1) {
    for(int k = 1; k < num_procs; k++) {
      MPI_Irecv(&res_grid[procs_info[k].start], procs_info[k].size, MPI_FLOAT, k, k, MPI_COMM_WORLD, &requests[k-1]);
    }
  }

  // Wait workers termination and check if 'io_thread' has exited
  if(!rank && pthread_join(io_thread, (void*) &rc)) 
    fprintf(stderr, "Join error, io_thread exited with: %d", rc);
  for(int i = 0; i < num_threads-1; i++) {
    if(pthread_join(threads[i], (void*) &rc)) 
      fprintf(stderr, "Join error, worker_thread[%d] exited with: %d", i, rc);
  }

  if(rank) MPI_Send(&res_grid[pad_size], proc_assigned_rows_size, MPI_FLOAT, 0, rank, MPI_COMM_WORLD);

  // Store computed matrix. Rank 0 has the whole matrix file in memory, other ranks have only the part they are interested in
  if (!rank) {
    if(num_procs > 1) {
      MPI_Status statuses[num_procs-1];
      MPI_Waitall(num_procs-1, requests, statuses);
    }
    write_time = PAPI_get_real_usec();
    int float_written = fwrite(res_grid, sizeof(float), grid_size, fp_result);
    fclose(fp_result);
    write_time = PAPI_get_real_usec() - write_time;

    if(float_written != grid_size) {
      fprintf(stderr, "Error in file writing: number of float grid elements written (%d) is different from the expected amount (%ld)\n", float_written, grid_size);
      exit(-1);
    }
  }

  // Print measures
  time_stop = PAPI_get_real_usec();
  printf("Rank[%d] | Elapsed time: %lld us\n", rank, (time_stop - time_start));
  if(!rank) printf("Rank[%d] I/O times | Reads from disk: %lld us | Write to disk: %llu us\n", rank, read_time, write_time);

  long_long* curr_meas;
  for(int i = 0; i < num_threads; i++) {
    curr_meas = thread_measures[i];
    printf("Thread[%d][%d]: Elapsed: %llu | Condition WT: %llu | Handlers mutex WT: %llu | MPI mutex WT: %llu | Total L2 cache misses: %lld\n", 
      rank, i, curr_meas[0], curr_meas[1], curr_meas[2], curr_meas[3], curr_meas[4]);
    free(curr_meas);
  }

  // Destroy pthread objects and free all used resources
  pthread_mutex_destroy(&mutex_mpi);
  pthread_mutex_destroy(&io_args.mutex);
  pthread_cond_destroy(&io_args.cond);
  for(int i = 0; i < num_threads; i++) {
    pthread_mutex_destroy(&handlers[i].mutex);
    pthread_cond_destroy(&handlers[i].pad_ready);
  }

  MPI_Finalize();
  free(thread_measures);
  free(handlers);
  free(grid);
  free(old_grid);
  free(kernel);
  exit(0);
}

/* This is executed only by rank 0 */
void* grid_input_thread(void* args) {
  struct io_thread_args *io_args = (struct io_thread_args*)args;
  *(io_args->read_time) = PAPI_get_real_usec();
  const int float_read = fread(&old_grid[pad_size], sizeof(float), grid_size, io_args->fp_grid);
  fclose(io_args->fp_grid);
  *(io_args->read_time) = PAPI_get_real_usec() - *(io_args->read_time);
  if(float_read < grid_size) {
    fprintf(stderr, "Error in file reading: number of float elements read (%d) is lower than the expected amount (%ld)\n", float_read, grid_size);
    exit(-1);
  }

  // Signal read completion and check if processes info are ready 
  pthread_mutex_lock(&(io_args->mutex));
  io_args->grid_ready = 1;
  pthread_cond_broadcast(&(io_args->cond));
  if(!io_args->procs_info[num_procs-1].size)
    pthread_cond_wait(&(io_args->cond), &(io_args->mutex));
  pthread_mutex_unlock(&(io_args->mutex));

  // Grid distribution
  if(num_procs > 1) {
    MPI_Request grid_reqs[num_procs-1];
    pthread_mutex_lock(&mutex_mpi);
    for(int i = 1; i < num_procs; i++) {
      MPI_Isend(&old_grid[io_args->procs_info[i].start], io_args->procs_info[i].size, MPI_FLOAT, i, i, MPI_COMM_WORLD, &grid_reqs[i-1]);
    }
    pthread_mutex_unlock(&mutex_mpi);

    // Info about result gathering. Pads are excluded
    for(int i = 1; i < num_procs; i++) {
      io_args->procs_info[i].start += num_pads * grid_width;            // Starting position final recv
      io_args->procs_info[i].size -= num_pads * 2 * grid_width;         // Payload size for final recv
      if(i == num_procs-1) io_args->procs_info[i].size += grid_width * num_pads;
    }
  }

  pthread_exit(0);
}

void* worker_thread(void* args) {
  struct thread_handler *handler = (struct thread_handler*)args;
  float *my_old_grid = old_grid;
  float *my_grid = grid;
  float *temp;                                      // Used only for grid swap
  int completed[3];                                 // If my top, bottom, or central rows have been completed
  int center_start;                                 // Center elements are completed one row at a time
  struct mpi_args *margs = &(struct mpi_args){0};   // Pointer to an empty struct, initialized later if this thread needs MPI 
  margs->request[1] = MPI_REQUEST_NULL;             // Test a null request in the first iteration (non-null value only for odd iterations)
  uint8_t prev_iter_index;                          // Even/odd index for curr/next iteration (and viceversa). Used as index of request array
  uint8_t mpi_needed = (num_procs > 1) && ((!handler->tid && rank) || (handler->tid == num_threads-1 && rank < num_procs-1));

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

  long_long t, cond_wait_time = 0, handler_mutex_wait_time = 0, mpi_mutex_wait_time = 0;
  long_long* measures[3] = {&cond_wait_time, &handler_mutex_wait_time, &mpi_mutex_wait_time};
  long_long* meas_to_return = malloc(sizeof(long_long) * 5);
  long_long time_start = PAPI_get_real_usec();

  // Thread setup
  if(affinity && stick_this_thread_to_core(handler->tid)) {
    fprintf(stderr, "Error occurred while setting thread affinity on core: %d\n", handler->tid);
    exit(-1);
  }
  initialize_thread_coordinates(handler);

  if(mpi_needed) {
    if(handler->tid == 0) {
      margs->send_position = handler->start;
      margs->recv_position = 0;
      margs->neighbour = rank - 1;
    } else {
      margs->send_position = (handler->end - pad_size);
      margs->recv_position = handler->end;
      margs->neighbour = rank + 1;
    }
  } else {
    margs = NULL;
  }

  // Synchronization point, check if grid data is ready and start convolution
  if(num_procs > 1 && rank && !handler->tid) {
    MPI_Wait(&(io_args.request), MPI_STATUS_IGNORE);
    pthread_mutex_lock(&(io_args.mutex));
    io_args.grid_ready = 1;
    pthread_cond_broadcast(&(io_args.cond));
    pthread_mutex_unlock(&(io_args.mutex));
  } else {
    pthread_mutex_lock(&(io_args.mutex));
    if(!io_args.grid_ready) 
      pthread_cond_wait(&(io_args.cond), &(io_args.mutex));
    pthread_mutex_unlock(&(io_args.mutex));
  }

  // First convolution iteration (starting with top and bottom rows)
  conv_subgrid(my_old_grid, my_grid, handler->start, (handler->start + pad_size));
  conv_subgrid(my_old_grid, my_grid, (handler->end - pad_size), handler->end);

  t = PAPI_get_real_usec();
  pthread_mutex_lock(&(handler->mutex));
  handler_mutex_wait_time += PAPI_get_real_usec() - t;
  handler->top_rows_done[0] = 1;
  handler->bot_rows_done[0] = 1;
  pthread_cond_broadcast(&(handler->pad_ready));
  pthread_mutex_unlock(&(handler->mutex));

  // Send top or bottom rows
  if(mpi_needed) {
    t = PAPI_get_real_usec();
    pthread_mutex_lock(&mutex_mpi);
    mpi_mutex_wait_time += PAPI_get_real_usec() - t;
    MPI_Isend(&my_grid[margs->send_position], pad_size, MPI_FLOAT, margs->neighbour, 0, MPI_COMM_WORLD, &(margs->request[0]));
    MPI_Irecv(&my_grid[margs->recv_position], pad_size, MPI_FLOAT, margs->neighbour, 0, MPI_COMM_WORLD, &(margs->request[2]));
    pthread_mutex_unlock(&mutex_mpi);
  }

  // Complete the first convolution iteration by computing central elements
  conv_subgrid(my_old_grid, my_grid, (handler->start + pad_size), (handler->end - pad_size));

  // Second or higher convolution iterations
  for(int iter = 1; iter < num_iterations; iter++) {
    temp = my_old_grid;
    my_old_grid = my_grid;
    my_grid = temp;
    prev_iter_index = (iter-1) % 2;
    center_start = handler->start + pad_size;
    memset(completed, 0, sizeof(int) * 3);
    if(margs != NULL) { 
      margs->requests_completed[prev_iter_index] = 0;
      margs->requests_completed[2] = 0;
    }

    while(!completed[TOP] || !completed[BOTTOM] || !completed[CENTER]) {
      if(!completed[TOP]) {
        test_and_update(TOP, iter, completed, margs, handler, measures);
      }

      if(!completed[BOTTOM]) {
        test_and_update(BOTTOM, iter, completed, margs, handler, measures);
      }

      // Computing central rows one at a time if top and bottom rows are incomplete
      if(!completed[CENTER]) {
        int center_end;
        if (completed[TOP] && completed[BOTTOM]) {
          center_end = handler->end - pad_size;
          completed[CENTER] = 1;
        } else {
          center_end = center_start + grid_width;
        }

        conv_subgrid(my_old_grid, my_grid, center_start, center_end);

        if(center_end == (handler->end - pad_size)) completed[CENTER] = 1;
        else center_start += grid_width;
      }
    }
  }

  // Retrieving execution info
  long_long num_cache_miss;
  if ((rc = PAPI_stop(event_set, &num_cache_miss)) != PAPI_OK)
    handle_PAPI_error(rc, "Error in PAPI_stop().");  
 
  meas_to_return[1] = cond_wait_time; 
  meas_to_return[2] = handler_mutex_wait_time; 
  meas_to_return[3] = mpi_mutex_wait_time; 
  meas_to_return[4] = num_cache_miss;
  meas_to_return[0] = (PAPI_get_real_usec() - time_start);
  thread_measures[handler->tid] = meas_to_return;

  if(handler->tid != num_threads-1) pthread_exit(0);
  return 0;
}

/* Test if pad rows are ready. If they are, compute their convolution and send/signal their completion */
void test_and_update(uint8_t position, uint8_t iter, int* completed, struct mpi_args* margs, struct thread_handler* handler, long_long** meas) {
  int tid;
  uint8_t index = (iter-1) % 2;
  uint8_t *rows_to_wait, *rows_to_assert;
  struct thread_handler* neigh_handler;
  long_long *condition_wait_time = meas[0];
  long_long t;

  switch(position) {
    case TOP:
      tid = 0;
      neigh_handler = handler->top;
      rows_to_assert = handler->top_rows_done;
      if(neigh_handler == NULL) break;
      rows_to_wait = handler->top->bot_rows_done;
      break;

    case BOTTOM:
      tid = num_threads-1;
      neigh_handler = handler->bottom;
      rows_to_assert = handler->bot_rows_done;
      if(neigh_handler == NULL) break;
      rows_to_wait = handler->bottom->top_rows_done;
      break;

    default:
      return;
  }

  if(handler->tid == tid && margs != NULL) {
    // In this branch if current thread has distributed memory dependency
    int outcount;
    int indexes[3] = {0, 0, 0};
    MPI_Status statuses[3];
    long_long *mpi_mutex_wait_time = meas[2];

    t = PAPI_get_real_usec();
    pthread_mutex_lock(&mutex_mpi);
    *mpi_mutex_wait_time += PAPI_get_real_usec() - t;
    MPI_Testsome(3, margs->request, &outcount, indexes, statuses);
    pthread_mutex_unlock(&mutex_mpi);
    for(int i = 0; i < outcount; i++) margs->requests_completed[indexes[i]] = 1;
    if(margs->requests_completed[index] && margs->requests_completed[2]) completed[position] = 1;

    if(!completed[position] && completed[!position] && completed[CENTER]) {
      struct timespec remaining;
      while(!completed[position]) {
        t = PAPI_get_real_usec();
        nanosleep(&WAIT_TIME, &remaining);
        *condition_wait_time += PAPI_get_real_usec() - t;

        t = PAPI_get_real_usec();
        pthread_mutex_lock(&mutex_mpi);
        *mpi_mutex_wait_time += PAPI_get_real_usec() - t;

        //MPI_Testsome(3, request, &outcount, indexes, statuses);
        MPI_Waitsome(3, margs->request, &outcount, indexes, statuses);
        pthread_mutex_unlock(&mutex_mpi);
        for(int i = 0; i < outcount; i++) margs->requests_completed[indexes[i]] = 1;
        if(margs->requests_completed[index] && margs->requests_completed[2]) completed[position] = 1;
      }
    }
  } else if(neigh_handler == NULL) {
    // If current thread is the "highest" or the "lowest" (no dependency with upper or lower thread)
    completed[position] = 1;
  } else {
    // If current thread has a shared memory dependency with upper or lower thread 
    long_long *handler_mutex_wait_time = meas[1];
    t = PAPI_get_real_usec();
    pthread_mutex_lock(&(neigh_handler->mutex));
    *handler_mutex_wait_time += PAPI_get_real_usec() - t;
    completed[position] = rows_to_wait[index];
    if(completed[position]) 
      rows_to_wait[index] = 0;
    else if(completed[!position] && completed[CENTER]) {
      t = PAPI_get_real_usec();
      while(!rows_to_wait[index]) {
        pthread_cond_wait(&(neigh_handler->pad_ready), &(neigh_handler->mutex));
      }
      *condition_wait_time += PAPI_get_real_usec() - t;
      rows_to_wait[index] = 0;
      completed[position] = 1;
    }
    pthread_mutex_unlock(&(neigh_handler->mutex));
  }

  
  if(!completed[position]) return;

  // If test was successful, compute convolution of the part tested
  long_long *handler_mutex_wait_time = meas[1];
  long_long *mpi_mutex_wait_time = meas[2];
  float *my_grid, *my_old_grid;
  if(iter % 2) {
    my_grid = old_grid;
    my_old_grid = grid;
  } else {
    my_grid = grid;
    my_old_grid = old_grid;
  }
  
  int start = (position == TOP) ? handler->start : handler->end - pad_size;
  conv_subgrid(my_old_grid, my_grid, start, (start + pad_size));

  if(iter+1 == num_iterations) return; 
  
  if(handler->tid == tid && margs != NULL) {
    t = PAPI_get_real_usec();
    pthread_mutex_lock(&mutex_mpi);
    mpi_mutex_wait_time += PAPI_get_real_usec() - t;
    MPI_Isend(&my_grid[margs->send_position], pad_size, MPI_FLOAT, margs->neighbour, 0, MPI_COMM_WORLD, &(margs->request[iter % 2]));
    MPI_Irecv(&my_grid[margs->recv_position], pad_size, MPI_FLOAT, margs->neighbour, 0, MPI_COMM_WORLD, &(margs->request[2]));
    // Avoid to overwrite data of previous Isend with next convolution
    MPI_Test(&(margs->request[index]), &(margs->requests_completed[index]), MPI_STATUS_IGNORE);
    pthread_mutex_unlock(&mutex_mpi);
  } else {
    t = PAPI_get_real_usec();
    pthread_mutex_lock(&(handler->mutex));
    handler_mutex_wait_time += PAPI_get_real_usec() - t;
    rows_to_assert[iter % 2] = 1;
    pthread_cond_broadcast(&(handler->pad_ready));
    pthread_mutex_unlock(&(handler->mutex));
  }
}

/* Compute convolution of "sub_grid" in the specified range. Save the result in "new_grid" */
void conv_subgrid(float *sub_grid, float *new_grid, int start_index, int end_index) {
  float result;
  float matrix_dot_sum;                    // Used for normalization
  int col = start_index % grid_width;      // Index of current column
  int row_start = start_index - col;       // Index of the first element in current row

  int offset;                              // How far is current element from its closest border
  int grid_index;                          // Current position in grid matrix
  int kern_index;                          // Current position in kernel matrix
  int kern_end;                            // Describes when it's time to change row
  int iterations;                          // How many iterations are necessary to calc a single value of "new_grid"
  //__m128 vec_grid, vec_kern, vec_temp;     // Floating point vector for grid and kernel (plus a temp vector)
  //__m128 vec_rslt;                         // Floating point vector of result, will be reduced at the end
  //__m128 vec_mxds;                         // Floating point vector of matrix dot sum, will be reduced at the end

  for(int i = start_index; i < end_index; i++) {
    // Setting indexes for current element
    if(col < num_pads) {
      for(offset = 0; i-offset > row_start && offset <= num_pads; offset++);
      grid_index = i-offset-pad_size;
      kern_index = (kern_width / 2) - offset;
      kern_end = kern_width-kern_index;
      iterations = (num_pads+col+1) *kern_width;
    } else if (col > grid_width-1-num_pads){
      int row_end = row_start + grid_width - 1;
      for(offset = 0; i+offset <= row_end && offset <= num_pads; offset++);
      grid_index = i-num_pads-pad_size;
      kern_index = 0;
      kern_end = kern_width-offset;
      iterations = (num_pads + grid_width-1-col) *kern_width;
    } else {
      grid_index = i-num_pads-pad_size;
      kern_index = 0;
      kern_end = kern_width;
      iterations = kern_size;
    }

    // Packed SIMD instructions are temporary available only for kernels having a 3 elements row 
    /*if(iterations == kern_size && kern_width == 3) {
      vec_rslt = _mm_setzero_ps();
      vec_mxds = _mm_setzero_ps();
      for(int i = 0; i < kern_width; i++) {
        vec_grid = _mm_maskload_ps(&sub_grid[grid_index], mask);
        vec_kern = _mm_loadu_ps(&kernel[kern_index]);
        vec_temp = _mm_mul_ps(vec_grid, vec_kern);
        vec_rslt = _mm_add_ps(vec_rslt, vec_temp);
        vec_temp = _mm_mul_ps(vec_grid, vec_grid);
        vec_mxds = _mm_add_ps(vec_mxds, vec_temp);
        grid_index += grid_width;
        kern_index += kern_width;
      }
      vec_mxds = _mm_hadd_ps(vec_mxds, vec_mxds);
      vec_mxds = _mm_hadd_ps(vec_mxds, vec_mxds);
      vec_rslt = _mm_hadd_ps(vec_rslt, vec_rslt);
      vec_rslt = _mm_hadd_ps(vec_rslt, vec_rslt);
      result = vec_rslt[0];
      matrix_dot_sum = vec_mxds[0];
    } else {*/
      // Convolution
      result = 0; offset = 0; matrix_dot_sum = 0;
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
    //}

    // Normalization (avoid NaN results by assigning the mean value 0 if needed)
    new_grid[i] = (!matrix_dot_sum) ? 0 : (result / sqrt(matrix_dot_sum * kern_dot_sum));

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
  kernel = malloc(sizeof(float) * kern_size);
  const int float_read = fread(kernel, sizeof(float), kern_size, fp_kernel);
  fclose(fp_kernel);

  if(float_read < kern_size) {
    fprintf(stderr, "Error in file reading: number of float kernel elements read (%d) is lower than the expected amount (%d)\n", float_read, kern_size);
    exit(-1);
  }
}

/* Print the appropriate message in case of PAPI error */
void handle_PAPI_error(int rc, char *msg) {
  char error_str[PAPI_MAX_STR_LEN];
  memset(error_str, 0, sizeof(char) * PAPI_MAX_STR_LEN);

  fprintf(stderr, "%s\nReturn code: %d - PAPI error message:\n", msg, rc);
  PAPI_perror(error_str);
  PAPI_strerror(rc);
  exit(-1);
}

/* Set thread affinity. If there are more threads than cores, no affinity will be set */
int stick_this_thread_to_core(int core_id) {
  const long num_cores = sysconf(_SC_NPROCESSORS_ONLN);
  if(num_threads > num_cores) return 0;
  if(core_id < 0) return 1;
  if((num_threads * 2) <= num_cores) core_id++;   // Trying to avoid hyperthreading in a bad way

  cpu_set_t cpuset;
  CPU_ZERO(&cpuset);
  CPU_SET(core_id, &cpuset);

  return pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);
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
 * distribute those additional rows starting from the center thread. 
 * 
 * For example, for 8 threads the assignment order of the additional rows would be: 
 * TID  : 0, 1, 2, 3, 4, 5, 6, 7
 * Order: 7, 5, 3, 1, 0, 2, 4, 6  
*/
void initialize_thread_coordinates(struct thread_handler* handler) {
  // Additional rows that need to be distributed between threads
  const int addrows = proc_assigned_rows % num_threads;
  const int fixed_rows_per_thread = proc_assigned_rows / num_threads;
  const int offset_from_center = handler->tid - num_threads/2;
  const int order = abs(offset_from_center) * 2 - ((offset_from_center >= 0) ? 0 : 1);
  const uint8_t thread_additional_row = addrows > order;

  // Previous thread may compute an additional row, compute how many additional rows for every thread
  int start_offset = addrows / 2;
  if(offset_from_center > 0) {
    for(int i = 2; (i <= order) && (addrows+1 >= i); i+=2) start_offset++;
  } else if (offset_from_center < 0) {
    for(int i = 1; (i <= order) && (addrows > i); i+=2) start_offset--;
  }

  // Initialize coordinates
  const int size = fixed_rows_per_thread * grid_width;
  handler->start = pad_size + handler->tid * size + start_offset * grid_width;
  handler->end = handler->start + size;
  if(thread_additional_row) handler->end += grid_width;
}
