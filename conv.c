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

#define DEFAULT_ITERATIONS 1
#define DEFAULT_THREADS 2
#define GRID_FILE_PATH "./io-files/grid.txt"
#define KERNEL_FILE_PATH "./io-files/kernel.txt"
#define RESULT_FILE_PATH "./io-files/result.txt"
#define NSEC_WAIT (25*1000)
#define TOP 0
#define BOTTOM 1
#define CENTER 2
#define EXP_CHARS 13                  // Standard "%+e" format has this num of chars (e.g. -9.075626e+20)

struct thread_handler {
  int tid;                            // Custom thread ID, not the one returned by "pthread_self()"
  uint start, end;                    // Matrix area of ​​interest for this thread
  uint8_t top_rows_done[2];           // Flags for current and next iteration
  uint8_t bot_rows_done[2];
  struct thread_handler* top;         // To exchange information about pads with neighbour threads
  struct thread_handler* bottom;
  pthread_mutex_t mutex;              // Mutex to access this handler
  pthread_cond_t pad_ready;           // Thread will wait if neighbour's top and bottom rows (pads) aren't ready
};

struct mpi_args {
  uint send_position, recv_position; 
  int neighbour;
  int requests_completed[3];
  MPI_Request request[3];             // There are at most two "Isend" and one "Irecv" not completed at the same time
};

void* worker_thread(void*);
void test_and_update(uint8_t, uint8_t, int*, struct mpi_args*, struct thread_handler*, long_long**);
void conv_subgrid(float*, float*, int, int);
void read_kernel(FILE*);
int echars_to_floats(float*, char*, int);
int floats_to_echars(float*, char*, int);
int stick_this_thread_to_core(int);
void handle_PAPI_error(int, char*);
void get_process_additional_row(int*);
void initialize_thread_coordinates(struct thread_handler*);

pthread_mutex_t mutex_mpi;            // To call MPI routines (will be used only by top and bottom thread)
uint8_t num_pads;                     // Number of rows that should be shared with other processes
uint8_t kern_width;                   // Number of elements in one kernel matrix row
uint16_t grid_width;                  // Number of elements in one grid matrix row
uint64_t grid_size;                   // Number of elements in whole grid matrix
uint16_t kern_size;                   // Number of elements in whole kernel matrix
uint16_t pad_size;                    // Number of elements in the pad section of the grid matrix
int proc_assigned_rows;               // Number of rows assigned to a process
int proc_assigned_rows_size;          // Number of elements assigned to a process
int row_num_chars;                    // Number of chars that represent a float row of the grid
int num_procs;                        // Number of MPI processes in the communicator
int num_threads;                      // Number of threads (main included) for every MPI process
int num_iterations;                   // Number of convolution iterations
int rank;                             // MPI process identifier
float kern_dot_sum;                   // Used for normalization, its value is equal to: sum(dot(kernel, kernel))
float *kernel;                        // Kernel buffer
float *grid;                          // Grid buffer
float *old_grid;                      // Old grid buffer
const struct timespec WAIT_TIME = {   // Wait time used by MPI threads
  .tv_sec = 0, 
  .tv_nsec = NSEC_WAIT
};

int main(int argc, char** argv) {
  int provided;                       // MPI thread level supported
  int rc;                             // Return code used in error handling
  long_long time_start, time_stop;    // To measure execution time
  char *whole_char_grid = NULL;       // Char buffer used for I/O from disk
  FILE *fp_grid, *fp_kernel;          // I/O files for grid and kernel matrices

  // Fetch from arguments how many convolution iterations do and the number of threads
  num_iterations = (argc > 1) ? atoi(argv[1]) : DEFAULT_ITERATIONS;
  num_threads = (argc > 2) ? atoi(argv[2]) : DEFAULT_THREADS;
  if(num_iterations < DEFAULT_ITERATIONS) {
    fprintf(stderr, "Invalid number of convolution iterations (first argument), value inserted: %d\n", num_iterations);
    exit(-1);
  }
  if(num_threads < DEFAULT_THREADS) {
    fprintf(stderr, "Invalid number of threads (second argument), value inserted: %d\n", num_threads);
    exit(-1);
  }

  // Main thread + worker threads
  pthread_t threads[num_threads-1];       

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
  MPI_Request requests[num_procs * 2];

  fp_grid = NULL;
  if(!rank) {
    // Opening input files
    if((fp_grid = fopen(GRID_FILE_PATH, "r")) == NULL) {
      fprintf(stderr, "Error while opening grid file\n");
      exit(-1);
    }
    if((fp_kernel = fopen(KERNEL_FILE_PATH, "r")) == NULL) {
      fprintf(stderr, "Error while opening kernel file\n");
      exit(-1);
    }

    // First token represent matrix dimension
    if(fscanf(fp_grid, "%hd\n", &grid_width) == EOF || fscanf(fp_kernel, "%hhd\n", &kern_width) == EOF) {
      fprintf(stderr, "Error in file reading: first element should be the row (or column) length of a square matrix\n");
      exit(-1);
    }

    // Exchange initial information 
    if(num_procs > 1) {
      int to_send[] = {grid_width, kern_width};
      for (int i = 1; i < num_procs; i++)
        MPI_Isend(to_send, 2, MPI_INT, i, i, MPI_COMM_WORLD, &requests[i]);
    }

    // Exchange kernel
    kern_size = kern_width * kern_width;
    read_kernel(fp_kernel);
    if(num_procs > 1) {
      for (int i = 1; i < num_procs; i++)
        MPI_Isend(kernel, kern_size, MPI_FLOAT, i, i, MPI_COMM_WORLD, &requests[i + num_procs]);
    }
  } else {
    int to_recv[2];
    MPI_Recv(to_recv, 2, MPI_INT, 0, rank, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    grid_width = to_recv[0];
    kern_width = to_recv[1];
    kern_size = kern_width * kern_width;
    kernel = malloc(sizeof(float) * kern_size);
    MPI_Irecv(kernel, kern_size, MPI_FLOAT, 0, rank, MPI_COMM_WORLD, &requests[0]);
  }

  // Variable initialization and data splitting
  grid_size = grid_width * grid_width;
  num_pads = (kern_width - 1) / 2;
  pad_size = grid_width * num_pads;
  row_num_chars = (grid_width * EXP_CHARS + grid_width) * sizeof(char);  // (Non-blank chars + blank chars) * sizeof(char)

  int process_rows_info[(rank) ? 1 : num_procs*2];                       // Used in a different way by rank 0
  get_process_additional_row(process_rows_info);                         // Which process must compute an additional row 

  const int fixed_rows_per_proc = (grid_width / num_procs);              // Minimum amout of rows distributed to each process
  proc_assigned_rows = fixed_rows_per_proc + *process_rows_info;         // Number of rows assigned to current process
  proc_assigned_rows_size = proc_assigned_rows * grid_width;             // Number of elements assigned to a process

  grid = malloc((proc_assigned_rows_size + pad_size*2) * sizeof(float));
  old_grid = malloc((proc_assigned_rows_size + pad_size*2) * sizeof(float));
  float* whole_grid = NULL;

  // Set a zero-pad for lowest and higher process 
  if(!rank){
    memset(grid, 0, pad_size * sizeof(float));
    memset(old_grid, 0, pad_size * sizeof(float));
  }
  if(rank == num_procs-1) {
    memset(&grid[proc_assigned_rows_size + pad_size], 0, pad_size * sizeof(float));
    memset(&old_grid[proc_assigned_rows_size + pad_size], 0, pad_size * sizeof(float));
  }

  // Read or receive grid data
  if(!rank){
    int char_buffer_size = row_num_chars * grid_width;
    whole_char_grid = malloc(char_buffer_size);
    int char_read = fread(whole_char_grid, sizeof(char), char_buffer_size, fp_grid);
    fclose(fp_grid);

    if(char_read < char_buffer_size) {
      fprintf(stderr, "Error in file reading: number of char grid elements read (%d) is lower than the expected amount (%d)\n", char_read, char_buffer_size);
      exit(-1);
    }

    whole_grid = malloc(grid_size * sizeof(float));
    int converted = echars_to_floats(whole_grid, whole_char_grid, grid_size);

    if(converted != grid_size) {
      fprintf(stderr, "Error in file reading: number of float grid elements read (%d) is different from the expected amount (%ld)\n", converted, grid_size);
      exit(-1);
    }

    // Send grid to ranks greater than 0
    if(num_procs > 1) {
      int start, size;
      int offset = process_rows_info[0];
      MPI_Request grid_reqs[num_procs-1];
      for(int i = 1; i < num_procs; i++) {
        // Info about data scattering. Pads are included (to avoid an MPI exchange in the first iteration)
        start = (fixed_rows_per_proc * i - num_pads + offset) * grid_width;                           // Starting position for Isend
        size = (fixed_rows_per_proc + num_pads*2 + process_rows_info[i]) * grid_width;                // Payload size for Isend
        offset += process_rows_info[i];
        if(i == num_procs-1) size -= pad_size;
        MPI_Isend(&whole_grid[start], size, MPI_FLOAT, i, i, MPI_COMM_WORLD, &grid_reqs[i-1]);

        // Info about result gathering. Pads are excluded
        process_rows_info[i+num_procs] = (fixed_rows_per_proc + process_rows_info[i]) * grid_width;   // Payload size for final recv 
        process_rows_info[i] = start + pad_size*2;                                                    // Starting position final recv
      }
    }
    process_rows_info[0] = pad_size;
    process_rows_info[num_procs] = proc_assigned_rows_size;
    memcpy(&old_grid[pad_size], whole_grid, (proc_assigned_rows_size + pad_size) * sizeof(float));    // Rank 0
  } else {
    int recv_size = proc_assigned_rows_size + pad_size*2; 
    if(rank == num_procs-1) recv_size -= pad_size;
    MPI_Recv(old_grid, recv_size, MPI_FLOAT, 0, rank, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  }

  // Complete kernel receive and compute "sum(dot(kernel, kernel))"
  if(rank) MPI_Wait(requests, MPI_STATUS_IGNORE);
  for(int pos = 0; pos < kern_size; pos++) {
    kern_dot_sum += kernel[pos] * kernel[pos];
  }

  time_start = PAPI_get_real_usec();
  pthread_mutex_init(&mutex_mpi, NULL);
  // PThreads arguments initialization 
  struct thread_handler* handlers = malloc(sizeof(struct thread_handler) * (num_threads));
  for(int i = 0; i < num_threads; i++) {
    handlers[i].tid = i;
    handlers[i].top_rows_done[0] = 0;
    handlers[i].top_rows_done[1] = 0;
    handlers[i].bot_rows_done[0] = 0;
    handlers[i].bot_rows_done[1] = 0;
    handlers[i].top = (i > 0) ? &handlers[i-1] : NULL;
    handlers[i].bottom = (i < num_threads-1) ? &handlers[i+1] : NULL;
    pthread_mutex_init(&handlers[i].mutex, NULL);
    pthread_cond_init(&handlers[i].pad_ready, NULL);
  }

  // PThreads creation
  for(int i = 0; i < num_threads-1; i++) {
    rc = pthread_create(&threads[i], NULL, worker_thread, (void*)&handlers[i]);
    if (rc) { 
      fprintf(stderr, "Error while creating pthread[%d]; Return code: %d\n", i, rc);
      exit(-1);
    }
  }
  worker_thread((void*) &handlers[num_threads-1]);   // Main thread is the bottom thread

  // Wait workers termination
  for(int i = 0; i < num_threads-1; i++) {
    if(pthread_join(threads[i], (void*) &rc)) 
      fprintf(stderr, "Join error, thread[%d] exited with: %d", i, rc);
  }

  // Gather results
  float *result = (num_iterations % 2) ? grid : old_grid;
  if(num_procs > 1) {
    if(rank != 0) {
      MPI_Send(&result[pad_size], proc_assigned_rows_size, MPI_FLOAT, 0, 11, MPI_COMM_WORLD);
    } else {
      result = realloc(result, (grid_size + pad_size) * sizeof(float));
      for(int k = 1; k < num_procs; k++) {
        MPI_Recv(&result[process_rows_info[k]], process_rows_info[k+num_procs], MPI_FLOAT, k, 11, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      }
    }
  }

  time_stop = PAPI_get_real_usec();
  printf("Rank[%d] | Elapsed time: %lld us\n", rank, (time_stop - time_start));

  // Store computed matrix
  if (!rank) {
    FILE *fp_result;
    if((fp_result = fopen(RESULT_FILE_PATH, "w")) == NULL) {
      fprintf(stderr, "Error while creating and/or opening result file\n");
      exit(-1);
    }
    int converted = floats_to_echars(&result[pad_size], whole_char_grid, grid_size);
    int char_written = fwrite(whole_char_grid, sizeof(char), converted, fp_result);
    fclose(fp_result);

    if(char_written != converted) {
      fprintf(stderr, "Error in file writing: number of char grid elements written (%d) is different from the expected amount (%d)\n", char_written, converted);
      exit(-1);
    }
  }

  // Destroy pthread objects and free all used resources
  pthread_mutex_destroy(&mutex_mpi);
  for(int i = 0; i < num_threads; i++) {
    pthread_mutex_destroy(&handlers[i].mutex);
    pthread_cond_destroy(&handlers[i].pad_ready);
  }

  MPI_Barrier(MPI_COMM_WORLD);
  MPI_Finalize();
  free(handlers);
  if(result == grid) free(old_grid);
  else free(grid);
  free(result);
  if(!rank) {
    free(whole_grid);
    free(whole_char_grid);
  }
  free(kernel);
  exit(0);
}

void* worker_thread(void* args) {
  struct thread_handler *handler = (struct thread_handler*)args;
  initialize_thread_coordinates(handler);

  if(stick_this_thread_to_core(handler->tid)) {
    fprintf(stderr, "Error occurred while setting thread affinity on core: %d\n", handler->tid);
    exit(-1);
  }

  float *my_old_grid = old_grid;
  float *my_grid = grid;
  float *temp;                                      // Used only for grid swap
  int completed[3];                                 // If my top, bottom, or central rows have been completed
  int center_start;                                 // Center elements are completed one row at a time
  struct mpi_args *margs = &(struct mpi_args){0};   // Pointer to an empty struct
  margs->request[1] = MPI_REQUEST_NULL;
  uint8_t prev_iter_index;
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
  long_long time_start = PAPI_get_real_usec();

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
    if(handler->tid == 0) {
      margs->send_position = handler->start;
      margs->recv_position = 0;
      margs->neighbour = rank - 1;
    } else {
      margs->send_position = (handler->end - pad_size);
      margs->recv_position = handler->end;
      margs->neighbour = rank + 1;
    }

    t = PAPI_get_real_usec();
    pthread_mutex_lock(&mutex_mpi);
    mpi_mutex_wait_time += PAPI_get_real_usec() - t;
    MPI_Isend(&my_grid[margs->send_position], pad_size, MPI_FLOAT, margs->neighbour, 0, MPI_COMM_WORLD, &(margs->request[0]));
    MPI_Irecv(&my_grid[margs->recv_position], pad_size, MPI_FLOAT, margs->neighbour, 0, MPI_COMM_WORLD, &(margs->request[2]));
    pthread_mutex_unlock(&mutex_mpi);
  } else {
    margs = NULL;
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
  t = PAPI_get_real_usec();
  long_long num_cache_miss;
  if ((rc = PAPI_stop(event_set, &num_cache_miss)) != PAPI_OK)
    handle_PAPI_error(rc, "Error in PAPI_stop().");
  
  printf("Thread[%d][%d]: Elapsed: %llu | Condition WT: %llu | Handlers mutex WT: %llu | MPI mutex WT: %llu | Total L2 cache misses: %lld\n", 
    rank, handler->tid, (t - time_start), cond_wait_time, handler_mutex_wait_time, mpi_mutex_wait_time, num_cache_miss);

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
  int grid_index;
  int kern_index;
  int kern_end;                            // Describes when it's time to change row
  int iterations;

  for(int i = start_index; i < end_index; i++) {
    // Setting indexes for current element
    if(col < num_pads) {
      for(offset = 0; i-offset > row_start && offset <= num_pads; offset++);
      grid_index = i-offset-pad_size;
      kern_index = (kern_width >> 1) - offset;
      kern_end = kern_width-kern_index;
      iterations = (num_pads+col+1) *kern_width;
    } else if (col > grid_width-1-num_pads){
      int row_end = row_start + grid_width - 1;
      for(offset = 0; i+offset <= row_end && offset <= num_pads; offset++);
      grid_index = i-num_pads-pad_size;
      kern_index = 0;
      kern_end = kern_width-offset;
      iterations = (num_pads + grid_width-col) *kern_width;
    } else {
      grid_index = i-num_pads-pad_size;
      kern_index = 0;
      kern_end = kern_width;
      iterations = kern_size;
    }

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
  const int kern_row_chars = (kern_width * EXP_CHARS + kern_width) * sizeof(char);
  const int char_buffer_size = kern_row_chars * kern_width;
  char *temp_char_buffer;
  
  temp_char_buffer = malloc(sizeof(char) * char_buffer_size);
  kernel = malloc(sizeof(float) * kern_size);
  
  int char_read = fread(temp_char_buffer, sizeof(char), char_buffer_size, fp_kernel);
  fclose(fp_kernel);

  if(char_read < char_buffer_size) {
    fprintf(stderr, "Error in file reading: number of char grid elements read (%d) is lower than the expected amount (%d)\n", char_read, char_buffer_size);
    exit(-1);
  }

  int converted = echars_to_floats(kernel, temp_char_buffer, kern_size);
  free(temp_char_buffer);

  if(converted != kern_size) {
    fprintf(stderr, "Error in file reading: number of kernel elements read (%d) is different from the expected amount (%d)\n", converted, kern_size);
    exit(-1);
  }
}

/* 
 * Assuming that scientific notation (%+e) is used in "char_buffer", get "count" floats from "char_buffer" and 
 * save them in "float_buffer". Returns how many floats have been stored in "float_buffer".
*/
int echars_to_floats(float *float_buffer, char* char_buffer, int count) {
  int fetched = 0, stored = 0;
  for(; stored < count; stored++) { 
    float_buffer[stored] = atof(&char_buffer[fetched]);
    while(char_buffer[fetched] >= '+') fetched++;                                 // Jump to next blank character
    while(char_buffer[fetched] != '\0' && char_buffer[fetched] < '+') fetched++;  // Jump all blank characters
  }
 
  return stored;
}

/* 
 * Convert "count" floats from "float_buffer" in "char_buffer" using scientific notation (%+e). 
 * Returns how many chars have been stored in "char_buffer".
*/
int floats_to_echars(float *float_buffer, char* char_buffer, int count) {
  int limit = grid_width-1;
  int stored = 0;

  for(int fetched = 0; fetched < count; fetched++){
    stored += sprintf(&char_buffer[stored], "%+e ", float_buffer[fetched]);
    if (fetched == limit) {
      limit += grid_width;
      stored += sprintf(&char_buffer[stored], "\n");
    }
  }

  return stored;
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
void get_process_additional_row(int* retval) {
  // Additional rows that need to be distributed between processes
  const int addrows = grid_width % num_procs;
  int offset_from_last_rank = num_procs - 1 - rank;
  const uint8_t closer_to_final_rank = rank > offset_from_last_rank;

  // This var assume value 0 for first rank, 1 for last, 2 for second, 3 for penultimate, ...
  int order = (closer_to_final_rank) ? (1 + offset_from_last_rank * 2) : (rank * 2);

  // This var assume a logical true value if this rank should compute one additional row
  const uint8_t proc_additional_row = addrows > order;
  if(rank) {
    *retval = proc_additional_row;
    return;
  }

  // Rank 0 needs info about other ranks
  retval[0] = proc_additional_row;
  for(int i = 1; i < num_procs; i++) {
    offset_from_last_rank = num_procs - 1 - i;
    order = (i > offset_from_last_rank) ? (1 + offset_from_last_rank * 2) : (i * 2);
    retval[i] =  addrows > order;
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
