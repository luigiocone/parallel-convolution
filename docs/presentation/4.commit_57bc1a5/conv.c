// conv.c
// Name: Tanay Agarwal, Nirmal Krishnan
// JHED: tagarwa2, nkrishn9

// 57bc1a50e09821c76ccfee769203d8592fae026e

#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <assert.h>
#include <string.h>
#include <math.h>
#include <papi.h>
#include <mpi.h>
#include <pthread.h>
#include <immintrin.h>

#define DEFAULT_ITERATIONS 1
#define DEFAULT_THREADS 2
#define GRID_FILE_PATH "./io-files/grids/haring.bin"
#define KERNEL_FILE_PATH "./io-files/kernels/gblur.bin"
#define RESULT_FILE_PATH "./io-files/result.txt"
#define MAX_CHARS 13     /* Standard "%e" format has at most this num of chars (e.g. -9.075626e+20) */
#define VEC_SIZE 4
#define DEBUG 0 
#define TOP 0
#define BOTTOM 1
#define CENTER 2

struct thread_handler {
  int rank, tid;
  uint start, end;
  uint8_t top_rows_done[2];
  uint8_t bot_rows_done[2];               /* Flags for current and next iteration */
  struct thread_handler* top;             /* To exchange information about pads with neighbor threads */
  struct thread_handler* bottom;
  pthread_mutex_t mutex;                  /* Mutex to access this handler */
  pthread_cond_t pad_ready;               /* Thread will wait if top and bottom rows (pads) aren't ready */
};

void* worker_thread(void*);
void update_log(uint8_t, uint8_t, uint8_t, int*, MPI_Request*, struct thread_handler*, long_long*);
void conv_subgrid(float*, float*, int, int);
void init_read(FILE*, FILE*);
void read_data(FILE*, float*, int);
void store_data(FILE*, float*, int);
void handle_PAPI_error(int, char*);
void read_float_matrix(FILE*, float*, int);
int stick_this_thread_to_core(int);

pthread_mutex_t mutex_mpi;    /* To call MPI routines (will be used only by top and bottom thread) */
uint8_t num_pads;             /* Number of rows that should be shared with other processes */
uint kern_width;           /* Number of elements in one kernel matrix row */
uint grid_width;          /* Number of elements in one grid matrix row */
uint64_t grid_size;           /* Number of elements in whole grid matrix */
uint16_t kern_size;           /* Number of elements in whole kernel matrix */
uint16_t pad_size;            /* Number of elements in the pad section of the grid matrix */
int rows_per_proc_size;       /* Number of elements assigned to a process */
int rows_per_thread_size;     /* Number of elements assigned to a thread */
int job_size;                 /* Number of elements in a job */
int log_buff_size;            /* Max number of jobs in "jobs" buffer */
int total_jobs;               /* Number of jobs for one iteration */
int last_job;                 /* Row index of last job (bordering bottom pads) */
int num_procs;                /* Number of MPI processes in the communicator */
int num_threads;              /* Number of threads for every MPI process */
int num_iterations;           /* Number of convolution operations */
int num_jobs;                 /* Number of jobs storable in a shared buffer */
float kern_dot_sum;           /* Used for normalization, its value is equal to: sum(dot(kernel, kernel)) */
float *kernel;                /* Kernel buffer */
float *grid;                  /* Grid buffer */
float *old_grid;              /* Old grid buffer */
__m128i last_mask;


int main(int argc, char** argv) {
  int rank;                               /* Current process identifier */
  int provided;                           /* MPI thread level supported */
  int rc;                                 /* Return code used in error handling */
  int event_set = PAPI_NULL;              /* Group of hardware events for PAPI library */
  long_long time_start, time_stop;        /* To measure execution time */
  long_long num_cache_miss;               /* To measure number of cache misses */
  FILE *fp_grid, *fp_kernel;              /* I/O files for grid and kernel matrices */

  /* How many times do the convolution operation and number of additional threads */
  num_iterations = (argc > 1) ? atoi(argv[1]) : DEFAULT_ITERATIONS;
  num_threads = (argc > 2) ? atoi(argv[2]) : DEFAULT_THREADS;
  assert(num_iterations > 0 && num_threads > 0);
  pthread_t threads[num_threads-1];
  
  /* MPI setup */
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
  int size = (num_procs < 4) ? 4 : 4 + num_procs-1;
  MPI_Status status[size];
  MPI_Request req[size];

  /* PAPI setup */
  if((rc = PAPI_library_init(PAPI_VER_CURRENT)) != PAPI_VER_CURRENT)
    handle_PAPI_error(rc, "Error in library init.");
  if((rc = PAPI_create_eventset(&event_set)) != PAPI_OK)
    handle_PAPI_error(rc, "Error while creating the PAPI eventset.");
  if((rc = PAPI_add_event(event_set, PAPI_L2_TCM)) != PAPI_OK)
    handle_PAPI_error(rc, "Error while adding L2 total cache miss event.");
  if((rc = PAPI_start(event_set)) != PAPI_OK) 
    handle_PAPI_error(rc, "Error in PAPI_start().");
  time_start = PAPI_get_real_usec();

  fp_grid = NULL;
  if(!rank) {
    /* Opening input files in dir "./io-files" */
    if((fp_grid = fopen(GRID_FILE_PATH, "rb")) == NULL) {
      fprintf(stderr, "Error while opening grid file\n");
      exit(-1);
    }
    if((fp_kernel = fopen(KERNEL_FILE_PATH, "rb")) == NULL) {
      fprintf(stderr, "Error while opening kernel file\n");
      exit(-1);
    }
    init_read(fp_grid, fp_kernel);
  } else {
    int to_recv[2];
    MPI_Ibcast(to_recv, 2, MPI_INT, 0, MPI_COMM_WORLD, req);   /* Async bcast needed by rank 0 */
    MPI_Wait(req, status);
    grid_width = to_recv[0];
    kern_width = to_recv[1];
    kern_size = kern_width * kern_width;
    kernel = malloc(sizeof(float) * kern_size);
    MPI_Ibcast(kernel, kern_size, MPI_FLOAT, 0, MPI_COMM_WORLD, req);
  }

  grid_size = grid_width * grid_width;
  num_pads = (kern_width - 1) / 2;
  pad_size = grid_width * num_pads;
  assert(grid_width % num_procs == 0);

  /* Data splitting and variable initialization */
  const int rows_per_proc = grid_width / num_procs;             /* Number of rows assigned to current process */
  const int rows_per_thread = rows_per_proc / num_threads;
  rows_per_proc_size = rows_per_proc * grid_width;              /* Number of elements assigned to a process */
  rows_per_thread_size = rows_per_thread * grid_width;

  /* Read grid data */
  grid = malloc((rows_per_proc + num_pads*2) * grid_width * sizeof(float));
  old_grid = malloc((rows_per_proc + num_pads*2) * grid_width * sizeof(float));
  float* whole_grid = NULL;
  if(!rank){
    memset(grid, 0, pad_size * sizeof(float));
    memset(old_grid, 0, pad_size * sizeof(float));
    if(num_procs == 1) {
      memset(&grid[(rows_per_proc_size+pad_size)], 0, pad_size * sizeof(float));
      memset(&old_grid[(rows_per_proc_size+pad_size)], 0, pad_size * sizeof(float));
    }
    whole_grid = malloc(grid_size * sizeof(float));
    read_data(fp_grid, whole_grid, rows_per_proc);
  } else if(rank == num_procs-1){
    memset(&grid[(rows_per_proc_size+pad_size)], 0, pad_size * sizeof(float));
    memset(&old_grid[(rows_per_proc_size+pad_size)], 0, pad_size * sizeof(float));
    MPI_Recv(old_grid, (rows_per_proc_size+pad_size), MPI_FLOAT, 0, rank, MPI_COMM_WORLD, &status[rank]);
  } else {
    MPI_Recv(old_grid, rows_per_proc_size+pad_size*2, MPI_FLOAT, 0, rank, MPI_COMM_WORLD, &status[rank]);
  }

  /* Computation of "last_mask" */
  uint rem = kern_width % VEC_SIZE;
  uint32_t to_load[VEC_SIZE];
  memset(to_load, 0, VEC_SIZE * sizeof(uint32_t));
  for(int i = 0; i < rem; i++) to_load[i] = UINT32_MAX;       // UINT32_MAX = -1
  last_mask = _mm_loadu_si128((__m128i*) to_load);

  /* First iteration - No rows exchange needed */
  if(rank) MPI_Wait(req, status);                               /* Complete kernel receive */
  for(int pos = 0; pos < kern_size; pos++){                     /* Computation of sum(dot(kernel, kernel)) */
    kern_dot_sum += kernel[pos] * kernel[pos];
  }
  
  pthread_mutex_init(&mutex_mpi, NULL);
  /* PThreads creation (every thread starts with a default job) */
  struct thread_handler* handlers = malloc(sizeof(struct thread_handler) * (num_threads));
  handlers[0].tid = 0;
  handlers[0].rank = rank;
  handlers[0].top_rows_done[0] = 0;
  handlers[0].top_rows_done[1] = 0;
  handlers[0].bot_rows_done[0] = 0;
  handlers[0].bot_rows_done[1] = 0;
  handlers[0].top = NULL;
  handlers[0].bottom = &handlers[1];
  pthread_mutex_init(&handlers[0].mutex, NULL);
  pthread_cond_init(&handlers[0].pad_ready, NULL);
  for(int i = 1; i < num_threads; i++) {
    handlers[i].tid = i;
    handlers[i].rank = rank;
    handlers[i].top_rows_done[0] = 0;
    handlers[i].top_rows_done[1] = 0;
    handlers[i].bot_rows_done[0] = 0;
    handlers[i].bot_rows_done[1] = 0;
    handlers[i].top = (i > 0) ? &handlers[i-1] : NULL;
    handlers[i].bottom = (i < num_threads-1) ? &handlers[i+1] : NULL;
    pthread_mutex_init(&handlers[i].mutex, NULL);
    pthread_cond_init(&handlers[i].pad_ready, NULL);
    
    rc = pthread_create(&threads[i-1], NULL, worker_thread, (void *)&handlers[i]);
    if (rc) { 
      fprintf(stderr, "Error while creating pthread[%d]; Return code: %d\n", i, rc);
      exit(-1);
    }
  }
  worker_thread((void *)&handlers[0]);

  /* Wait workers termination */
  void* ret;
  for(int i = 1; i < num_threads; i++) {
    if(pthread_join(threads[i-1], &ret)) 
      fprintf(stderr, "Join error, thread[%d] exited with: %d", i, *((int*)ret));
  }

  float *write_buffer = malloc(grid_size * sizeof(float) - grid_width * (grid_width / num_procs));
  float *my_grid = (num_iterations % 2) ? grid : old_grid;
  if(rank != 0) {
    MPI_Send(&my_grid[pad_size], rows_per_proc_size, MPI_FLOAT, 0, 11, MPI_COMM_WORLD);
    //MPI_Isend(&my_grid[pad_size], rows_per_proc_size, MPI_FLOAT, 0, 11, MPI_COMM_WORLD, req);
  } else {
    // wb
    for(int k = 0; k < num_procs-1; k++) {
      MPI_Irecv(&write_buffer[grid_width * (grid_width / num_procs) * k], rows_per_proc_size, MPI_FLOAT, k+1, 11, MPI_COMM_WORLD, &req[k]);
      //MPI_Recv(&write_buffer[grid_width * (grid_width / num_procs) * k], rows_per_proc_size, MPI_FLOAT, k+1, 11, MPI_COMM_WORLD, status);
    }
  }
  
  if(!rank) MPI_Waitall(num_procs-1, req, status);
  time_stop = PAPI_get_real_usec();
  if(!rank) printf("Rank[%d] | Elapsed time: %lld us\n", rank, (time_stop - time_start));

  /* Stop the count! */ 
  if ((rc = PAPI_stop(event_set, &num_cache_miss)) != PAPI_OK)
    handle_PAPI_error(rc, "Error in PAPI_stop().");
  if(!rank) printf("Rank[%d] | Total L2 cache misses:%lld\n", rank, num_cache_miss);
  
  /* Store computed matrix */
  if (DEBUG && !rank) {
    FILE *fp_result;
    if((fp_result = fopen(RESULT_FILE_PATH, "w")) == NULL) {
      fprintf(stderr, "Error while creating and/or opening result file\n");
      exit(-1);
    }
    store_data(fp_result, &my_grid[pad_size], rows_per_proc_size);
    store_data(fp_result, write_buffer, grid_size - rows_per_proc_size);
    fclose(fp_result);
  }

  /* Destroy pthread objects */
  pthread_mutex_destroy(&mutex_mpi);
  for(int i = 0; i < num_threads; i++) {
    pthread_mutex_destroy(&handlers[i].mutex);
    pthread_cond_destroy(&handlers[i].pad_ready);
  }

  MPI_Barrier(MPI_COMM_WORLD);
  MPI_Finalize();
  free(write_buffer);
  free(handlers);
  free(grid);
  free(old_grid);
  if(!rank) free(whole_grid);
  free(kernel);
  exit(0);
}

void* worker_thread(void* args){
  struct thread_handler *handler = (struct thread_handler*)args;
  handler->start = pad_size + handler->tid * rows_per_thread_size;
  handler->end = handler->start + rows_per_thread_size;

  float *my_old_grid = old_grid;
  float *my_grid = grid;
  float *temp;
  
  /* If my top, bottom, or central rows have been completed */
  int completed[3];
  int central_start;
  int neighbour = 0;
  uint8_t mpi_needed = (num_procs > 1) && ((!handler->tid && handler->rank) || (handler->tid == num_threads-1 && handler->rank < num_procs-1));
  uint8_t prev_iter_index;
  uint send_position = 0, recv_position = 0; 
  MPI_Request request[3];     /* There are at most two "Isend" and one "Irecv" not completed at the same time */

  long_long total_wait_time = 0;
  //long_long t;
  //long_long time_start = PAPI_get_real_usec();

  stick_this_thread_to_core(handler->tid);

  /* First convolution iteration (starting with top and bottom rows) */
  conv_subgrid(my_old_grid, my_grid, handler->start, (handler->start + pad_size));
  conv_subgrid(my_old_grid, my_grid, (handler->end - pad_size), handler->end);

  pthread_mutex_lock(&(handler->mutex));
  handler->top_rows_done[0] = 1;
  handler->bot_rows_done[0] = 1;
  pthread_cond_broadcast(&(handler->pad_ready));
  pthread_mutex_unlock(&(handler->mutex));

  /* Send top or bottom rows */
  if(mpi_needed) {
    if(handler->tid == 0) {
      send_position = handler->start;
      recv_position = 0;
      neighbour = handler->rank - 1;
    } else {
      send_position = (handler->end - pad_size);
      recv_position = handler->end;
      neighbour = handler->rank + 1;
    }
    pthread_mutex_lock(&mutex_mpi);
    MPI_Isend(&my_grid[send_position], pad_size, MPI_FLOAT, neighbour, 0, MPI_COMM_WORLD, &request[0]);
    MPI_Irecv(&my_grid[recv_position], pad_size, MPI_FLOAT, neighbour, 0, MPI_COMM_WORLD, &request[2]);
    pthread_mutex_unlock(&mutex_mpi);
  }

  conv_subgrid(my_old_grid, my_grid, (handler->start + pad_size), (handler->end - pad_size));

  /* Second or higher convolution iterations */
  for(uint8_t iter = 1; iter < num_iterations; iter++) {
    temp = my_old_grid;
    my_old_grid = my_grid;
    my_grid = temp;
    prev_iter_index = (iter-1) % 2;
    central_start = handler->start + pad_size;
    memset(completed, 0, sizeof(int) * 3);

    while(!completed[TOP] || !completed[BOTTOM] || !completed[CENTER]) {
      if(!completed[TOP]) {
        update_log(TOP, mpi_needed, prev_iter_index, completed, &request[2], handler, &total_wait_time);

        if(completed[TOP]) {
          conv_subgrid(my_old_grid, my_grid, handler->start, (handler->start + pad_size));
          if(iter+1 < num_iterations) {
            if(handler->tid == 0 && mpi_needed) {
              pthread_mutex_lock(&mutex_mpi);
              MPI_Irecv(&my_grid[recv_position], pad_size, MPI_FLOAT, neighbour, 0, MPI_COMM_WORLD, &request[2]);
              MPI_Isend(&my_grid[send_position], pad_size, MPI_FLOAT, neighbour, 0, MPI_COMM_WORLD, &request[iter % 2]);
              MPI_Wait(&request[prev_iter_index], MPI_STATUS_IGNORE);   /* Avoid to overwrite data of previous Isend with next convolution */
              pthread_mutex_unlock(&mutex_mpi);
            } else {
              pthread_mutex_lock(&(handler->mutex));
              handler->top_rows_done[iter % 2] = 1;
              pthread_cond_broadcast(&(handler->pad_ready));
              pthread_mutex_unlock(&(handler->mutex));
            }
          }
        }
      }

      if(!completed[BOTTOM]) {
        update_log(BOTTOM, mpi_needed, prev_iter_index, completed, &request[2], handler, &total_wait_time);

        if(completed[BOTTOM]) {
          conv_subgrid(my_old_grid, my_grid, (handler->end - pad_size), handler->end);
          if(iter+1 < num_iterations) {
            if(handler->tid == num_threads-1 && mpi_needed) {
              pthread_mutex_lock(&mutex_mpi);
              MPI_Irecv(&my_grid[recv_position], pad_size, MPI_FLOAT, neighbour, 0, MPI_COMM_WORLD, &request[2]);
              MPI_Isend(&my_grid[send_position], pad_size, MPI_FLOAT, neighbour, 0, MPI_COMM_WORLD, &request[iter % 2]);
              MPI_Wait(&request[prev_iter_index], MPI_STATUS_IGNORE);   /* Avoid to overwrite data of previous Isend with next convolution */
              pthread_mutex_unlock(&mutex_mpi);
            } else {
              pthread_mutex_lock(&(handler->mutex));
              handler->bot_rows_done[iter % 2] = 1;
              pthread_cond_broadcast(&(handler->pad_ready));
              pthread_mutex_unlock(&(handler->mutex));
            }
          }
        }
      }

      /* Computing central rows one at a time if top and bottom rows are incomplete */
      if(!completed[CENTER]) {
        int central_end;
        if (completed[TOP] && completed[BOTTOM]) {
          central_end = handler->end - pad_size;
          completed[CENTER] = 1;
        } else {
          central_end = central_start + grid_width;
        }

        conv_subgrid(my_old_grid, my_grid, central_start, central_end);

        if(central_end == (handler->end - pad_size)) completed[CENTER] = 1;
        else central_start += grid_width;
      }
    }
  }

  //t = PAPI_get_real_usec();
  //printf("Thread[%d][%d]: Elapsed time: %llu | Total cond. wait time: %llu\n", handler->rank, handler->tid, (t - time_start), total_wait_time);
  if(handler->tid) pthread_exit(0);
  else return 0;
}

void update_log(uint8_t position, uint8_t mpi_needed, uint8_t index, int* completed, MPI_Request* request, struct thread_handler* handler, long_long* elapsed) {
  int tid;
  uint8_t* rows_done;
  struct thread_handler* neigh_handler;

  switch(position) {
    case TOP:
      tid = 0;
      neigh_handler = handler->top;
      if(neigh_handler == NULL) break;
      rows_done = handler->top->bot_rows_done;
      break;

    case BOTTOM:
      tid = num_threads-1;
      neigh_handler = handler->bottom;
      if(neigh_handler == NULL) break;
      rows_done = handler->bottom->top_rows_done;
      break;

    default:
      return;
  }

  long_long t;
  if(handler->tid == tid && mpi_needed) {
    /* If current thread has distributed memory dependency */
    pthread_mutex_lock(&mutex_mpi);
    MPI_Test(request, &completed[position], MPI_STATUS_IGNORE);
    if(!completed[position] && completed[!position] && completed[CENTER]) {
      //MPI_Waitany(1, &request)
      MPI_Wait(request, MPI_STATUS_IGNORE);
      completed[position] = 1;
    }
    pthread_mutex_unlock(&mutex_mpi);
  } else if(neigh_handler == NULL) {
    /* If current thread is the "highest" or the "lowest" (no dependency with upper or lower thread) */  
    completed[position] = 1;
  } else {
    /* If current thread has a shared memory dependency with upper or lower thread */
    pthread_mutex_lock(&(neigh_handler->mutex));
    completed[position] = rows_done[index];
    if(completed[position]) 
      rows_done[index] = 0;
    else if(completed[!position] && completed[CENTER]) {
      t = PAPI_get_real_usec();
      while(rows_done[index] == 0) {
        pthread_cond_wait(&(neigh_handler->pad_ready), &(neigh_handler->mutex));
      }
      *elapsed += PAPI_get_real_usec() - t;
      rows_done[index] = 0;
      completed[position] = 1;
    }
    pthread_mutex_unlock(&(neigh_handler->mutex));
  }
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
    if(col < num_pads) {
      for(offset = 0; i-offset > row_start && offset <= num_pads; offset++);
      grid_index = i-offset-num_pads;
      kern_index = (kern_width / 2) - offset;
      kern_end = kern_width-kern_index;
      iterations = (num_pads+col+1) *kern_width;
    } else if (col > grid_width-1-num_pads){
      int row_end = row_start + grid_width - 1;
      for(offset = 0; i+offset <= row_end && offset <= num_pads; offset++);
      grid_index = i-num_pads-pad_size;
      kern_index = 0;
      kern_end = kern_width-offset;
      iterations = (num_pads + grid_width-1-col) * kern_width;
    } else {
      grid_index = i-num_pads-pad_size;
      kern_index = 0;
      kern_end = kern_width;
      iterations = kern_size;
    }

    // Convolution
    if(iterations == kern_size) {
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
            //cmp_mask = _mm_cmpeq_ps(vec_kern, vec_kern);               // Comparing NaN value always return false
            //vec_kern = _mm_and_ps(vec_kern, cmp_mask);                 // NaN value are now 0
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

void init_read(FILE *fp_grid, FILE *fp_kernel){
  MPI_Request request[2];

  /* First token represent matrix dimension */
  if(fread(&grid_width, sizeof(uint), 1, fp_grid) != 1 || fread(&kern_width, sizeof(uint32_t), 1, fp_kernel) != 1) {
    fprintf(stderr, "Error in file reading: first element should be the row (or column) length of a square matrix\n");
    exit(-1);
  }

  /* Exchange initial information */
  int to_send[] = {grid_width, kern_width};
  MPI_Ibcast(to_send, 2, MPI_INT, 0, MPI_COMM_WORLD, request);

  /* Non-blank chars + Blank chars */
  /* Kernel data from file */
  kern_size = kern_width * kern_width;
  kernel = malloc(sizeof(float) * kern_size);
  read_float_matrix(fp_kernel, kernel, kern_size);

  /* Exchange kernel */
  MPI_Ibcast(kernel, kern_size, MPI_FLOAT, 0, MPI_COMM_WORLD, &request[1]);
}

void read_data(FILE *fp_grid, float* whole_grid, int rows_per_proc) {
  int start, size;
  MPI_Request req[num_procs];

  /* Grid data from file */
  read_float_matrix(fp_grid, whole_grid, grid_size);

  /* Exchange of read data */
  size = rows_per_proc_size;
  if (num_procs != 1) size += pad_size;
  memcpy(&old_grid[pad_size], whole_grid, size * sizeof(float));    /* rank 0 */
  
  for(int rank = 1; rank < num_procs; rank++) {
    start = (rows_per_proc * rank - num_pads) * grid_width;
    size = rows_per_proc_size + pad_size;
    if(rank != num_procs-1) size += pad_size;
    MPI_Isend(&whole_grid[start], size, MPI_FLOAT, rank, rank, MPI_COMM_WORLD, &req[rank]);
  }
}

void store_data(FILE *fp_result, float *float_buffer, int count){
  /* Buffer size: Num_floats * (Non-blank chars + Blank chars) */
  char* buffer = malloc(count * (grid_width * MAX_CHARS + grid_width) * sizeof(char));
  int offset = 0;
  int limit = grid_width-1;

  /* Write buffer filling */
  for(int i = 0; i < count; i++){
    offset += sprintf(&buffer[offset], "%+e ", float_buffer[i]);
    if (i == limit) {
      limit += grid_width;
      offset += sprintf(&buffer[offset], "\n");
    }
  }
  fwrite(buffer, sizeof(char), offset, fp_result);
  free(buffer);
}

void handle_PAPI_error(int rc, char *msg) {
  char error_str[PAPI_MAX_STR_LEN];
  memset(error_str, 0, sizeof(char)*PAPI_MAX_STR_LEN);

  fprintf(stderr, "%s\nReturn code: %d - PAPI error message:\n", msg, rc);
  PAPI_perror(error_str); PAPI_strerror(rc);
  exit(-1);
}

// Read in binary mode "count" floating point values from "fp" into "buffer" 
void read_float_matrix(FILE* fp, float* buffer, int count) {
  const uint float_read = fread(buffer, sizeof(float), count, fp);

  if(ferror(fp)) {
    perror("Error while reading from file:");
    exit(-1);
  }
  if(float_read < count) {
    fprintf(stderr, "Error in file reading: number of float elements read (%d) is lower than the expected amount (%d)\nEOF %sreached\n", 
      float_read, count, (feof(fp) ? "" : "not "));
    exit(-1);
  }
}

/* Set thread affinity. If there are more threads than cores, no affinity will be set */
int stick_this_thread_to_core(int core_id) {
  const long num_cores = sysconf(_SC_NPROCESSORS_ONLN);
  if(num_threads > num_cores) return 0;
  if(core_id < 0) return 1;
  //if((num_threads * 2) <= num_cores) core_id *= 2;   // Worse performance if Hyper-Threading is avoided

  cpu_set_t cpuset;
  CPU_ZERO(&cpuset);
  CPU_SET(core_id, &cpuset);

  return pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);
}