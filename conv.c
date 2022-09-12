// conv.c
// Name: Tanay Agarwal, Nirmal Krishnan
// JHED: tagarwa2, nkrishn9

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <assert.h>
#include <string.h>
#include <math.h>
#include <papi.h>
#include <mpi.h>
#include <pthread.h>
#include <unistd.h>

#define DEFAULT_ITERATIONS 1
#define DEFAULT_THREADS 2
#define GRID_FILE_PATH "./io-files/grid.txt"
#define KERNEL_FILE_PATH "./io-files/kernel.txt"
#define RESULT_FILE_PATH "./io-files/result.txt"
#define MAX_CHARS 13     /* Standard "%e" format has at most this num of chars (e.g. -9.075626e+20) */
#define ROWS_PER_JOB 8

void* worker_thread(void*);
void deposit_jobs(int, int, int, float*, float*);
void conv_subgrid(float*, float*, int, int);
float normalize(float, float*);
void init_read(FILE*, FILE*);
void read_data(FILE*, int, int, int);
void store_data(FILE*, float*, int);
void handle_PAPI_error(int, char*);

struct job {
    int start, end, iter;
};

struct pthread_args {
    int tid;
    //int rank;
};
pthread_mutex_t mutex_job = PTHREAD_MUTEX_INITIALIZER;
pthread_mutex_t mutex_log = PTHREAD_MUTEX_INITIALIZER;
pthread_cond_t pad_done = PTHREAD_COND_INITIALIZER;
pthread_cond_t not_full = PTHREAD_COND_INITIALIZER;
pthread_cond_t not_empty = PTHREAD_COND_INITIALIZER;
int front = 0, rear = 0, count = 0;

struct job *jobs;             /* Queue of jobs */
uint64_t* iteration_log;      /* 64 bit map of completed job (in a current iteration) */
uint8_t num_pads;             /* Number of rows that should be shared with other processes */
uint8_t kern_width;           /* Number of elements in one kernel matrix row */
uint16_t grid_width;          /* Number of elements in one grid matrix row */
uint64_t grid_size;           /* Number of elements in whole grid matrix */
uint16_t kern_size;           /* Number of elements in whole kernel matrix */
uint16_t pad_size;            /* Number of elements in the pad section of the grid matrix */
int job_size;                 /* Number of elements in a job */
int last_job;                 /* Row index of last job (bordering bottom pads) */
int num_procs;                /* Number of MPI processes in the communicator */
int num_threads;              /* Number of threads for every MPI process */
int num_iterations;           /* Number of convolution operations */
int num_jobs;                 /* Number of jobs storable in a shared buffer */
float kern_dot_sum;           /* Used for normalization, its value is equal to: sum(dot(kernel, kernel)) */
float *kernel;                /* Kernel buffer */
float *grid;                  /* Grid buffer */
float *old_grid;              /* Old grid buffer */


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
  pthread_t threads[num_threads];
  
  /* MPI setup */
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

  /* Opening input files in dir "./io-files" */
  if((fp_grid = fopen(GRID_FILE_PATH, "r")) == NULL) {
    fprintf(stderr, "Error while opening grid file\n");
    exit(-1);
  }
  if((fp_kernel = fopen(KERNEL_FILE_PATH, "r")) == NULL) {
    fprintf(stderr, "Error while opening kernel file\n");
    exit(-1);
  }
  init_read(fp_grid, fp_kernel);
  assert(grid_width % num_procs == 0);
  assert(ROWS_PER_JOB > num_pads);

  /* Computation of sum(dot(kernel, kernel)) */
  for(int pos = 0; pos < kern_size; pos++){
    kern_dot_sum += kernel[pos] * kernel[pos];
  }

  /* Data splitting and variable initialization */
  const int grid_height = (grid_width / num_procs);             /* Grid matrix is square, so grid_width and grid_height are the same before data splitting */
  const int start = grid_height * rank;                         /* Index of the first row for current process */
  const int end = grid_height - 1 + start;                      /* Index of the final row for current process */
  const int rows_per_proc = end + 1 - start;                    /* Number of rows assigned to a process */
  const int rows_per_proc_size = rows_per_proc*grid_width;      /* Number of elements assigned to a process */
  const uint8_t next = (rank != num_procs-1) ? rank+1 : 0;      /* Rank of process having rows next to this process */
  const uint8_t prev = (rank != 0) ? rank-1 : num_procs-1;      /* Rank of process having rows prev to this process */
  const int total_jobs = (grid_width / num_procs) / ROWS_PER_JOB;
  iteration_log = calloc(sizeof(uint64_t), (total_jobs / 64)+1);

  /* Read grid data */
  grid = malloc((rows_per_proc + num_pads*2) * grid_width * sizeof(float));
  old_grid = malloc((rows_per_proc + num_pads*2) * grid_width * sizeof(float));
  if(!rank){
    memset(grid, 0, pad_size * sizeof(float));
    memset(old_grid, 0, pad_size * sizeof(float));
  }
  else if(rank == num_procs-1){
    memset(&grid[(rows_per_proc+num_pads) * grid_width], 0, pad_size * sizeof(float));
    memset(&old_grid[(rows_per_proc+num_pads) * grid_width], 0, pad_size * sizeof(float));
  }
  read_data(fp_grid, start, rows_per_proc, rank);

  /* First iteration - No rows exchange needed */
  time_start = PAPI_get_real_usec();
  float *my_old_grid = old_grid;
  float *my_grid = grid;
  float *temp;

  /* PThreads creation */
  struct pthread_args* args = malloc(sizeof(struct pthread_args) * num_threads);
  for(int i = 0; i < num_threads; i++) {
    args[i].tid = i;
    //args[i].rank = rank;
    rc = pthread_create(&threads[i], NULL, worker_thread, (void *)&args[i]);
    if (rc) { 
      fprintf(stderr, "Error while creating pthread[%d]; Return code: %d\n", i, rc);
      exit(-1);
    }
  }

  /* Insert high priority job first (the bottom rows bordering pads) */
  last_job = (grid_height + num_pads - ROWS_PER_JOB) * grid_width;
  pthread_mutex_lock(&mutex_job);
  while(count == num_jobs) {
    pthread_cond_wait(&not_full, &mutex_job);
  }
  jobs[rear].start = last_job;
  jobs[rear].end = rows_per_proc_size+pad_size;
  jobs[rear].iter = 0;
  rear = (rear+1) % num_jobs;
  count++;
  pthread_cond_signal(&not_empty);
  pthread_mutex_unlock(&mutex_job);
  
  /* Deposit all remaining jobs (index after initial thread creation job) */
  deposit_jobs(pad_size + num_threads * job_size, last_job, 0, my_old_grid, my_grid);
  /*int shift = (total_jobs-1) % 64;
  int arr_index = (total_jobs-1) / 64;
  uint64_t mask = (uint64_t) 1 << shift;*/
  
  uint64_t end_mask = UINT64_MAX >> (64 - total_jobs);
  pthread_mutex_lock(&mutex_log);
  //while( !(iteration_log[0] & 1)  || !(iteration_log[arr_index] & mask)) {
  while(iteration_log[0] != end_mask) {
    pthread_cond_wait(&pad_done, &mutex_log);
  }
  memset(iteration_log, 0, sizeof(uint64_t)*total_jobs/64);
  pthread_mutex_unlock(&mutex_log);

  /* Second (or higher) iterations */
  for(uint8_t iters = 1; iters < num_iterations; iters++) {
    /* Swap grid pointers */
    temp = my_old_grid;
    my_old_grid = my_grid;
    my_grid = temp;

    if(num_procs > 1) {
      if (!rank) {
        /* Process with rank 0 doesn't have a "prev" process */
        MPI_Isend(&my_old_grid[rows_per_proc_size], pad_size, MPI_FLOAT, next, 0, MPI_COMM_WORLD, &req[1]);
        MPI_Irecv(&my_old_grid[rows_per_proc_size+pad_size], pad_size, MPI_FLOAT, next, 1, MPI_COMM_WORLD, req);
      } else if (rank == num_procs-1) {
        /* Last process doesn't have a "next" process */
        MPI_Isend(&my_old_grid[pad_size], pad_size, MPI_FLOAT, prev, 1, MPI_COMM_WORLD, &req[1]);
        MPI_Irecv(my_old_grid, pad_size, MPI_FLOAT, prev, 0, MPI_COMM_WORLD, req);
      } else {
        /* Every other process */
        MPI_Isend(&my_old_grid[pad_size], pad_size, MPI_FLOAT, prev, 1, MPI_COMM_WORLD, &req[2]);
        MPI_Irecv(my_old_grid, pad_size, MPI_FLOAT, prev, 0, MPI_COMM_WORLD, req);
        MPI_Isend(&my_old_grid[rows_per_proc_size], pad_size, MPI_FLOAT, next, 0, MPI_COMM_WORLD, &req[3]);
        MPI_Irecv(&my_old_grid[rows_per_proc_size+pad_size], pad_size, MPI_FLOAT, next, 1, MPI_COMM_WORLD, &req[1]);
      }
    }

    /* Start convolution only of central grid elements */
    int curr_job = pad_size + job_size;
    deposit_jobs(curr_job, last_job, iters, my_old_grid, my_grid);

    /* Convolution of top and bottom rows (the ones bordering pad rows) */
    if(num_procs > 1) {
      if(!rank || rank == num_procs-1)
        rc = MPI_Wait(req, status);
      else 
        rc = MPI_Waitall(2, req, status);
    }

    /* Deposit jobs for pad convolution */
    pthread_mutex_lock(&mutex_job);
    while(count > num_jobs-2) {
      pthread_cond_wait(&not_full, &mutex_job);
    }
    jobs[rear].start = pad_size;
    jobs[rear].end = pad_size + job_size;
    jobs[rear].iter = iters;
    rear = (rear+1) % num_jobs;
    jobs[rear].start = last_job;
    jobs[rear].end = rows_per_proc_size + pad_size;
    jobs[rear].iter = iters;
    rear = (rear+1) % num_jobs;
    count+=2;
    pthread_cond_signal(&not_empty);
    pthread_mutex_unlock(&mutex_job);

    /* Wait */
    pthread_mutex_lock(&mutex_log);
    //while( !(iteration_log[0] & 1)  || !(iteration_log[arr_index] & mask)) {
    while(iteration_log[0] != end_mask) {
      pthread_cond_wait(&pad_done, &mutex_log);
    }
    memset(iteration_log, 0, sizeof(uint64_t)*total_jobs/64+1);
    pthread_mutex_unlock(&mutex_log);
  }

  /* Deposit jobs for termination */
  for(int i = 0; i < num_threads; i++) {
    pthread_mutex_lock(&mutex_job);
    while(count == num_jobs) {
      pthread_cond_wait(&not_full, &mutex_job);
    }
    jobs[rear].iter = num_iterations;
    rear = (rear+1) % num_jobs;
    count++;
    pthread_cond_signal(&not_empty);
    pthread_mutex_unlock(&mutex_job);
  }

  MPI_Barrier(MPI_COMM_WORLD);
  if(!rank) {
    time_stop = PAPI_get_real_usec();
    printf("(PAPI) Elapsed time: %lld us\n", (time_stop - time_start));
  }

  float *write_buffer = malloc(grid_size * sizeof(float) - grid_width * (grid_width / num_procs));
  if(rank != 0) {
    MPI_Isend(&my_grid[pad_size], rows_per_proc_size, MPI_FLOAT, 0, 11, MPI_COMM_WORLD, req);
  } else {
    for(int k = 0; k < num_procs-1; k++) {
      MPI_Irecv(&write_buffer[grid_width * (grid_width / num_procs) * k], rows_per_proc_size, MPI_FLOAT, k+1, 11, MPI_COMM_WORLD, &req[k]);
    }
  }

  /* Stop the count! */ 
  if ((rc = PAPI_stop(event_set, &num_cache_miss)) != PAPI_OK)
    handle_PAPI_error(rc, "Error in PAPI_stop().");
  printf("Rank: %d, total cache misses:%lld\n", rank, num_cache_miss);
  
  /* Store computed matrix */
  if (!rank) {
    FILE *fp_result;
    if((fp_result = fopen(RESULT_FILE_PATH, "w")) == NULL) {
      fprintf(stderr, "Error while creating and/or opening result file\n");
      exit(-1);
    }
    store_data(fp_result, &my_grid[pad_size], rows_per_proc_size);
    MPI_Waitall(num_procs-1, req, status);
    store_data(fp_result, write_buffer, grid_size-rows_per_proc_size);
    fclose(fp_result);
  }
  
  MPI_Finalize();
  free(write_buffer);
  free(iteration_log);
  free(args);
  free(jobs);
  free(grid);
  free(old_grid);
  free(kernel);
  pthread_exit(0);
}

void deposit_jobs(int curr_job, int end_job, int iter, float *my_old_grid, float *my_grid){
  int temp;
  while(curr_job < end_job) {
    pthread_mutex_lock(&mutex_job);
    while(count == num_jobs) {
      /* Manager does some work if buffer is full */
      if(curr_job < end_job) {
        pthread_mutex_unlock(&mutex_job);
        temp = curr_job;
        curr_job += job_size;
        conv_subgrid(my_old_grid, my_grid, temp, curr_job);
        pthread_mutex_lock(&mutex_job);
      } else
        pthread_cond_wait(&not_full, &mutex_job);
    }
    while(count != num_jobs && curr_job < end_job) {
      jobs[rear].start = curr_job;
      jobs[rear].end = curr_job + job_size;
      jobs[rear].iter = iter;
      rear = (rear+1) % num_jobs;
      count++;
      curr_job += job_size;
    }
    pthread_cond_signal(&not_empty);
    pthread_mutex_unlock(&mutex_job);
  }
}

void* worker_thread(void* args){
  struct pthread_args *my_args = (struct pthread_args*)args;
  int start = pad_size + my_args->tid * job_size;
  int end = start + job_size;
  int iteration = 0, temp_iter;
  float *my_old_grid = old_grid;
  float *my_grid = grid;
  float *temp;
  long_long time_start, time_stop;
  long_long total_wait_time = 0;

  /* All threads starts with an initial job after creation */
  conv_subgrid(my_old_grid, my_grid, start, end);
  while(1) {
    time_start = PAPI_get_real_usec();
    pthread_mutex_lock(&mutex_job);
    while(count == 0) {
      pthread_cond_wait(&not_empty, &mutex_job);
    }
    time_stop = PAPI_get_real_usec();

    /* Fetching a job from jobs queue */
    start = jobs[front].start;
    end = jobs[front].end;
    temp_iter = jobs[front].iter;
    front = (front+1) % num_jobs;
    count--;
    pthread_cond_signal(&not_full);
    pthread_mutex_unlock(&mutex_job);
    total_wait_time += time_stop - time_start;
    
    /* Check current iteration */
    if(temp_iter != iteration){
      if(temp_iter == num_iterations) {
        printf("Thread[%d]: Job queue waiting time: %llu us\n", my_args->tid, total_wait_time);
        pthread_exit(0);
      }
      iteration = temp_iter;
      temp = my_old_grid;
      my_old_grid = my_grid;
      my_grid = temp;
    }

    /* Job computing */
    conv_subgrid(my_old_grid, my_grid, start, end);
  }
}

void log_job(int start){
  int job_num = (start-pad_size) / job_size;
  int shift = job_num % 64;
  int arr_index = job_num / 64;
  pthread_mutex_lock(&mutex_log);
  iteration_log[arr_index] |= (uint64_t) 1 << shift;
  //if(start == pad_size || start == last_job) {
  pthread_cond_signal(&pad_done);
  pthread_mutex_unlock(&mutex_log);
}

void conv_subgrid(float *sub_grid, float *new_grid, int start_index, int end_index) {
  float result;
  float matrix[kern_size];                 /* Temp buffer used for normalization */
  int col = start_index % grid_width;      /* Index of current column */
  int row_start = start_index - col;       /* Index of the first element in current row */

  int offset;                              /* How far is current element from its closest border */
  int grid_index;
  int kern_index;
  int kern_end;                            /* Describes when it's time to change row */
  int iterations;

  for(int i = start_index; i < end_index; i++) {
    /* Setting indexes for current element */
    if(col < num_pads) {
      for(offset = 0; i-offset > row_start && offset <= num_pads; offset++);
      grid_index = i-offset-pad_size;
      kern_index = (kern_width >> 1) - offset;
      kern_end = kern_width-kern_index;
      iterations = (num_pads+col+1) *kern_width;
      memset(matrix, 0, kern_size*sizeof(float));
    } else if (col > grid_width-1-num_pads){
      int row_end = row_start + grid_width - 1;
      for(offset = 0; i+offset <= row_end && offset <= num_pads; offset++);
      grid_index = i-num_pads-pad_size;
      kern_index = 0;
      kern_end = kern_width-offset;
      iterations = (num_pads + grid_width-col) *kern_width;
      memset(matrix, 0, kern_size*sizeof(float));
    } else {
      grid_index = i-num_pads-pad_size;
      kern_index = 0;
      kern_end = kern_width;
      iterations = kern_size;
    }

    /* Convolution */
    result = 0; offset = 0;
    for (int iter=0; iter < iterations; iter++) {
      result += sub_grid[grid_index+offset] * kernel[kern_index+offset];
      matrix[kern_index+offset] = sub_grid[grid_index+offset];
      if (offset != kern_end-1) 
        offset++;
      else { 
        grid_index += grid_width;
        kern_index += kern_width;
        offset = 0;
      }
    }

    new_grid[i] = normalize(result, matrix);

    /* Setting row and col index for next element */
    if (col != grid_width-1)
      col++;
    else{
      row_start += grid_width;
      col = 0;
    }
  }
  log_job(start_index);
}

float normalize(float conv_res, float *matrix) {
  float matrix_dot_sum = 0;
  for(int pos = 0; pos < kern_size; pos++){
    matrix_dot_sum += matrix[pos] * matrix[pos];
  }

  float res = conv_res / sqrt(matrix_dot_sum * kern_dot_sum);
  /* Resolution problem if too many convolution iteration are done. The mean value is returned */
  if(isnan(res)) {
    return 0;
  }
  return res;
}

void init_read(FILE *fp_grid, FILE *fp_kernel){
  char *buffer;
  int kern_row_chars, buffer_size;
  int i, offset;

  /* First token represent matrix dimension */
  if(fscanf(fp_grid, "%hd\n", &grid_width) == EOF || fscanf(fp_kernel, "%hhd\n", &kern_width) == EOF) {
    fprintf(stderr, "Error in file reading: first element should be the row (or column) length of a square matrix\n");
    exit(-1);
  }
  grid_size = grid_width * grid_width;
  kern_size = kern_width * kern_width;
  num_pads = (kern_width - 1) / 2;
  pad_size = grid_width * num_pads;
  job_size = grid_width * ROWS_PER_JOB;
  
  /* Non-blank chars + Blank chars */
  kern_row_chars = (kern_width * MAX_CHARS + kern_width) * sizeof(char);
  /* Kernel data from file */
  buffer_size = kern_row_chars * kern_width;
  buffer = malloc(sizeof(char) * buffer_size);
  kernel = malloc(sizeof(float) * kern_size);
  buffer_size = fread(buffer, sizeof(char), buffer_size, fp_kernel);
  fclose(fp_kernel);

  offset = 0;
  for(i = 0; i < kern_size && offset < buffer_size; i++) {
    kernel[i] = atof(&buffer[offset]);
    while(buffer[offset] >= '+') offset++;
    while(buffer[offset] != '\0' && buffer[offset] < '+') offset++;
  }

  if(i != kern_size) {
    fprintf(stderr, "Error in file reading: number of kernel elements read is different from the expected amount\n");
    exit(-1);
  }

  /* Jobs buffer */
  num_jobs = num_threads * 2;
  jobs = malloc(sizeof(struct job) * num_jobs);
  free(buffer);
}

void read_data(FILE *fp_grid, int start, int rows_per_proc, int rank) {
  /* Positive non-blank chars + Blank chars */
  const int grid_row_chars = (grid_width * (MAX_CHARS-1) + grid_width) * sizeof(char);
  int offset = 0;
  int i = 0;
  int buffer_size = rows_per_proc + num_pads;
  int iterations = rows_per_proc + num_pads*2;
  char* buffer;

  if(rank) {
    start -= num_pads;
    if(rank != num_procs-1) buffer_size += num_pads;
    else iterations -= num_pads;
  } 
  else {
    i = pad_size;
    if(num_procs == 1) iterations -= num_pads;
  }
  
  start *= grid_row_chars;
  buffer_size *= grid_row_chars;
  iterations *= grid_width;

  /* Grid data from file */  
  if(start != 0 && fseek(fp_grid, start, SEEK_CUR)) {
    fprintf(stderr, "Error while executing fseek\n");
    exit(-1);
  }
  buffer = malloc(buffer_size * sizeof(char));
  buffer_size = fread(buffer, sizeof(char), buffer_size, fp_grid);
  fclose(fp_grid);
  
  for(; i < iterations && offset < buffer_size; i++) { 
    old_grid[i] = atof(&buffer[offset]);
    while(buffer[offset] >= '+') offset++;
    while(buffer[offset] != '\0' && buffer[offset] < '+') offset++;
  }

  if(i != iterations) {
    fprintf(stderr, "Error in file reading: number of grid elements read is different from the expected amount\n");
    exit(-1);
  }
  free(buffer);
}

void store_data(FILE *fp_result, float *float_buffer, int count){
  /* Buffer size: Num_floats * (Non-blank chars + Blank chars) */
  char* buffer = malloc(count * (grid_width * MAX_CHARS + grid_width) * sizeof(char));
  int offset = 0;
  int limit = grid_width-1;

  /* Write buffer filling */
  for(int i = 0; i < count; i++){
    offset += sprintf(&buffer[offset], "%e ", float_buffer[i]);
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
