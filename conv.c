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
#define INT_NUM_BITS (8 * sizeof(int))
#define BM_VALUE(LOG, JOB_NUM) (((LOG)[(JOB_NUM) / INT_NUM_BITS] & (1 << ((JOB_NUM) % INT_NUM_BITS))) != 0)

void* worker_thread(void*);
void help_worker_threads(float*, float*, uint*);
void deposit_jobs(float*, float*, uint*, uint*, int, int, int);
int halfiteration_deposit(float*, float*, uint*, uint*, uint*, int);
void deposit_pad_jobs(int);
int check_pad_jobs_dependencies(MPI_Request*, MPI_Status*, uint*, int*, int, int);
void wait_pads_completion(uint*);
int is_log_high(uint*);
void conv_subgrid(float*, float*, uint*, int, int);
void log_job(uint*, int);
void init_read(FILE*, FILE*);
void read_data(FILE*, float*, int);
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
pthread_cond_t job_done = PTHREAD_COND_INITIALIZER;
pthread_cond_t not_full = PTHREAD_COND_INITIALIZER;
pthread_cond_t not_empty = PTHREAD_COND_INITIALIZER;
int front = 0, rear = 0, count = 0;

struct job *jobs;             /* Queue of jobs */
uint* curr_iteration_log;     /* Bit map of completed job for current convolution iteration */
uint* next_iteration_log;     /* Bit map of completed job for next convolution iteration */
uint8_t num_pads;             /* Number of rows that should be shared with other processes */
uint8_t kern_width;           /* Number of elements in one kernel matrix row */
uint16_t grid_width;          /* Number of elements in one grid matrix row */
uint64_t grid_size;           /* Number of elements in whole grid matrix */
uint16_t kern_size;           /* Number of elements in whole kernel matrix */
uint16_t pad_size;            /* Number of elements in the pad section of the grid matrix */
int rows_per_proc_size;       /* Number of elements assigned to a process */
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

  if(!rank) {
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
  job_size = grid_width * ROWS_PER_JOB;
  num_jobs = num_threads * 2;
  jobs = malloc(sizeof(struct job) * num_jobs);                 /* Jobs buffer */
  assert(grid_width % num_procs == 0);
  assert(ROWS_PER_JOB > num_pads);

  /* Data splitting and variable initialization */
  const uint8_t next = (rank != num_procs-1) ? rank+1 : 0;      /* Rank of process having rows next to this process */
  const uint8_t prev = (rank != 0) ? rank-1 : num_procs-1;      /* Rank of process having rows prev to this process */
  const int rows_per_proc = grid_width / num_procs;             /* Number of rows assigned to current process */
  rows_per_proc_size = rows_per_proc * grid_width;              /* Number of elements assigned to a process */
  total_jobs = rows_per_proc / ROWS_PER_JOB;
  log_buff_size = (total_jobs + INT_NUM_BITS-1) / INT_NUM_BITS;
  curr_iteration_log = calloc(sizeof(uint), log_buff_size);
  next_iteration_log = calloc(sizeof(uint), log_buff_size);
  uint* inserted_log = calloc(sizeof(uint), log_buff_size);
  uint* log_copy = malloc(sizeof(uint) * log_buff_size);

  /* Read grid data */
  grid = malloc((rows_per_proc + num_pads*2) * grid_width * sizeof(float));
  old_grid = malloc((rows_per_proc + num_pads*2) * grid_width * sizeof(float));
  float* whole_grid;
  if(!rank){
    memset(grid, 0, pad_size * sizeof(float));
    memset(old_grid, 0, pad_size * sizeof(float));
    whole_grid = malloc(grid_size * sizeof(float));
    read_data(fp_grid, whole_grid, rows_per_proc);
  } else if(rank == num_procs-1){
    memset(&grid[(rows_per_proc_size+pad_size)], 0, pad_size * sizeof(float));
    memset(&old_grid[(rows_per_proc_size+pad_size)], 0, pad_size * sizeof(float));
    MPI_Recv(old_grid, (rows_per_proc_size+pad_size), MPI_FLOAT, 0, rank, MPI_COMM_WORLD, &status[rank]);
  } else {
    MPI_Recv(old_grid, rows_per_proc_size+pad_size*2, MPI_FLOAT, 0, rank, MPI_COMM_WORLD, &status[rank]);
  }

  /* First iteration - No rows exchange needed */
  time_start = PAPI_get_real_usec();
  float *my_old_grid = old_grid;
  float *my_grid = grid;
  float *temp_grid;
  uint *my_curr_log = curr_iteration_log;
  uint *my_next_log = next_iteration_log;
  uint *temp_log;

  if(rank) MPI_Wait(req, status);                               /* Complete kernel receive */
  for(int pos = 0; pos < kern_size; pos++){                     /* Computation of sum(dot(kernel, kernel)) */
    kern_dot_sum += kernel[pos] * kernel[pos];
  }

  /* PThreads creation (every thread starts with a default job) */
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
  last_job = (rows_per_proc + num_pads - ROWS_PER_JOB) * grid_width;
  pthread_mutex_lock(&mutex_job);
  while(count == num_jobs) {
    pthread_cond_wait(&not_full, &mutex_job);
  }
  jobs[rear].start = last_job;
  jobs[rear].end = last_job + job_size;
  jobs[rear].iter = 0;
  rear = (rear+1) % num_jobs;
  count++;
  pthread_cond_signal(&not_empty);
  pthread_mutex_unlock(&mutex_job);

  /* Deposit all remaining jobs */
  deposit_jobs(my_old_grid, my_grid, NULL, my_curr_log, (pad_size + num_threads * job_size), last_job, 0);

  /* flags[0]: if second and penultimate jobs has been completed 
   * flags[1]: if new pads has been received (pointless if there is only one process) */
  int flags[2];
  if(num_procs == 1) flags[1] = 1;

  /* Second (or higher) iterations */
  for(uint8_t iter = 1; iter < num_iterations; iter++) {
    temp_grid = my_old_grid;
    my_old_grid = my_grid;
    my_grid = temp_grid;

    flags[0] = 0;
    if(num_procs > 1) flags[1] = 0;
    
    /* Jobs belonging to previous iteration are still in jobs buffer (or under processing by worker
     * threads). To send my top and bottom rows (bordering pads) their jobs have to be done */
    wait_pads_completion(my_curr_log);

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

    /* Next iteration pad jobs have the priority, insert them as soon as possible */
    rc = check_pad_jobs_dependencies(req, status, my_curr_log, flags, rank, iter);
    if(rc) deposit_pad_jobs(iter);
    
    /* Before inserting the next iteration jobs (except for pads), main thread will fetch some job. After this call, the  
     * situation will be: count <= num_threads and there are at most "num_threads" jobs belonging to previous iteration */
    help_worker_threads(my_grid, my_old_grid, my_curr_log);

    if(!rc) {
      rc = check_pad_jobs_dependencies(req, status, my_curr_log, flags, rank, iter);
      if(rc) deposit_pad_jobs(iter);
    }

    /* Will be deposited only the jobs belonging to the next iteration (giving priority to pad jobs if not
     * already inserted). Jobs of previous iteration are all already inserted, but not completed yet (probably) */
    int prev_iter_completed;
    int inserted, total_inserted = 0;

    pthread_mutex_lock(&mutex_log);
    memcpy(log_copy, my_curr_log, sizeof(uint) * log_buff_size);
    pthread_mutex_unlock(&mutex_log);
    prev_iter_completed = is_log_high(log_copy);
    while(!prev_iter_completed && total_inserted < total_jobs-2) {
      inserted = halfiteration_deposit(my_old_grid, my_grid, inserted_log, log_copy, my_next_log, iter);
      if(!rc) {
        rc = check_pad_jobs_dependencies(req, status, my_curr_log, flags, rank, iter);
        if(rc) deposit_pad_jobs(iter); 
      }

      /* Checking if current log_copy is completely consumed (its jobs are all inserted) */
      if(total_inserted != (total_inserted + inserted)) {
        total_inserted += inserted;
        continue;
      }

      /* Log_copy is completely consumed. If the shared log is different from the copy then copy it again, else wait */
      int i = 0;
      pthread_mutex_lock(&mutex_log);
      for(; i < log_buff_size && log_copy[i] == my_curr_log[i]; i++)
      if (i == log_buff_size)
        pthread_cond_wait(&job_done, &mutex_log);
      memcpy(log_copy, my_curr_log, sizeof(uint) * log_buff_size);
      pthread_mutex_unlock(&mutex_log);
      prev_iter_completed = is_log_high(log_copy);
    }
    /* At this point, all jobs of the previous iteration are done */

    /* Final pad jobs dependency check (this time will wait if necessary) */
    if(!rc) {
      if(!flags[0]){
        pthread_mutex_lock(&mutex_log);
        while(!(my_curr_log[0] & 0x2) || !BM_VALUE(my_curr_log, total_jobs-2)) {
          pthread_cond_wait(&job_done, &mutex_log);
        }
        pthread_mutex_unlock(&mutex_log);
      }
      if(num_procs > 1 && !flags[1]) {
        int num = (!rank || rank == num_procs-1) ? 2 : 4;
        MPI_Waitall(num, req, status);
      }
      deposit_pad_jobs(iter);
    }
    
    /* Reset and swap logs for next iteration */
    pthread_mutex_lock(&mutex_log);
    memset(my_curr_log, 0, sizeof(uint) * log_buff_size);
    pthread_mutex_unlock(&mutex_log);
    temp_log = my_curr_log;
    my_curr_log = my_next_log;
    my_next_log = temp_log;

    /* Deposit all jobs left (avoiding the already inserted ones) */
    deposit_jobs(my_old_grid, my_grid, inserted_log, my_curr_log, (pad_size + job_size), last_job, iter);
    memset(inserted_log, 0, sizeof(uint) * log_buff_size);
  }

  help_worker_threads(my_old_grid, my_grid, my_curr_log);

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

  /* Wait workers termination */
  pthread_mutex_lock(&mutex_log);
  while(!is_log_high(my_curr_log)) {
    pthread_cond_wait(&job_done, &mutex_log);
  }
  pthread_mutex_unlock(&mutex_log);

  time_stop = PAPI_get_real_usec();
  printf("Rank[%d] | Elapsed time: %lld us\n", rank, (time_stop - time_start));
  
  float *write_buffer = malloc(grid_size * sizeof(float) - grid_width * (grid_width / num_procs));
  if(rank != 0) {
    MPI_Send(&my_grid[pad_size], rows_per_proc_size, MPI_FLOAT, 0, 11, MPI_COMM_WORLD);
    //MPI_Isend(&my_grid[pad_size], rows_per_proc_size, MPI_FLOAT, 0, 11, MPI_COMM_WORLD, req);
  } else {
    // wb
    for(int k = 0; k < num_procs-1; k++) {
      //MPI_Irecv(&write_buffer[grid_width * (grid_width / num_procs) * k], rows_per_proc_size, MPI_FLOAT, k+1, 11, MPI_COMM_WORLD, &req[k]);
      MPI_Recv(&write_buffer[grid_width * (grid_width / num_procs) * k], rows_per_proc_size, MPI_FLOAT, k+1, 11, MPI_COMM_WORLD, status);
    }
  }

  /* Stop the count! */ 
  if ((rc = PAPI_stop(event_set, &num_cache_miss)) != PAPI_OK)
    handle_PAPI_error(rc, "Error in PAPI_stop().");
  printf("Rank[%d] | Total L2 cache misses:%lld\n", rank, num_cache_miss);
  
  /* Store computed matrix */
  if (!rank) {
    FILE *fp_result;
    if((fp_result = fopen(RESULT_FILE_PATH, "w")) == NULL) {
      fprintf(stderr, "Error while creating and/or opening result file\n");
      exit(-1);
    }
    store_data(fp_result, &my_grid[pad_size], rows_per_proc_size);
    //MPI_Waitall(num_procs-1, req, status);
    store_data(fp_result, write_buffer, grid_size - rows_per_proc_size);
    fclose(fp_result);
  }

  MPI_Barrier(MPI_COMM_WORLD);
  MPI_Finalize();
  free(write_buffer);
  free(curr_iteration_log);
  free(next_iteration_log);
  free(log_copy);
  free(args);
  free(jobs);
  free(grid);
  free(old_grid);
  if(!rank) free(whole_grid);
  free(kernel);
  exit(0);
}

void* worker_thread(void* args){
  struct pthread_args *my_args = (struct pthread_args*)args;
  int start = pad_size + my_args->tid * job_size;
  int end = start + job_size;
  int temp_iter, iter = 0;
  float *my_old_grid = old_grid;
  float *my_grid = grid;
  uint* my_curr_log = curr_iteration_log;
  long_long time_start, time_stop;
  long_long total_wait_time = 0;

  /* All threads starts with an initial job after creation */
  conv_subgrid(my_old_grid, my_grid, my_curr_log, start, end);

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
    if(temp_iter != iter){
      if(temp_iter == num_iterations) {
        printf("Thread[%d]: Job queue waiting time: %llu us\n", my_args->tid, total_wait_time);
        pthread_exit(0);
      }
      iter = temp_iter;
      /* Swap grid and log pointers */
      if(iter % 2) {
        my_old_grid = grid;
        my_grid = old_grid;
        my_curr_log = next_iteration_log;
      } else {
        my_old_grid = old_grid;
        my_grid = grid;
        my_curr_log = curr_iteration_log;
      }
    }

    /* Job computing */
    conv_subgrid(my_old_grid, my_grid, my_curr_log, start, end);
  }
}

void help_worker_threads(float* my_old_grid, float* my_grid, uint* iteration_log) {
  int start, end;
  pthread_mutex_lock(&mutex_job);
  while(count > num_threads) {
    /* Fetching a job from jobs queue */
    start = jobs[front].start;
    end = jobs[front].end;
    front = (front+1) % num_jobs;
    count--;
    pthread_mutex_unlock(&mutex_job);
    
    /* Job computing */
    conv_subgrid(my_old_grid, my_grid, iteration_log, start, end);
    pthread_mutex_lock(&mutex_job);
  }
  pthread_mutex_unlock(&mutex_job);
}

void deposit_jobs(float *my_old_grid, float *my_grid, uint* inserted_log, uint* iteration_log, int curr_job, int end_job, int iter){
  int odd = 0;
  int curr = 0; 
  int last;
  int next[total_jobs];
  
  /* Store indexes of jobs already done (to avoid re-insertion in jobs buffer) */
  if (inserted_log != NULL) {
    for(int i = 1;  i < total_jobs-1; i++) {
      if(BM_VALUE(inserted_log, i)) {
        next[curr] = i * job_size + pad_size;
        curr++;
      }
    }
  }
  
  /* Start job insertion */
  last = curr-1;
  curr = 0;
  while(curr_job < end_job) {
    if(curr <= last && curr_job == next[curr]) {
      curr_job += job_size;
      curr += 1;
      continue;
    }

    pthread_mutex_lock(&mutex_job);
    while(count == num_jobs) {
      /* Manager does some work if buffer is full */
      if(curr_job < end_job) {
        if(curr <= last && curr_job == next[curr]) {
          curr_job += job_size;
          curr += 1;
          continue;
        }
        pthread_mutex_unlock(&mutex_job);
        conv_subgrid(my_old_grid, my_grid, iteration_log, curr_job, curr_job+job_size);
        curr_job += job_size;
        pthread_mutex_lock(&mutex_job);
      } else 
        pthread_cond_wait(&not_full, &mutex_job);
    }

    /* Insert jobs in this order: first, last, second, penultimate, ... */
    while(count != num_jobs && curr_job < end_job) {
      if(odd % 2) {
        if(curr <= last && curr_job == next[curr]) {
          curr_job += job_size;
          curr += 1;
          continue;
        }
        jobs[rear].start = curr_job;
        jobs[rear].end = curr_job + job_size;
        curr_job += job_size;
      } else {
        if(curr <= last && (end_job-job_size) == next[last]) {
          end_job -= job_size;
          last -= 1;
          continue;
        }
        jobs[rear].start = end_job-job_size;
        jobs[rear].end = end_job;
        end_job -= job_size;
      }
      jobs[rear].iter = iter;
      rear = (rear+1) % num_jobs;
      count++;
      odd++;
    }
    pthread_cond_signal(&not_empty);
    pthread_mutex_unlock(&mutex_job);
  }
}

int halfiteration_deposit(float* my_old_grid, float* my_grid, uint* inserted_log, uint* curr_log_copy, uint* next_log, int iter) {
  int inserted = 0;
  int odd = 0;
  int condition, job, job_start;

  for(int i = 1; i < (total_jobs)/2 && inserted < num_threads;) {
    job = (odd % 2) ? i : (total_jobs-1-i);

    /* Check if a not already inserted job has previous, current and next job completed */
    condition = !BM_VALUE(inserted_log, job) && BM_VALUE(curr_log_copy, job-1) 
             && BM_VALUE(curr_log_copy, job) && BM_VALUE(curr_log_copy, job+1);

    if(condition) {
      job_start = job * job_size + pad_size;
      inserted += 1;
      inserted_log[job / INT_NUM_BITS] |= (1 << (job % INT_NUM_BITS));
      pthread_mutex_lock(&mutex_job);
      if (count == num_jobs) {
        pthread_mutex_unlock(&mutex_job);
        conv_subgrid(my_old_grid, my_grid, next_log, job_start, (job_start+job_size));
      } else {
        jobs[rear].start = job_start;
        jobs[rear].end = job_start + job_size;
        jobs[rear].iter = iter;
        rear = (rear+1) % num_jobs;
        count++;
        pthread_cond_signal(&not_empty);
        pthread_mutex_unlock(&mutex_job);
      }
    }

    if(odd % 2) i++;
    odd++;
  }
  return inserted;
}

void deposit_pad_jobs(int iters) {
  /* Deposit jobs bordering pads (top and bottom ones) */
  pthread_mutex_lock(&mutex_job);
  while(count > num_jobs-2) {
    pthread_cond_wait(&not_full, &mutex_job);
  }
  jobs[rear].start = pad_size;
  jobs[rear].end = pad_size + job_size;
  jobs[rear].iter = iters;
  rear = (rear+1) % num_jobs;
  jobs[rear].start = last_job;
  jobs[rear].end = last_job + job_size;  // == rows_per_proc_size + pad_size;
  jobs[rear].iter = iters;
  rear = (rear+1) % num_jobs;
  count += 2;
  pthread_cond_signal(&not_empty);
  pthread_mutex_unlock(&mutex_job);
}

int check_pad_jobs_dependencies(MPI_Request* requests, MPI_Status* status, uint* my_curr_log, int* flags, int rank, int iters) {
  /* Check if 2nd and penultimate jobs have been computed */
  if(!flags[0]) {
    pthread_mutex_lock(&mutex_log);
    flags[0] = BM_VALUE(my_curr_log, 1) && BM_VALUE(my_curr_log, total_jobs-2);
    pthread_mutex_unlock(&mutex_log);
  }

  /* Check if new pads have been receveid */
  if(!flags[1]) {
    int num = (!rank || rank == num_procs-1) ? 2 : 4;
    MPI_Testall(num, requests, &flags[1], status);
  }
  
  return (flags[0] && flags[1]);
}

void wait_pads_completion(uint* iteration_log){
  const uint arr_index = (total_jobs-1) / INT_NUM_BITS;
  const uint shift = (total_jobs-1) % INT_NUM_BITS;
  const uint mask = 1 << shift;

  pthread_mutex_lock(&mutex_log);
  while(!(iteration_log[0] & 1) || !(iteration_log[arr_index] & mask)) {
    pthread_cond_wait(&job_done, &mutex_log);
  }
  pthread_mutex_unlock(&mutex_log);
}

int is_log_high(uint* log){
  const uint last_mask = UINT_MAX >> (INT_NUM_BITS - (total_jobs % INT_NUM_BITS));
  for(int i = 0; i < log_buff_size-1; i++) 
    if(log[i] != UINT_MAX) return 0;
  if(log[log_buff_size-1] != last_mask) return 0;
  return 1;
}

void conv_subgrid(float *sub_grid, float *new_grid, uint *iteration_log, int start_index, int end_index) {
  float result;
  float matrix_dot_sum;                    /* Used for normalization */
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

    /* Convolution */
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

    /* Normalization */
    result = result / sqrt(matrix_dot_sum * kern_dot_sum);
    /* Resolution problem if too many convolution iteration are done. The mean value is returned */
    if(isnan(result)) 
      result = 0;
    new_grid[i] = result;

    /* Setting row and col index for next element */
    if (col != grid_width-1)
      col++;
    else{
      row_start += grid_width;
      col = 0;
    }
  }

  /* Log job as completed */
  log_job(iteration_log, start_index);
}

void log_job(uint* iteration_log, int start) {
  const int job_num = (start-pad_size) / job_size;
  const int shift = job_num % INT_NUM_BITS;
  const int arr_index = job_num / INT_NUM_BITS;
  pthread_mutex_lock(&mutex_log);
  iteration_log[arr_index] |= 1 << shift;
  pthread_cond_signal(&job_done);
  pthread_mutex_unlock(&mutex_log);
}

void init_read(FILE *fp_grid, FILE *fp_kernel){
  char *buffer;
  int kern_row_chars, buffer_size;
  int i, offset;
  MPI_Request request[2];

  /* First token represent matrix dimension */
  if(fscanf(fp_grid, "%hd\n", &grid_width) == EOF || fscanf(fp_kernel, "%hhd\n", &kern_width) == EOF) {
    fprintf(stderr, "Error in file reading: first element should be the row (or column) length of a square matrix\n");
    exit(-1);
  }

  /* Exchange initial information */
  int to_send[] = {grid_width, kern_width};
  MPI_Ibcast(to_send, 2, MPI_INT, 0, MPI_COMM_WORLD, request);

  /* Non-blank chars + Blank chars */
  kern_row_chars = (kern_width * MAX_CHARS + kern_width) * sizeof(char);
  /* Kernel data from file */
  kern_size = kern_width * kern_width;
  buffer_size = kern_row_chars * kern_width;
  buffer = malloc(sizeof(char) * buffer_size);
  kernel = malloc(sizeof(float) * kern_size);
  buffer_size = fread(buffer, sizeof(char), buffer_size, fp_kernel);
  fclose(fp_kernel);

  offset = 0;
  for(i = 0; i < kern_size && offset < buffer_size; i++) {
    kernel[i] = atof(&buffer[offset]);
    while(buffer[offset] >= '+') offset++;                              /* Jump to next blank character */
    while(buffer[offset] != '\0' && buffer[offset] < '+') offset++;     /* Jump all blank characters */
  }

  if(i != kern_size) {
    fprintf(stderr, "Error in file reading: number of kernel elements read is different from the expected amount\n");
    exit(-1);
  }

  /* Exchange kernel */
  MPI_Ibcast(kernel, kern_size, MPI_FLOAT, 0, MPI_COMM_WORLD, &request[1]);
  free(buffer);
}

void read_data(FILE *fp_grid, float* whole_grid, int rows_per_proc) {
  /* Positive non-blank chars + Blank chars */
  const int grid_row_chars = (grid_width * (MAX_CHARS-1) + grid_width) * sizeof(char);
  int buffer_size = grid_row_chars * grid_width;
  int offset = 0, i = 0;
  int iterations = grid_size;
  int start, size;
  char* buffer;
  MPI_Request req[num_procs];

  /* Grid data from file */  
  buffer = malloc(buffer_size * sizeof(char));
  buffer_size = fread(buffer, sizeof(char), buffer_size, fp_grid);
  fclose(fp_grid);

  /* Char grid to float */
  for(; i < iterations && offset < buffer_size; i++) { 
    whole_grid[i] = atof(&buffer[offset]);
    while(buffer[offset] >= '+') offset++;
    while(buffer[offset] != '\0' && buffer[offset] < '+') offset++;
  }

  if(i != iterations) {
    fprintf(stderr, "Error in file reading: number of grid elements read is different from the expected amount\n");
    exit(-1);
  }
  free(buffer);

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
