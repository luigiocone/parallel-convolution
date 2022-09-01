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

#define DEFAULT_ITERATIONS 1
#define GRID_FILE_PATH "./io-files/grid.txt"
#define KERNEL_FILE_PATH "./io-files/kernel.txt"
#define RESULT_FILE_PATH "./io-files/result.txt"
#define MAX_DIGITS 13    /* Standard "%e" format has at most this num of digits (e.g. -9.075626e+20) */

void conv_subgrid(float*, float*, int, int);
float normalize(float, float*);
void read_data(int*);
void store_data(FILE*, int, int);
void handle_PAPI_error(int, char*);

uint8_t num_pads;             /* Number of rows that should be shared with other processes */
uint8_t kern_width;           /* Number of elements in one kernel matrix row */
uint16_t grid_width;          /* Number of elements in one grid matrix row */
uint16_t kern_size;           /* Number of elements in whole kernel matrix */
uint16_t pad_size;            /* Number of elements in the pad section of the grid matrix */
float kern_dot_sum;           /* Used for normalization, its value is equal to: sum(dot(kernel, kernel)) */
float *kernel;                /* Kernel buffer */
float *grid;                  /* Grid buffer */


int main(int argc, char** argv) {
  int rank;                          /* Current process identifier */
  int num_procs;                     /* Number of MPI processes in the communicator */
  uint8_t num_iterations;            /* How many times do the convolution operation */
  int grid_size;                     /* Number of elements of whole grid matrix */
  long_long time_start, time_stop;   /* To measure execution time */
  long_long num_cache_miss;          /* To measure number of cache miss */
  int event_set = PAPI_NULL;         /* Group of hardware events for PAPI library */
  int papi_rc;                       /* PAPI return code, used in error handling */

  num_iterations = (argc == 2) ? atoi(argv[1]) : DEFAULT_ITERATIONS;
  
  /* MPI Setup */
  if(MPI_Init(&argc, &argv) != MPI_SUCCESS) {
    printf("MPI_Init error\n");
    exit(-1);
  }
  MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  int size = (num_procs < 4) ? 4 : 4 + num_procs-1;
  MPI_Status status[size];
  MPI_Request req[size];

  /* PAPI setup */
  if((papi_rc = PAPI_library_init(PAPI_VER_CURRENT)) != PAPI_VER_CURRENT)
    handle_PAPI_error(papi_rc, "Error in library init.");
  if((papi_rc = PAPI_create_eventset(&event_set)) != PAPI_OK)
    handle_PAPI_error(papi_rc, "Error while creating the PAPI eventset.");
  if((papi_rc = PAPI_add_event(event_set, PAPI_L2_TCM)) != PAPI_OK)
    handle_PAPI_error(papi_rc, "Error while adding L2 total cache miss event.");
  
  time_start = PAPI_get_real_usec();

  /* Read data from files in "./io-files" dir */
  read_data(&grid_size);
  assert(grid_width % num_procs == 0);

  /* Computation of sum(dot(kernel, kernel)) */
  for(int pos = 0; pos < kern_size; pos++){
    kern_dot_sum += kernel[pos] * kernel[pos];
  }

  /* Data splitting */
  int grid_height = (grid_width / num_procs);    /* grid matrix is square, so grid_width and grid_height are the same before data splitting */
  int start = grid_height * rank;
  int end = grid_height - 1 + start;
  int num_rows = end + 1 - start;                /* Number of rows assigned to one process */
  int assigned_rows_size = num_rows*grid_width;
  uint8_t next = (rank != num_procs-1) ? rank+1 : 0;
  uint8_t prev = (rank != 0) ? rank-1 : num_procs-1;

  /* Rows holded by this process and needed by another one */
  float *upper = &grid[grid_width * start];
  float *lower = &grid[grid_width * (end - num_pads + 1)];
  /* Rows holded by other process and needed for this one */
  float *pad_row_upper;
  float *pad_row_lower;
  
  if ((papi_rc = PAPI_start(event_set)) != PAPI_OK) 
    handle_PAPI_error(papi_rc, "Error in PAPI_start().");

  for(uint8_t iters = 0; iters < num_iterations; iters++) {
    float sub_grid[grid_width * (num_rows + (2 * num_pads))];
    pad_row_upper = sub_grid;
    pad_row_lower = &sub_grid[grid_width * (num_rows + num_pads)];

    if(num_procs > 1) {
      if (!rank) {
        /* Process with rank 0 doesn't have a "prev" process */
        MPI_Isend(lower, pad_size, MPI_FLOAT, next, 0, MPI_COMM_WORLD, &req[1]);
        MPI_Irecv(pad_row_lower, pad_size, MPI_FLOAT, next, 1, MPI_COMM_WORLD, req);
      } else if (rank == num_procs-1) {
        /* Last process doesn't have a "next" process */
        MPI_Isend(upper, pad_size, MPI_FLOAT, prev, 1, MPI_COMM_WORLD, &req[1]);
        MPI_Irecv(pad_row_upper, pad_size, MPI_FLOAT, prev, 0, MPI_COMM_WORLD, req);
      } else {
        /* Every other process */
        MPI_Isend(upper, pad_size, MPI_FLOAT, prev, 1, MPI_COMM_WORLD, &req[2]);
        MPI_Irecv(pad_row_upper, pad_size, MPI_FLOAT, prev, 0, MPI_COMM_WORLD, req);
        MPI_Isend(lower, pad_size, MPI_FLOAT, next, 0, MPI_COMM_WORLD, &req[3]);
        MPI_Irecv(pad_row_lower, pad_size, MPI_FLOAT, next, 1, MPI_COMM_WORLD, &req[1]);
      }  
    } 

    if (rank == 0) {
      memset(pad_row_upper, 0, pad_size*sizeof(float));
    }
    if (rank == (num_procs - 1)) {
      memset(pad_row_lower, 0, pad_size*sizeof(float));
    }

    memcpy(&sub_grid[pad_size], &grid[grid_width * start], sizeof(float) * assigned_rows_size);

    /* Start convolution only of central grid elements */
    float *changed_subgrid = malloc(assigned_rows_size * sizeof(float));
    conv_subgrid(sub_grid, changed_subgrid, grid_width*(num_pads << 1), (grid_width * (num_rows-num_pads)));

    /* Pad convolution */
    if(num_procs > 1) {
      if(!rank || rank == num_procs-1)
        MPI_Wait(req, status);
      else 
        MPI_Waitall(2, req, status);
    }
    conv_subgrid(sub_grid, changed_subgrid, pad_size, (grid_width * (num_pads << 1)));
    conv_subgrid(sub_grid, changed_subgrid, (grid_width * (num_rows-num_pads)), (grid_width * (num_rows+num_pads)));

    if(rank != 0) {
      MPI_Isend(changed_subgrid, assigned_rows_size, MPI_FLOAT, 0, 11, MPI_COMM_WORLD, req);
    } else {
      for(int k = 1; k < num_procs; k++) {
        MPI_Irecv(&grid[grid_width * (grid_width / num_procs) * k], assigned_rows_size, MPI_FLOAT, k, 11, MPI_COMM_WORLD, &req[k-1]);
      }
    }
    memcpy(&grid[grid_width*start], changed_subgrid, assigned_rows_size * sizeof(float));
  }

  /* Stop the count! */ 
  if ((papi_rc = PAPI_stop(event_set, &num_cache_miss)) != PAPI_OK)
    handle_PAPI_error(papi_rc, "Error in PAPI_stop().");
  printf("Rank: %d, total cache misses:%lld\n", rank, num_cache_miss);
  
  /* Store computed matrix */
  if (!rank) {
    FILE *fp_result;
    if((fp_result = fopen(RESULT_FILE_PATH, "w")) == NULL) {
      printf("fopen result file error\n");
      exit(-1);
    }
    store_data(fp_result, start, assigned_rows_size);
    MPI_Waitall(num_procs-1, req, status);
    store_data(fp_result, (end+1)*grid_width, grid_size);
    fclose(fp_result);
  }

  MPI_Barrier(MPI_COMM_WORLD);
  if(!rank) {
    time_stop = PAPI_get_real_usec();
    printf("(PAPI) Elapsed time: %lld us\n", (time_stop - time_start));
  }

  MPI_Finalize();
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
    result = 0;
    for (int iter=0, offset=0; iter < iterations; iter++) {
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

    new_grid[i-pad_size] = normalize(result, matrix);

    /* Setting row and col index for next element */
    if (col != grid_width-1)
      col++;
    else{
      row_start += grid_width;
      col = 0;
    }
  }
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

void read_data(int *grid_size) {
  FILE *fp_grid, *fp_kernel;         /* Input files containing grid and kernel matrix */

  /* Opening input files */
  if((fp_grid = fopen(GRID_FILE_PATH, "r")) == NULL) {
    printf("fopen grid file error");
    exit(-1);
  }
  if((fp_kernel = fopen(KERNEL_FILE_PATH, "r")) == NULL) {
    printf("fopen kernel file error");
    exit(-1);
  }

  /* First token represent matrix dimension */
  if(fscanf(fp_grid, "%hd", &grid_width) == EOF || fscanf(fp_kernel, "%hhd", &kern_width) == EOF) {
    printf("Error in file reading\n");
    exit(-1);
  }

  *grid_size = grid_width*grid_width;
  kern_size = kern_width*kern_width;
  num_pads = (kern_width - 1) >> 1;
  pad_size = grid_width * num_pads;

  /* Reading data from files */
  grid = malloc(*grid_size*sizeof(float));
  for(int i = 0; fscanf(fp_grid, "%e", &grid[i]) != EOF; i++);
  fclose(fp_grid);

  kernel = malloc(kern_size*sizeof(float));
  for(int i = 0; fscanf(fp_kernel, "%f", &kernel[i]) != EOF; i++);
  fclose(fp_kernel);
}

void store_data(FILE *fp_result, int start_position, int end_position){
  /* count*2 for blank chars, MAX_DIGITS+1 in case of negative numbers */
  char buffer[(end_position-start_position+1)*2*(MAX_DIGITS+1)];
  int col = start_position % grid_width;
  int row = start_position - col;
  int offset = 0;

  /* Buffer filling */
  while(row + col < end_position) {
    offset += sprintf(&buffer[offset], "%e ", grid[row+col]);
    if (col != grid_width-1) 
      col++;
    else {
      row += grid_width;
      col = 0;
      offset += sprintf(&buffer[offset], "\n");
    }
  }
  fwrite(buffer, sizeof(char), offset, fp_result);
}

void handle_PAPI_error(int rc, char *msg) {
  char error_str[PAPI_MAX_STR_LEN];
  memset(error_str, 0, PAPI_MAX_STR_LEN);

  printf("%s\nReturn code: %d - PAPI error message:\n", msg, rc);
  PAPI_perror(error_str); PAPI_strerror(rc);
  exit(-1);
}
