// conv.c
// Name: Tanay Agarwal, Nirmal Krishnan
// JHED: tagarwa2, nkrishn9

// start commit: 1029afecb3b943ed5facfb75160418f903d23d3d
// end commit:   7689fd1b69a8704b00ca162f11f5f56c413c296a

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <assert.h>
#include <string.h>
#include <math.h>
#include <papi.h>
#include <mpi.h>
#include <immintrin.h>

#define DEFAULT_ITERATIONS 1
#define GRID_FILE_PATH "./io-files/grids/haring.bin"
#define KERNEL_FILE_PATH "./io-files/kernels/gblur.bin"
#define RESULT_FILE_PATH "./io-files/result.txt"
#define MAX_DIGITS 13    /* Standard "%e" format has at most this num of digits (e.g. -9.075626e+20) */
#define VEC_SIZE 4
#define DEBUG 0

void conv_subgrid(float*, float*, int, int);
float normalize(float, float*);
void read_data(int*);
void read_float_matrix(FILE*, float*, int);
void save_txt(float*);
void float_to_echars(float*, char*, int, int);
void handle_PAPI_error(int, char*);

uint8_t num_pads;             /* Number of rows that should be shared with other processes */
uint32_t kern_width;          /* Number of elements in one kernel matrix row */
uint32_t grid_width;          /* Number of elements in one grid matrix row */
uint16_t kern_size;           /* Number of elements in whole kernel matrix */
uint16_t pad_size;            /* Number of elements in the pad section of the grid matrix */
float kern_dot_sum;           /* Used for normalization, its value is equal to: sum(dot(kernel, kernel)) */
float *kernel;                /* Kernel buffer */
float *grid;                  /* Grid buffer */
__m128i last_mask;


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
  MPI_Status status;

  /* PAPI setup */
  if((papi_rc = PAPI_library_init(PAPI_VER_CURRENT)) != PAPI_VER_CURRENT)
    handle_PAPI_error(papi_rc, "Error in library init.");
  if((papi_rc = PAPI_create_eventset(&event_set)) != PAPI_OK)
    handle_PAPI_error(papi_rc, "Error while creating the PAPI eventset.");
  if((papi_rc = PAPI_add_event(event_set, PAPI_L2_TCM)) != PAPI_OK)
    handle_PAPI_error(papi_rc, "Error while adding L2 total cache miss event.");
  if((papi_rc = PAPI_start(event_set)) != PAPI_OK) 
    handle_PAPI_error(papi_rc, "Error in PAPI_start().");
  
  time_start = PAPI_get_real_usec();

  /* Read data from files in "./io-files" dir */
  if(!rank) read_data(&grid_size);
  assert(grid_width % num_procs == 0);

  uint32_t temp_buffer[2];
  if(!rank) {
    temp_buffer[0] = grid_width;
    temp_buffer[1] = kern_width;
  } 

  MPI_Bcast(temp_buffer, 2, MPI_UINT32_T, 0, MPI_COMM_WORLD);
  if (rank) {
    kern_width = temp_buffer[1];
    kern_size = kern_width * kern_width;
    kernel = malloc(sizeof(float) * kern_size);
  }

  MPI_Bcast(kernel, kern_size, MPI_FLOAT, 0, MPI_COMM_WORLD);
  if(rank) {
    grid_width = temp_buffer[0];
    grid_size = grid_width * grid_width;
    num_pads = (kern_width - 1) / 2;
    pad_size = grid_width * num_pads;
    grid = malloc(grid_size*sizeof(float));
  }

  MPI_Bcast(grid, grid_size, MPI_FLOAT, 0, MPI_COMM_WORLD);

  /* Computation of sum(dot(kernel, kernel)) */
  for(int pos = 0; pos < kern_size; pos++){
    kern_dot_sum += kernel[pos] * kernel[pos];
  }

  /* Data splitting */
  int grid_height = (grid_width / num_procs);    /* grid matrix is square, so grid_width and grid_height are the same before data splitting */
  int start = grid_height * rank;
  int end = grid_height - 1 + start;
  int assigned_rows = end + 1 - start;                /* Number of rows assigned to one process */
  int assigned_rows_size = assigned_rows*grid_width;
  uint8_t next = (rank != num_procs-1) ? rank+1 : 0;
  uint8_t prev = (rank != 0) ? rank-1 : num_procs-1;

  /* Rows holded by this process and needed by another one */
  float *upper = &grid[grid_width * start];
  float *lower = &grid[grid_width * (end - num_pads + 1)];
  /* Rows holded by other process and needed for this one */
  float *sub_grid = malloc(sizeof(float) * grid_width * (assigned_rows + (2 * num_pads)));
  float *pad_row_upper = sub_grid;
  float *pad_row_lower = &sub_grid[grid_width * (assigned_rows + num_pads)];

  // Computation of "last_mask"
  uint32_t rem = kern_width % VEC_SIZE;
  uint32_t to_load[VEC_SIZE];
  memset(to_load, 0, VEC_SIZE * sizeof(uint32_t));
  for(int i = 0; i < rem; i++) to_load[i] = UINT32_MAX;        // UINT32_MAX = -1
  last_mask = _mm_loadu_si128((__m128i*) to_load);

  for(int iters = 0; iters < num_iterations; iters++) {
    if(num_procs > 1) {
      if (!rank) {
        // Process with rank 0 doesn't have a "prev" process
        MPI_Recv(pad_row_lower, pad_size, MPI_FLOAT, next, 0, MPI_COMM_WORLD, &status);
        MPI_Send(lower, pad_size, MPI_FLOAT, next, 0, MPI_COMM_WORLD);
      } else if (rank == num_procs-1) {
        // Last process doesn't have a "next" process
        MPI_Send(upper, pad_size, MPI_FLOAT, prev, 0, MPI_COMM_WORLD);
        MPI_Recv(pad_row_upper, pad_size, MPI_FLOAT, prev, 0, MPI_COMM_WORLD, &status);
      } else if(!(rank & 1)) {
        // Even processes (except first)
        MPI_Recv(pad_row_lower, pad_size, MPI_FLOAT, next, 0, MPI_COMM_WORLD, &status);
        MPI_Recv(pad_row_upper, pad_size, MPI_FLOAT, prev, 0, MPI_COMM_WORLD, &status);
        MPI_Send(upper, pad_size, MPI_FLOAT, prev, 0, MPI_COMM_WORLD);
        MPI_Send(lower, pad_size, MPI_FLOAT, next, 0, MPI_COMM_WORLD);
      } else {
        // Odd processes (except last)
        MPI_Send(upper, pad_size, MPI_FLOAT, prev, 0, MPI_COMM_WORLD);
        MPI_Send(lower, pad_size, MPI_FLOAT, next, 0, MPI_COMM_WORLD);
        MPI_Recv(pad_row_lower, pad_size, MPI_FLOAT, next, 0, MPI_COMM_WORLD, &status);
        MPI_Recv(pad_row_upper, pad_size, MPI_FLOAT, prev, 0, MPI_COMM_WORLD, &status);
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
    conv_subgrid(sub_grid, changed_subgrid, grid_width*(num_pads << 1), (grid_width * (assigned_rows-num_pads)));

    /* Pad convolution */
    conv_subgrid(sub_grid, changed_subgrid, pad_size, (grid_width * (num_pads << 1)));
    conv_subgrid(sub_grid, changed_subgrid, (grid_width * (assigned_rows-num_pads)), (grid_width * (assigned_rows+num_pads)));

    if(rank != 0) {
      MPI_Send(changed_subgrid, assigned_rows * grid_width, MPI_INT, 0, 11, MPI_COMM_WORLD);
      MPI_Recv(&grid[0], grid_width * grid_width, MPI_INT, 0, 10, MPI_COMM_WORLD, &status);
    } else {
      for(int i = 0; i < assigned_rows * grid_width; i++) {
        grid[i] = changed_subgrid[i];
      }

      for(int k = 1; k < num_procs; k++) {
        MPI_Recv(&grid[grid_width * (grid_width / num_procs) * k], assigned_rows * grid_width, MPI_INT, k, 11, MPI_COMM_WORLD, &status);
      }

      for(int i = 1; i < num_procs; i++) {
        MPI_Send(grid, grid_width*grid_width, MPI_INT, i, 10, MPI_COMM_WORLD);
      }
    }
  }

  
  /* Stop the count! */ 
  if ((papi_rc = PAPI_stop(event_set, &num_cache_miss)) != PAPI_OK)
    handle_PAPI_error(papi_rc, "Error in PAPI_stop().");
  MPI_Barrier(MPI_COMM_WORLD);
  time_stop = PAPI_get_real_usec();
  if(!rank) printf("[%d] Elapsed time: %lld us | Total L2 cache misses: %lld\n", rank, (time_stop - time_start), num_cache_miss);  
  
  /* Store computed matrix */
  if (DEBUG && !rank) {
    save_txt(grid);
  }

  MPI_Finalize();
  free(grid);
  free(kernel);
  free(sub_grid);
  return 0;
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
    new_grid[i-pad_size] = (!matrix_dot_sum) ? 0 : (result / sqrt(matrix_dot_sum * kern_dot_sum));

    // Setting row and col indexes for next element
    if (col != grid_width-1)
      col++;
    else{
      row_start += grid_width;
      col = 0;
    }
  }
}

void read_data(int *grid_size) {
  FILE *fp_grid, *fp_kernel;         /* Input files containing grid and kernel matrix */

  /* Opening input files */
  if((fp_grid = fopen(GRID_FILE_PATH, "rb")) == NULL) {
    
    perror("fopen grid file error:");
    exit(-1);
  }
  if((fp_kernel = fopen(KERNEL_FILE_PATH, "rb")) == NULL) {
    printf("fopen kernel file error");
    exit(-1);
  }

  // First token represent matrix dimension
  if(fread(&grid_width, sizeof(uint32_t), 1, fp_grid) != 1 || fread(&kern_width, sizeof(uint32_t), 1, fp_kernel) != 1) {
    fprintf(stderr, "Error in file reading: first element should be the row (or column) length of a square matrix\n");
    exit(-1);
  }

  *grid_size = grid_width*grid_width;
  kern_size = kern_width*kern_width;
  num_pads = (kern_width - 1) >> 1;
  pad_size = grid_width * num_pads;

  /* Reading data from files */
  grid = malloc(*grid_size*sizeof(float));
  read_float_matrix(fp_grid, grid, *grid_size);
  fclose(fp_grid);

  kernel = malloc(kern_size*sizeof(float));
  read_float_matrix(fp_kernel, kernel, kern_size);
  fclose(fp_kernel);
}

/* Convert an float array to char array (in "%+e" format) */
int floats_to_echars(float *float_buffer, char* char_buffer, int count, int row_len) {
  int limit = row_len-1;
  int stored = 0;

  for(int fetched = 0; fetched < count; fetched++){
    stored += sprintf(&char_buffer[stored], "%+e", float_buffer[fetched]);
    if (fetched == limit) {
      limit += row_len;
      char_buffer[stored] = '\n';
    } else {
      char_buffer[stored] = ' ';
    }
    stored++;
  }

  return stored;
}

/* Save a float array to file in textual mode */
void save_txt(float* res_grid){
  FILE* fp_result_txt;
  if((fp_result_txt = fopen(RESULT_FILE_PATH, "w")) == NULL) {
    perror("Error while opening txt result debug file\n");
    return;
  }
  uint grid_elems = grid_width*grid_width;

  char* char_buffer = malloc(sizeof(char) * (grid_elems*2) * (MAX_DIGITS + 1));
  const uint count = floats_to_echars(res_grid, char_buffer, grid_elems, grid_width);
  const uint char_written = fwrite(char_buffer, sizeof(char), count, fp_result_txt);
  free(char_buffer);
  if(ferror(fp_result_txt)) {
    perror("Error while writing txt result: ");
    exit(-1);
  }
  if(char_written < count) {
    fprintf(stderr, "Number of chars written: %d | Expected amount: %d\n", char_written, count);
    exit(-1);
  }
  fclose(fp_result_txt);
}

void handle_PAPI_error(int rc, char *msg) {
  char error_str[PAPI_MAX_STR_LEN];
  memset(error_str, 0, PAPI_MAX_STR_LEN);

  printf("%s\nReturn code: %d - PAPI error message:\n", msg, rc);
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
