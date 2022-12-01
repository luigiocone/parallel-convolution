// conv.c
// Name: Tanay Agarwal, Nirmal Krishnan
// JHED: tagarwa2, nkrishn9

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <papi.h>
#include <immintrin.h>

#define DEFAULT_ITERATIONS 1
#define GRID_FILE_PATH "../../../io-files/grids/haring.bin"
#define KERNEL_FILE_PATH "../../../io-files/kernels/gblur.bin"
#define VEC_SIZE 4
#define MEAN_VALUE 0
#define EXP_CHARS 13                              // Format "%+e" has this num of chars (e.g. -9.075626e+20)
#define DEBUG 1                                   // True to save result in textual and binary mode
#define DEBUG_EXP_CHARS 13                        // Format "%+e" has this num of chars (e.g. -9.075626e+20)
#define DEBUG_TXT_PATH "./io-files/result.txt"    // Path of the file where the result matrix will be saved in textual mode
#define DEBUG_BIN_PATH "./io-files/result.bin"    // Path of the file where the result matrix will be saved in binary mode

void conv_subgrid(float*, float*, int, int);      // Convolution operation
void handle_PAPI_error(int, char*);               // Print a consistent error message if it occurred
void read_float_matrix(FILE*, float*, int);       // Read (in binary mode) a matrix of floating point values from file 
int floats_to_echars(float*, char*, int, int);    // Convert float matrix to chars using format "%+e"
void save_txt(float*);                            // Save the computed matrix in textual mode (for debug)
void save_bin(float*);                            // Save the computed matrix in binary mode (for debug)

float *kernel;                        // Kernel used for convolution
float *new_grid;                      // Input/Result grid, swapped at every iteration
float *old_grid;                      // Input/Result grid, swapped at every iteration
float kern_dot_sum;                   // Used for normalization, its value is equal to: sum(dot(kernel, kernel))
uint32_t kern_width;                  // Number of elements in one kernel matrix row
uint32_t grid_width;                  // Number of elements in one grid matrix row
uint pad_nrows;                       // Number of rows that should be shared with other processes
uint pad_elems;                       // Number of elements in the pad section of the grid matrix
uint kern_elems;                      // Number of elements in whole kernel matrix
uint grid_elems;                      // Number of elements in whole grid matrix
int num_iterations;                   // Number of convolution operations
__m128i last_mask;                    // Mask used by SIMD instructions for loading

int main(int argc, char** argv) {
  long_long time_start, time_stop;    // To measure execution time
  long_long num_cache_miss;           // To measure number of cache misses 
  int event_set = PAPI_NULL;          // Group of hardware events for PAPI library 
  int rc;                             // Return code used in error handling 
  FILE *fp_grid = NULL;               // I/O files for grid and kernel matrices
  FILE *fp_kernel = NULL;

  /* How many times do the convolution operation */
  num_iterations = (argc > 1) ? atoi(argv[1]) : DEFAULT_ITERATIONS;
  if(num_iterations < DEFAULT_ITERATIONS) {
    fprintf(stderr, "Invalid number of convolution iterations (first argument), value inserted: %d\n", num_iterations);
    exit(-1);
  }

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
 
  /* Opening input files in dir "./io-files" */
  if((fp_grid = fopen(GRID_FILE_PATH, "rb")) == NULL) {
    perror("Error while opening grid file:");
    exit(-1);
  }
  if((fp_kernel = fopen(KERNEL_FILE_PATH, "rb")) == NULL) {
    perror("Error while opening kernel file:");
    exit(-1);
  }

  /* First token represent matrix dimension */
  if(fread(&grid_width, sizeof(uint), 1, fp_grid) != 1 || fread(&kern_width, sizeof(uint32_t), 1, fp_kernel) != 1) {
    fprintf(stderr, "Error in file reading: first element should be the row (or column) length of a square matrix\n");
    exit(-1);
  }
  grid_elems = grid_width * grid_width;
  kern_elems = kern_width * kern_width;
  pad_nrows = (kern_width - 1) / 2;
  pad_elems = grid_width * pad_nrows;

  kernel = malloc(sizeof(float) * kern_elems);
  read_float_matrix(fp_kernel, kernel, kern_elems);

  /* Computation of sum(dot(kernel, kernel)) */
  for(int pos = 0; pos < kern_elems; pos++)
    kern_dot_sum += kernel[pos] * kernel[pos];

  /* Computation of "last_mask" */
  uint rem = kern_width % VEC_SIZE;
  uint32_t to_load[VEC_SIZE];
  memset(to_load, 0, VEC_SIZE * sizeof(uint32_t));
  for(int i = 0; i < rem; i++) to_load[i] = UINT32_MAX;       /* UINT32_MAX = -1 */
  last_mask = _mm_loadu_si128((__m128i*) to_load);

  /* Data splitting */
  const uint start = pad_elems;                               /* Index of the first row for current process */
  const uint end = grid_elems + pad_elems;                    /* Index of the final row for current process */
  const uint proc_nrows = grid_width;                      /* Number of rows assigned to a process */

  /* Read grid data */
  new_grid = malloc((proc_nrows + pad_nrows*2) * grid_width * sizeof(float));
  old_grid = malloc((proc_nrows + pad_nrows*2) * grid_width * sizeof(float));
  memset(new_grid, 0, pad_elems * sizeof(float));
  memset(old_grid, 0, pad_elems * sizeof(float));
  memset(&new_grid[(proc_nrows+pad_nrows) * grid_width], 0, pad_elems * sizeof(float));
  memset(&old_grid[(proc_nrows+pad_nrows) * grid_width], 0, pad_elems * sizeof(float));
  read_float_matrix(fp_grid, &old_grid[pad_elems], grid_elems);
 
  /* Second (or higher) iterations */
  float *tmp_grid;
  for(uint8_t iters = 0; iters < num_iterations; iters++) {
    conv_subgrid(old_grid, new_grid, start, end);
    /* Swap grid pointers */
    tmp_grid = old_grid;
    old_grid = new_grid;
    new_grid = tmp_grid;
  }

  /* Stop the count! */ 
  if ((rc = PAPI_stop(event_set, &num_cache_miss)) != PAPI_OK)
    handle_PAPI_error(rc, "Error in PAPI_stop().");

  time_stop = PAPI_get_real_usec();
  printf("Rank[0] | Elapsed time: %lld us | Total L2 cache misses: %llu\n", (time_stop - time_start), num_cache_miss);  
  
  /* Debug purposes */
  if (DEBUG) {
    save_bin(old_grid);
    save_txt(old_grid);
  }
  
  free(new_grid);
  free(old_grid);
  free(kernel);
  exit(0);
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

/* Print the appropriate message in case of PAPI error */
void handle_PAPI_error(int rc, char *msg) {
  char error_str[PAPI_MAX_STR_LEN];
  memset(error_str, 0, sizeof(char) * PAPI_MAX_STR_LEN);

  fprintf(stderr, "%s\nReturn code: %d - PAPI error message:\n", msg, rc);
  PAPI_perror(error_str);
  PAPI_strerror(rc);
  exit(-1);
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
  if((fp_result_txt = fopen(DEBUG_TXT_PATH, "w")) == NULL) {
    perror("Error while opening txt result debug file:");
    return;
  }

  char* char_buffer = malloc(sizeof(char) * (grid_elems*2) * (DEBUG_EXP_CHARS + 1));
  const uint count = floats_to_echars(&res_grid[pad_elems], char_buffer, grid_elems, grid_width);
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

/* Save a float array to file in binary mode */
void save_bin(float* res_grid){
  FILE* fp_result_bin;
  if ((fp_result_bin = fopen(DEBUG_BIN_PATH, "wb")) == NULL) {
    perror("Error while opening bin result debug file\n");
    return;
  }
  
  const uint float_written = fwrite(&res_grid[pad_elems], sizeof(float), grid_elems, fp_result_bin);
  if(ferror(fp_result_bin)) {
    perror("Error while writing bin result: ");
    exit(-1);
  }
  if(float_written < grid_elems) {
    fprintf(stderr, "Number of float elements written: %d | Expected amount: %d\n", float_written, grid_elems);
    exit(-1);
  }
  fclose(fp_result_bin);
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
