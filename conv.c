// conv.c
// Name: Tanay Agarwal, Nirmal Krishnan
// JHED: tagarwa2, nkrishn9

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <assert.h>
#include <string.h>
#include <sys/time.h>
#include "papi.h"
#include "mpi.h"

#define DEFAULT_ITERATIONS 1
#define GRID_FILE_PATH "./io-files/grid.txt"
#define KERNEL_FILE_PATH "./io-files/kernel.txt"
#define RESULT_FILE_PATH "./io-files/result.txt"

uint8_t num_rows;             /* Number of rows assigned to one process */
uint8_t num_pads;             /* Number of rows that should be shared with other processes */
uint8_t len_krow;             /* Length of one row of the kernel matrix */
uint16_t len_row;             /* Length of one row of the grid matrix */
uint8_t middle_krow_index;    /* Index of the middle row of the kernel matrix */
int8_t *kernel;               /* Kernel buffer */
int *grid;                    /* Grid buffer */                        

int conv_column(int*, int);
int conv(int*, int);
int *check(int*);
void read_data(int*);
void handle_PAPI_error(int, char*);

int conv_column(int *sub_grid, int i) {
  int counter = 0;
  
  for (int j = 1; j <= num_pads; j++) {
    counter += sub_grid[i + j*len_row] * kernel[middle_krow_index + j*len_krow];
    counter += sub_grid[i - j*len_row] * kernel[middle_krow_index - j*len_krow];
  }
  counter += sub_grid[i] * kernel[middle_krow_index];
  
  return counter;
}

int conv(int *sub_grid, int i) {
  int counter = 0;
  //convolve middle column
  counter += conv_column(sub_grid, i);

  //get first and last element of current row of global grid
  int first = (i / len_row) * len_row;
  int end = first + len_row - 1;

  //convolve left and right columns
  for (int j = 1; j <= num_pads; j++) {
    /* checking if column indexes don't exceed global grid */
    if (i + j - end <= 0) {
      counter += conv_column(sub_grid, i + j);
    }
    if (i - j - first >= 0) {
      counter += conv_column(sub_grid, i - j);
    }
  }
  
  return counter;
}

int *check(int *sub_grid) {
  int val;
  int *new_grid = calloc(len_row * num_rows, sizeof(int));
  for(int i = (num_pads * len_row); i < (len_row * (num_pads + num_rows)); i++) {
    val = conv(sub_grid, i);
    new_grid[i - (num_pads * len_row)] = val;
  }
  return new_grid;
}

int main(int argc, char** argv) {
  MPI_Status status;
  int rank;                          /* Current process identifier */
  int num_procs;                     /* Number of MPI processes in the communicator */
  uint8_t num_iterations;            /* How many times do the convolution operation */
  int len_grid;                      /* Length of whole input grid matrix */
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
  assert(len_row % num_procs == 0);

  /* PAPI setup */
  if(papi_rc = PAPI_library_init(PAPI_VER_CURRENT) != PAPI_VER_CURRENT)
    handle_PAPI_error(papi_rc, "Error in library init.");
  if(papi_rc = PAPI_create_eventset(&event_set) != PAPI_OK)
    handle_PAPI_error(papi_rc, "Error while creating the PAPI eventset.");
	if(papi_rc = PAPI_add_event(event_set, PAPI_L2_TCM) != PAPI_OK)
    handle_PAPI_error(papi_rc, "Errore while adding L2 total cache miss event.");
  
  if(!rank) time_start = PAPI_get_real_usec();
  read_data(&len_grid);

  /* Data splitting */
  int start = (len_row / num_procs) * rank;
  int end = (len_row / num_procs) - 1 + start;
  num_rows = end + 1 - start;
  uint8_t next = (rank != num_procs-1) ? rank+1 : 0;
  uint8_t prev = (rank != 0) ? rank-1 : num_procs-1;

  int upper[len_row*num_pads];   /* Rows holded by this process and needed by another one */
  int lower[len_row*num_pads];
  int *pad_row_upper;            /* Rows holded by other process and needed for this one */
  int *pad_row_lower;
  
  if (papi_rc = PAPI_start(event_set) != PAPI_OK) 
    handle_PAPI_error(papi_rc, "Error in PAPI_start().");

  for(uint8_t iters = 0; iters < num_iterations; iters++) {

    memcpy(lower, &grid[len_row * (end - num_pads + 1)], sizeof(int) * len_row * num_pads);
    pad_row_lower = malloc(sizeof(int) * len_row * num_pads);
    
    memcpy(upper, &grid[len_row * start], sizeof(int) * len_row * num_pads);
    pad_row_upper = malloc(sizeof(int) * len_row * num_pads);

    if(num_procs > 1) {
      if (!rank) {
        /* Process with rank 0 doesn't have a "prev" process */
        MPI_Recv(pad_row_lower, len_row * num_pads, MPI_INT, next, 0, MPI_COMM_WORLD, &status);
        MPI_Send(lower, len_row * num_pads, MPI_INT, next, 0, MPI_COMM_WORLD);
      } else if (rank == num_procs-1) {
        /* Last process doesn't have a "next" process */
        MPI_Send(upper, len_row * num_pads, MPI_INT, prev, 0, MPI_COMM_WORLD);
        MPI_Recv(pad_row_upper, len_row * num_pads, MPI_INT, prev, 0, MPI_COMM_WORLD, &status);
      } else if(!(rank & 1)) {
        /* Even processes (except first) */
        MPI_Recv(pad_row_lower, len_row * num_pads, MPI_INT, next, 0, MPI_COMM_WORLD, &status);
        MPI_Recv(pad_row_upper, len_row * num_pads, MPI_INT, prev, 0, MPI_COMM_WORLD, &status);
        MPI_Send(upper, len_row * num_pads, MPI_INT, prev, 0, MPI_COMM_WORLD);
        MPI_Send(lower, len_row * num_pads, MPI_INT, next, 0, MPI_COMM_WORLD);
      } else {
        /* Odd processes (except last) */
        MPI_Send(upper, len_row * num_pads, MPI_INT, prev, 0, MPI_COMM_WORLD);
        MPI_Send(lower, len_row * num_pads, MPI_INT, next, 0, MPI_COMM_WORLD);
        MPI_Recv(pad_row_lower, len_row * num_pads, MPI_INT, next, 0, MPI_COMM_WORLD, &status);
        MPI_Recv(pad_row_upper, len_row * num_pads, MPI_INT, prev, 0, MPI_COMM_WORLD, &status);
      }  
    } else {
      pad_row_lower = upper;
      pad_row_upper = lower;
    }

    int sub_grid[len_row * (num_rows + (2 * num_pads))];
    if (rank == 0) {
      memset(pad_row_upper, 0, len_row*sizeof(int)*num_pads);
    }
    if (rank == (num_procs - 1)) {
      memset(pad_row_lower, 0, len_row*sizeof(int)*num_pads);
    }
    memcpy(sub_grid, pad_row_upper, sizeof(int) * len_row * num_pads); 
    memcpy(&sub_grid[len_row * num_pads], &grid[len_row * start], sizeof(int) * len_row * num_rows);    
    memcpy(&sub_grid[len_row * (num_rows + num_pads)], pad_row_lower, sizeof(int) * len_row * num_pads);
    int * changed_subgrid = check(sub_grid);

    if(rank != 0) {
      MPI_Send(changed_subgrid, num_rows * len_row, MPI_INT, 0, 11, MPI_COMM_WORLD);
      MPI_Recv(&grid[0], len_row * len_row, MPI_INT, 0, 10, MPI_COMM_WORLD, &status);
    } else {
      for(int i = 0; i < num_rows * len_row; i++) {
        grid[i] = changed_subgrid[i];
      }

      for(int k = 1; k < num_procs; k++) {
        MPI_Recv(&grid[len_row * (len_row / num_procs) * k], num_rows * len_row, MPI_INT, k, 11, MPI_COMM_WORLD, &status);
      }

      for(int i = 1; i < num_procs; i++) {
        MPI_Send(grid, len_row * len_row, MPI_INT, i, 10, MPI_COMM_WORLD);
      }
      
    }
  }

  /* Stop the count! */ 
	if (papi_rc = PAPI_stop(event_set, &num_cache_miss) != PAPI_OK){
    handle_PAPI_error(papi_rc, "Error in PAPI_stop().");
	}

  printf("Rank: %d, total cache misses:%lld\n", rank, num_cache_miss);
  
  /* Store computed matrix */
  if (!rank) {
    FILE *fp_result;
    if((fp_result = fopen(RESULT_FILE_PATH, "w")) == NULL) {
      printf("fopen result file error\n");
      exit(-1);
    }
    
    int row = 0; int col = 0;
    while(row + col < len_grid) {
      fprintf(fp_result, "%d ", grid[row+col]);
      if (col != len_row-1) 
        col++;
      else {
        row += len_row;
        col = 0;
        fprintf(fp_result, "\n");
      }
    }
  }

  if(num_procs >= 2) {
    free(pad_row_upper);
    free(pad_row_lower);
  }

  MPI_Barrier(MPI_COMM_WORLD);
  if(!rank) {
    time_stop = PAPI_get_real_usec();
    printf("(PAPI) Elapsed time: %lld us\n", (time_stop - time_start));
  }

  MPI_Finalize();
}

void read_data(int *len_grid) {
  FILE *fp_grid, *fp_kernel;         /* Input files containing grid and kernel matrix */
  FILE *fp_result;                   /* Output file */
  uint8_t len_kernel;                /* Length of whole input kernel matrix */

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
  fscanf(fp_grid, "%hd", &len_row);
  fscanf(fp_kernel, "%hhd", &len_krow);

  *len_grid = len_row*len_row;
  len_kernel = len_krow*len_krow;
  middle_krow_index = (len_kernel - 1) >> 1;
  num_pads = (len_krow - 1) >> 1;

  /* Reading data from files */
  grid = malloc(*len_grid*sizeof(int));
  for(int i = 0; fscanf(fp_grid, "%d", &grid[i]) != EOF; i++);
  fclose(fp_grid);

  kernel = malloc(len_kernel);
  for(int i = 0; fscanf(fp_kernel, "%hhd", &kernel[i]) != EOF; i++);
  fclose(fp_kernel);
}

void handle_PAPI_error(int rc, char *msg) {
  char error_str[PAPI_MAX_STR_LEN];

  PAPI_perror(error_str); PAPI_strerror(rc);
  printf("%s\nReturn code: %d - %s \n", msg, rc, error_str);
  exit(-1);
}
