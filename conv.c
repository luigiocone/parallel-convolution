// conv.c
// Name: Tanay Agarwal, Nirmal Krishnan
// JHED: tagarwa2, nkrishn9

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <string.h>
#include <sys/time.h>
#include "papi.h"
#include "mpi.h"

#define DEFAULT_ITERATIONS 1
#define GRID_FILE_PATH "./io-files/grid.txt"
#define KERNEL_FILE_PATH "./io-files/kernel.txt"

// documentation: http://mpitutorial.com/tutorials/mpi-send-and-receive/
// http://mpitutorial.com/tutorials/dynamic-receiving-with-mpi-probe-and-mpi-status/

int conv_column(int *, int, int, int, int *, int);
int conv(int *, int, int, int, int *, int);
int * check(int *, int, int, int *, int);

int conv_column(int * sub_grid, int i, int nrows, int DIM, int * kernel, int kernel_dim) {
  int counter = 0;
  int num_pads = (kernel_dim - 1) / 2;
  
  for (int j = 1; j < (num_pads + 1); j++) {
    counter = counter + sub_grid[i + j*DIM] * kernel[(((kernel_dim - 1)*(kernel_dim + 1)) / 2) + j*kernel_dim];
    counter = counter + sub_grid[i - j*DIM] * kernel[(((kernel_dim - 1)*(kernel_dim + 1)) / 2) - j*kernel_dim];
  }
  counter = counter + sub_grid[i] * kernel[(((kernel_dim - 1)*(kernel_dim + 1)) / 2)];
  
  return counter;
}

int conv(int * sub_grid, int i, int nrows, int DIM, int * kernel, int kernel_dim) {
  int counter = 0;
  int num_pads = (kernel_dim - 1) / 2;
  //convolve middle column
  counter = counter + conv_column(sub_grid, i, nrows, DIM, kernel, kernel_dim);

  //convolve left and right columns
  for (int j = 1; j < (num_pads + 1); j++) {
    //get last element of current row
    int end = (((i / DIM) + 1) * DIM) - 1;
    if (i + j - end <= 0) { //if column is valid
      counter = counter + conv_column(sub_grid, i + j, nrows, DIM, kernel, kernel_dim);
    }
    //get first element of current row
    int first = (i / DIM) * DIM;
    if (i - j - first >= 0) {
      counter = counter + conv_column(sub_grid, i - j, nrows, DIM, kernel, kernel_dim);
    }
  }
  
  return counter;
}

int * check(int * sub_grid, int nrows, int DIM, int * kernel, int kernel_dim) {
  int val;
  int num_pads = (kernel_dim - 1) / 2;
  int * new_grid = calloc(DIM * nrows, sizeof(int));
  for(int i = (num_pads * DIM); i < (DIM * (num_pads + nrows)); i++) {
    val = conv(sub_grid, i, nrows, DIM, kernel, kernel_dim);
    new_grid[i - (num_pads * DIM)] = val;
  }
  return new_grid;
}

int main ( int argc, char** argv ) {
  int num_procs;     /* Number of MPI processes in the communicator */
  int rank;          /* Current process identifier */ 
  int iters = 0;
  int num_iterations;
  int DIM;
  int GRID_WIDTH;
  int KERNEL_DIM;
  int KERNEL_SIZE;
  long_long time_start, time_stop;
  struct timeval t1, t2;
  double time; 
  
  if (PAPI_library_init(PAPI_VER_CURRENT) != PAPI_VER_CURRENT) {
    printf("Init PAPI error\n");
    exit(-1);
  }

  time_start = PAPI_get_real_usec();

  /* Reading data from file */
  FILE *fp_grid;
  FILE *fp_kernel;
	
	if((fp_grid = fopen(GRID_FILE_PATH, "r")) == NULL) {
		printf("fopen grid file error");
		exit(-1);
	}
  if((fp_kernel = fopen(KERNEL_FILE_PATH, "r")) == NULL) {
		printf("fopen kernel file error");
		exit(-1);
	}

  /* First token represent matrix dimension */
  fscanf(fp_grid, "%d", &DIM);
  fscanf(fp_kernel, "%d", &KERNEL_DIM);

	GRID_WIDTH = DIM*DIM;
  KERNEL_SIZE = KERNEL_DIM * KERNEL_DIM;
  num_iterations = (argc == 2) ? atoi(argv[1]) : DEFAULT_ITERATIONS;
  
  /* Reading operations */
  int main_grid[GRID_WIDTH];
  for(int i = 0; fscanf(fp_grid, "%d", &main_grid[i]) != EOF; i++);
  fclose(fp_grid);

  int kernel[KERNEL_SIZE];
  for(int i = 0; fscanf(fp_kernel, "%d", &kernel[i]) != EOF; i++);
  fclose(fp_kernel);

  int num_pads = (KERNEL_DIM - 1) / 2;

  // Messaging variables
  MPI_Status status;

  // MPI Setup
  if(MPI_Init(&argc, &argv) != MPI_SUCCESS) {
    printf ( "MPI_Init error\n" );
    exit(-1);
  }

  MPI_Comm_size(MPI_COMM_WORLD, &num_procs); // Set the num_procs
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  assert(DIM % num_procs == 0);

  int upper[DIM * num_pads];
  int lower[DIM * num_pads];
  
  int * pad_row_upper;
  int * pad_row_lower;
  
  int start = (DIM / num_procs) * rank;
  int end = (DIM / num_procs) - 1 + start;
  int nrows = end + 1 - start;
  int next = (rank + 1) % num_procs;
  int prev = (rank != 0) ? rank - 1 : num_procs - 1;
  
  for ( iters = 0; iters < num_iterations; iters++ ) {

    memcpy(lower, &main_grid[DIM * (end - num_pads + 1)], sizeof(int) * DIM * num_pads);
    pad_row_lower = malloc(sizeof(int) * DIM * num_pads);
    
    memcpy(upper, &main_grid[DIM * start], sizeof(int) * DIM * num_pads);
    pad_row_upper = malloc(sizeof(int) * DIM * num_pads);

    if(num_procs > 1) {
      if(rank % 2 == 1) {
        MPI_Recv(pad_row_lower, DIM * num_pads, MPI_INT, next, 1, MPI_COMM_WORLD, &status);
        MPI_Recv(pad_row_upper, DIM * num_pads, MPI_INT, prev, 1, MPI_COMM_WORLD, &status);
      } else {
        MPI_Send(upper, DIM * num_pads, MPI_INT, prev, 1, MPI_COMM_WORLD);
        MPI_Send(lower, DIM * num_pads, MPI_INT, next, 1, MPI_COMM_WORLD);
      }  
      if(rank % 2 == 1) {
        MPI_Send(upper, DIM * num_pads, MPI_INT, prev, 0, MPI_COMM_WORLD);
        MPI_Send(lower, DIM * num_pads, MPI_INT, next, 0, MPI_COMM_WORLD);
      } else {
        MPI_Recv(pad_row_lower, DIM * num_pads, MPI_INT, next, 0, MPI_COMM_WORLD, &status);
        MPI_Recv(pad_row_upper, DIM * num_pads, MPI_INT, prev, 0, MPI_COMM_WORLD, &status);
      }
    } else {
      pad_row_lower = upper;
      pad_row_upper = lower;
    }

    int sub_grid[DIM * (nrows + (2 * num_pads))];
    if (rank == 0) {
      memset(pad_row_upper, 0, DIM*sizeof(int)*num_pads);
    }
    if (rank == (num_procs - 1)) {
      memset(pad_row_lower, 0, DIM*sizeof(int)*num_pads);
    }
    memcpy(sub_grid, pad_row_upper, sizeof(int) * DIM * num_pads); 
    memcpy(&sub_grid[DIM * num_pads], &main_grid[DIM * start], sizeof(int) * DIM * nrows);    
    memcpy(&sub_grid[DIM * (nrows + num_pads)], pad_row_lower, sizeof(int) * DIM * num_pads);
    int * changed_subgrid = check(sub_grid, nrows, DIM, kernel, KERNEL_DIM);

    if(rank != 0) {
      MPI_Send(changed_subgrid, nrows * DIM, MPI_INT, 0, 11, MPI_COMM_WORLD);
      MPI_Recv(&main_grid[0], DIM * DIM, MPI_INT, 0, 10, MPI_COMM_WORLD, &status);
    } else {
      for(int i = 0; i < nrows * DIM; i++) {
        main_grid[i] = changed_subgrid[i];
      }

      for(int k = 1; k < num_procs; k++) {
        MPI_Recv(&main_grid[DIM * (DIM / num_procs) * k], nrows * DIM, MPI_INT, k, 11, MPI_COMM_WORLD, &status);
      }

      for(int i = 1; i < num_procs; i++) {
        MPI_Send(main_grid, DIM * DIM, MPI_INT, i, 10, MPI_COMM_WORLD);
      }
      
    }

    // Output the updated grid state
    /*if (rank == 0) { 
      printf("\nConvolution Output: \n"); 
      for (int j = 0; j < GRID_WIDTH; j++) { 
        if (j % DIM == 0) { 
          printf( "\n" ); 
        } 
        printf("%d ", main_grid[j]); 
      } 
      printf("\n"); 
    }*/
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

  MPI_Finalize(); // finalize so I can exit
}

