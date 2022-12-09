#ifndef CONVUTILS_H
#define CONVUTILS_H

#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <papi.h>
#include <mpi.h>
#include <pthread.h>
#include <unistd.h>

#define SIM_RECV 2                                // Per-process max number of simultaneous MPI receive requests (from TOP and BOTTOM)
#define SIM_REQS (SIM_RECV * 3)                   // Per-process max number of simultaneous MPI requests
#define DEBUG_EXP_CHARS 13                        // Format "%+e" has this num of chars (e.g. -9.075626e+20)
#define DEBUG_TXT_PATH "./io-files/result.txt"    // Path of the file where the result matrix will be saved in textual mode
#define DEBUG_BIN_PATH "./io-files/result.bin"    // Path of the file where the result matrix will be saved in binary mode

extern uint grid_elems, pad_elems, grid_width; 
extern int num_threads;

enum POSITION {                          // Submatrix positions of interests for dependencies handling
    TOP = 0,
    BOTTOM = 1,
    CENTER = 2
};

enum DATA {                              // Data that must be prepared before starting convolution
    GRID = 0,
    KERNEL = 1,
    SEND_INFO = 2
};

struct thread_handler {                  // Used by active threads to handle a matrix portion and to synchronize with neighbours
  int tid;                               // Custom thread ID, not the one returned by "pthread_self()"
  uint start, end;                       // Matrix area of ​​interest for this thread
  uint8_t rows_done[2][2];               // Flags for TOP/BOTTOM bordering rows and for curr/next iteration
  struct thread_handler* neighbour[2];   // To exchange information about pads with neighbour threads
  pthread_mutex_t mutex;                 // Mutex to access this handler
  pthread_cond_t pad_ready;              // Thread will wait if neighbour's top and bottom rows (pads) aren't ready
};

struct proc_info {                       // Info about data scattering and gathering
  uint8_t has_additional_row;            // If this process must compute an additional row
  int sstart, ssize;                     // Used for initial input scattering (multiple MPI_Isend)
  int gstart, gsize;                     // Used for final result gathering (multiple MPI_Irecv)
};

struct io_thread_args {
  FILE* fp_grid;                         // File containing input grid
  MPI_Request* requests;                 // Used for grid and kernel receive
  uint8_t flags[SEND_INFO+1];            // Describes if a data structure is ready to be used
  struct proc_info* procs_info;          // Info used for MPI distribution of grid
  pthread_mutex_t mutex;                 // Mutex to access shared variables beetween main and setup thread
  pthread_cond_t cond;                   // Necessary some synchronization points beetween main and setup thread
};

struct mpi_data {                        // Data used to interact with distributed memory neighbours
  uint send_position;                    // Grid position of data that will be sent by MPI
  uint recv_position;                    // Where the payload received (through MPI) should be stored
  int neighbour;                         // MPI rank of the neighbour process (TOP and BOTTOM)
};

struct worker_data {                     // Data used to interact with shared memory neighbours
  struct thread_handler* self;           // Handler of communication promoter
  int completed[CENTER+1];               // Matrix completed positions. Indexed through enumeration (TOP, BOTTOM, ...)
  struct mpi_data* mpi;                  // Is != NULL if this worker has MPI dependencies. Contains data about MPI neighbours
};

struct load_balancer {                   // Used by worker threads to do some additional work if they end earlier. Not an active thread
  struct thread_handler* handler;        // Load balancer have neighbour dependecies too
  pthread_mutex_t mutex;                 // Used to access at shared variable of the load balancing
  pthread_cond_t iter_completed;         // In some cases a thread could access to load_balancer while previous lb_iter was not completed
  uint iter;                             // Used in load balancing to track current iteration
  uint curr_start;                       // To track how many load balancer rows have been reserved (but not yet computed)
  uint rows_completed;                   // To track how many load balancer rows have been computed
  uint nrwos_pad_completed[2];           // To track how many load balancer pad rows have been computed
  uint nrows, size;                      // Number of rows in load balancer submatrix (i.e. handled dynamically for load balancing)
};

int stick_this_thread_to_core(int);             // Thread affinity
void handle_PAPI_error(int, char*);             // Print a consistent error message if a PAPI error occurred
void read_float_matrix(FILE*, float*, int);     // Read (in binary mode) a matrix of floating point values from file 
int floats_to_echars(float*, char*, int, int);  // Convert float matrix to chars using format "%+e" (for debug)
void save_txt(float*);                          // Save the computed matrix in textual mode (for debug)
void save_bin(float*);                          // Save the computed matrix in binary mode (for debug)

#endif
