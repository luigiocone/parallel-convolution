#ifndef CONVUTILS_H
#define CONVUTILS_H

#define SIM_RECV 2                                // Per-process max number of simultaneous MPI receive requests (from TOP and BOTTOM)
#define SIM_REQS (SIM_RECV * 3)                   // Per-process max number of simultaneous MPI requests
#define DEBUG_EXP_CHARS 13                        // Number of chars for format "%+e" used in float to echars conversion
#define DEBUG_TXT_PATH "./io-files/result.txt"    // Path of the file where the result matrix will be saved in textual mode
#define DEBUG_BIN_PATH "./io-files/result.bin"    // Path of the file where the result matrix will be saved in binary mode

extern uint grid_elems, pad_elems, grid_width; 
extern int num_threads;

enum POSITIONS {                      // Submatrix positions of interests for dependencies handling
    TOP = 0,
    BOTTOM = 1,
    CENTER = 2
};

enum DATA {                           // Data that setup thread must prepare before starting convolution
    GRID = 0,
    KERNEL = 1,
    HANDLERS = 2,
    SEND_INFO = 3
};

struct thread_handler {               // Used by active threads to handle a matrix portion and to synchronize with neighbours
  int tid;                            // Custom thread ID, not the one returned by "pthread_self()"
  uint start, end;                    // Matrix area of ​​interest for this thread
  uint8_t top_rows_done[2];           // Flags for curr and next iteration. "True" if this thread has computed its top rows
  uint8_t bot_rows_done[2];
  struct thread_handler* top;         // To exchange information about pads with neighbour threads
  struct thread_handler* bottom;
  pthread_mutex_t mutex;              // Mutex to access this handler
  pthread_cond_t pad_ready;           // Thread will wait if neighbour's top and bottom rows (pads) aren't ready
};

struct proc_info {                    // Info about data scattering and gathering
  uint8_t has_additional_row;         // If this process must compute an additional row
  uint sstart, ssize;                 // Used for initial input scattering (multiple MPI_Isend)
  uint gstart, gsize;                 // Used for final result gathering (multiple MPI_Irecv)
};

struct setup_args {
  MPI_Request requests[2];            // Used for grid and kernel receive
  uint8_t flags[SEND_INFO+1];         // Describes if a data structure is ready to be used
  pthread_t* threads;                 // Worker threads
  struct proc_info* procs_info;       // Info used for MPI distribution of grid
  struct thread_handler* handlers;    // Thread handlers of main and worker threads
  pthread_mutex_t mutex;              // Mutex to access shared variables beetween main and setup thread
  pthread_cond_t cond;                // Necessary some synchronization points beetween main and setup thread
};

struct node_data {                    // Data used to interact with distributed memory neighbours
  MPI_Request requests[SIM_REQS];     // There are at most two "Isend" and one "Irecv" not completed at the same time per pad, hence six per process
  MPI_Status statuses[SIM_REQS];
  int recv_completed[SIM_RECV];       // Log of completed MPI receive (no need to check send)
  uint send_position[SIM_RECV];       // Grid position of data that will be sent by MPI
  uint recv_position[SIM_RECV];       // Where the payload received (through MPI) should be stored
  int neighbour[SIM_RECV];            // MPI rank of the neighbour process (TOP and BOTTOM)
  uint8_t send_offset[SIM_RECV];      // Used to reference the send request
};

struct local_data {                   // Data used to interact with shared memory neighbours
  struct thread_handler* self;        // Handler of communication promoter
  int completed[CENTER+1];            // Matrix completed positions. Indexed through enumeration (TOP, BOTTOM, ...)
  struct thread_handler* neigh[2];    // Handlers of neighbour thread (TOP and BOTTOM)
  uint8_t* rows_to_wait[2];           // Pointers to the flag to wait (TOP and BOTTOM)
  uint8_t* rows_to_assert[2];         // Pointers to the flag to signal (TOP and BOTTOM)
};

struct load_balancer {                // Used by worker threads to do some additional work if they end earlier. Not an active thread
  struct thread_handler* handler;     // Load balancer have neighbour dependecies too
  pthread_mutex_t mutex;              // Used to access at shared variable of the load balancing
  pthread_cond_t iter_completed;      // In some cases a thread could access to load_balancer while previous lb_iter was not completed
  uint iter;                          // Used in load balancing to track current iteration
  uint curr_start;                    // To track how many load balancer rows have been reserved (but not yet computed)
  uint rows_completed;                // To track how many load balancer rows have been computed
  uint top_pad, bot_pad;              // To track how many load balancer pad rows have been computed
  uint nrows, size;                   // Number of rows in load balancer submatrix (i.e. handled dynamically for load balancing)
};

int stick_this_thread_to_core(int);             // Thread affinity
void handle_PAPI_error(int, char*);             // Print a consistent error message if it occurred
void read_float_matrix(FILE*, float*, int);     // Read (in binary mode) a matrix of floating point values from file 
int floats_to_echars(float*, char*, int, int);  // Convert float matrix to chars using format "%+e"
void save_txt(float*);                          // Save the computed matrix in textual mode (for debug)
void save_bin(float*);                          // Save the computed matrix in binary mode (for debug)

#endif
