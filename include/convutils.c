#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <papi.h>
#include <mpi.h>
#include <pthread.h>
#include <unistd.h>
#include "convutils.h"

#define DEBUG_TXT_PATH "./io-files/result.txt"

extern uint grid_elems, pad_elems, grid_width, num_threads;

/* Set thread affinity. If there are more threads than cores, no affinity will be set */
int stick_this_thread_to_core(int core_id) {
  const long num_cores = sysconf(_SC_NPROCESSORS_ONLN);
  if(num_threads > num_cores) return 0;
  if(core_id < 0) return 1;
  if((num_threads * 2) <= num_cores) core_id++;   // Trying to avoid hyperthreading in a bad way

  cpu_set_t cpuset;
  CPU_ZERO(&cpuset);
  CPU_SET(core_id, &cpuset);

  return pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);
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

/* Save a float array to a text file */
void save_txt(float* res_grid){
  FILE* fp_result_txt;
  if((fp_result_txt = fopen(DEBUG_TXT_PATH, "w")) == NULL) {
    fprintf(stderr, "Error while opening result debug file\n");
    exit(-1);
  }
  char* temp_buffer = malloc(sizeof(char) * (grid_elems*2) * (13 + 1));
  int count = floats_to_echars(&res_grid[pad_elems], temp_buffer, grid_elems, grid_width);
  fwrite(temp_buffer, count, sizeof(char), fp_result_txt);
  free(temp_buffer);
  fclose(fp_result_txt);
}
