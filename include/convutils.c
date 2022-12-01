#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <papi.h>
#include <mpi.h>
#include <pthread.h>
#include <unistd.h>
#include "convutils.h"

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

/* Save a float array to file in textual mode */
void save_txt(float* res_grid){
  FILE* fp_result_txt;
  if((fp_result_txt = fopen(DEBUG_TXT_PATH, "w")) == NULL) {
    fprintf(stderr, "Error while opening txt result debug file\n");
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
    fprintf(stderr, "Error while opening bin result debug file\n");
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
