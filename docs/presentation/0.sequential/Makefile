CC=mpicc
.PHONY: all

all: seq

seq: seq.c
	$(CC) seq.c -o seq /usr/local/lib/libpapi.a -lm -std=gnu11 -mavx -Wall -O3
