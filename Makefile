CC=mpicc
.PHONY: all

all: conv

conv: conv.c
	$(CC) conv.c -o conv /usr/local/lib/libpapi.a -std=c11 -Wall
