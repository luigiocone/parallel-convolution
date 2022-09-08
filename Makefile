CC=mpicc
.PHONY: all

all: conv

conv: conv.c
	$(CC) conv.c -o conv /usr/local/lib/libpapi.a -lm -lpthread -std=gnu11 -Wall
