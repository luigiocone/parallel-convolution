CC=mpicc
.PHONY: all

all: conv

conv: conv.o convutils.o
	$(CC) conv.o convutils.o -o conv /usr/local/lib/libpapi.a -lm -lpthread -mavx -Wall -O3

conv.o: conv.c
	$(CC) conv.c -c -o conv.o -std=gnu11 -mavx -Wall -O3

convutils.o: ./include/convutils.h ./include/convutils.c
	$(CC) ./include/convutils.c -c -o convutils.o -std=gnu11 -Wall -O3
