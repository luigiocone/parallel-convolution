CC=mpicc
.PHONY: all

all: conv

conv: conv1.c
	$(CC) conv1.c -o conv /usr/local/lib/libpapi.a -lm -std=gnu11 -mavx -Wall -O3

# PSIMD instructions returns a sligthly different result (result image is the same)
# make && mpirun -np 2 ./conv 50 && diff ./io-files/result.txt ../0.sequential/io-files/result.txt | grep '^[1-9]'
