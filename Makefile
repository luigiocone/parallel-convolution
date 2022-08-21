CC=mpicc
RUN=mpirun

all: run

compile: conv.c
	$(CC) conv.c -o conv

run: compile
ifeq ($(HPC),y) # Run on cluster (HPC=y) or locally
	$(RUN) --mca btl self,openib -np 64 -machinefile mf -np 64 ./conv 256 3 2
else
	$(RUN) -np 4 ./conv 64 3 2
endif

clean:
	rm conv
	