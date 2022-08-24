CC=mpicc
RUN=mpirun
WORKSPACE=/home/ocone/convolution
#${REMOTE_USERNAME} is an environment variable

all: compile

compile: conv.c
	$(CC) conv.c -o conv /usr/local/lib/libpapi.a -std=c11

clean:
	rm conv conv.clog2 conv.slog2
	pkill -u ocone

lrun: compile   # Run locally (from host)
	$(RUN) -np 4 ./conv 2

send:           # Transfer source code (conv.c) to cluster
	scp -P 22001 conv.c Makefile ${REMOTE_USERNAME}:$(WORKSPACE)

receive:        # Receive .clog2 file from cluster
	scp -P 22001 ${REMOTE_USERNAME}:$(WORKSPACE)/conv.clog2 ./

conversion:     # Convert .clog2 to .slog2 file
	java -jar clog2TOslog2.jar conv.clog2

visualize: conversion
	java -jar jumpshot.jar conv.slog2




connect:       # Connect to cluster
	ssh -p 22001 $(REMOTE_USERNAME)

grun: compile   # Run on cluster (from cluster)
#   ssh compute-0-0
	$(RUN) --mca btl self,openib -np 2 -machinefile mf ./conv 2

clog:          # Create a .clog2 file
	mpecc -mpilog -lpthread -o conv conv.c
	$(RUN) --mca btl self,openib -np 2 -machinefile mf ./conv 2


# --mca btl self,openib -> per utilizzare InfiniBand nativa
# -np                   -> numero di processi da mandare in esecuzione
# -machinefile          -> file contenente la lista dei nodi su cui lanciare i processi 
# ./conv arg            -> arg = num. iterazioni