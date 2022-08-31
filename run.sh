#!/bin/bash

if [ $# = 0 ]; then
    mpirun -np 4 ./conv 2    # Suggested only one iteration (./conv 1), two are used just for testing
    exit 0
fi

WORKSPACE="/home/ocone/convolution"
# $CLUSTER_UNISANNIO is an environment variable

case $1 in
  hpc)       # Cluster only
    mpirun --mca btl self,openib -np 64 -machinefile mf ./conv 2
    ;;

  connect)   # Connect to cluster
    ssh -p 22001 $CLUSTER_UNISANNIO
    ;;

  clean)
    rm -f conv conv.clog2 conv.slog2
    pkill -u ocone
    ;;

  send)      # Transfer source code, makefile, and run.sh to cluster
    scp -P 22001 conv.c Makefile run.sh $CLUSTER_UNISANNIO:$WORKSPACE
    ;;

  receive)   # Transfer .clog2 file from cluster to local machine
    scp -P 22001 $CLUSTER_UNISANNIO:$WORKSPACE/conv.clog2 ./
    ;;
  
  createlog) # Create a .clog2 file. Java commands must be executed on local machine 
    mpecc -mpilog -lpthread -o conv conv.c
    mpirun --mca btl self,openib -np 2 -machinefile mf ./conv 2
    ;;

  convert)   # Convert .clog2 to .slog2 file
    java -jar clog2TOslog2.jar conv.clog2
    ;;

  showlog)
    java -jar jumpshot.jar conv.slog2
    ;;

  paranoid)   # Temporary reduce paranoid level to allow cache event counters
    sudo sh -c 'echo -1 >/proc/sys/kernel/perf_event_paranoid'
    ;;

  *)
    echo "unknown command"
    ;;
esac


# --mca btl self,openib -> per utilizzare InfiniBand nativa
# -np                   -> numero di processi da mandare in esecuzione
# -machinefile          -> file contenente la lista dei nodi su cui lanciare i processi 
# ./conv arg            -> arg = num. iterazioni

# $? -> return value
# $# -> num args
# $1 -> first arg
