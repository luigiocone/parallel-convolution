#!/bin/bash

if [ $# = 0 ]; then
    mpirun -np 1 ./conv 1 2
    exit $?
fi

# Cluster workspace
WORKSPACE="/home/ocone/convolution"

case $1 in
  hpc)       # Cluster only
    mpirun --mca btl self,openib -np 4 -machinefile mf --map-by node ./conv 500 16 1
    ;;

  papi)      # Temporary reduce paranoid level to allow cache event counters (PAPI)
    sudo sh -c 'echo 2 >/proc/sys/kernel/perf_event_paranoid'
    ;;

  test)      # Check results differences between sequential and parallel version
    SEQ_PATH="/home/luigi/Desktop/seq/io-files/result.txt"
    PAR_PATH="/home/luigi/Desktop/parallel-convolution/io-files/result.txt"
    echo "wrong lines: "
    diff $PAR_PATH $SEQ_PATH | grep '^[1-9]'
    ;;

  connect)   # Connect to cluster ($CLUSTER_UNISANNIO is an environment variable)
    ssh -p 22001 $CLUSTER_UNISANNIO
    ;;

  send)      # Transfer source code, makefile, and run.sh to cluster
    scp -P 22001 conv.c $CLUSTER_UNISANNIO:$WORKSPACE
    ;;

  receive)   # Transfer .clog2 file from cluster to local machine
    scp -P 22001 $CLUSTER_UNISANNIO:$WORKSPACE/conv.clog2 ./
    ;;
  
  clean)
    rm -f conv*
    pkill -u ocone
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

#valgrind --leak-check=full --show-leak-kinds=all --track-origins=yes --verbose --log-file=valgrind-out.txt mpirun -np 2 ./conv 2
