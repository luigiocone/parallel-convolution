#!/bin/bash

mpicc -o conv conv.c
if [ $? -ne 0 ]; then
  echo "Compile error."
  exit
fi

if [ $# > 0 ] && [ $1 = "-h" ]; then
  mpirun --mca btl self,openib -np 64 -machinefile mf -np 64 ./conv 256 3 2
else
  mpirun -np 4 ./conv 64 3 2
fi

# --mca btl self,openib -> per utilizzare InfiniBand nativa
# -np -> numero di processi da mandare in esecuzione
# -machinefile -> file contenente la lista dei nodi su cui lanciare i processi 
# -h = hpc
# args x y z -> x = dimensione della matrice; y = dimensione del kernel; z = num. iterazioni

# $? -> return value
# $# -> num args
# $1 -> first arg