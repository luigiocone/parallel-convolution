#!/bin/bash
# Testing different interleavings (hopefully)
SEQ_PATH="/home/luigi/Desktop/seq/io-files/result.txt"
PAR_PATH="/home/luigi/Desktop/Parallel-Convolution-MPI/io-files/result.txt"
COND=''
declare -i i=0

make
echo ""
echo "Printing executions errors"

if [ $# = 1 ]; then
    # Testing until error (infinite loop if program is correct) 
    while [ "$COND" = '' ] 
    do
        echo "Execution $i:"
        ./run.sh > outputs.txt
        COND=$(diff $PAR_PATH $SEQ_PATH | grep '^[1-9]')
        i=$(( i + 1 ))
    done
    echo "${COND}"
else
    # Testing a finite number of executions
    for i in {1..20} 
    do
       echo "Execution $i:"
       ./run.sh > /dev/null   # silencing the output
       diff $PAR_PATH $SEQ_PATH | grep '^[1-9]'
    done
fi
