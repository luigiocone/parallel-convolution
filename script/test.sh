#!/bin/bash
# Exit if CTRL-C is pressed
trap ctrl_c INT
function ctrl_c() {
    echo "Trapped CTRL-C"
    exit 0
}

# Testing different interleavings (hopefully)
SEQ_PATH="/home/luigi/Desktop/seq"
PAR_PATH="/home/luigi/Desktop/parallel-convolution"
COND=''
declare -i i=0
declare -i rc=0

cd $PAR_PATH
make
echo ""
echo "Printing executions errors"

if [ $# = 1 ]; then
    # Testing until error (infinite loop if program is correct)
    while [ "$COND" = '' ] && [ $rc -eq 0 ];
    do
        echo "Execution $i:"
        ./run.sh > outputs.txt
        rc=$?
        COND=$(diff $PAR_PATH/io-files/result.txt $SEQ_PATH/io-files/result.txt | grep '^[1-9]')
        i=$(( i + 1 ))
    done
    echo "return code: ${rc}"
    echo "wrong lines: ${COND}"
else
    # Testing a finite number of executions
    for i in {1..20} 
    do
       echo "Execution $i:"
       ./run.sh > /dev/null   # silencing the output
       diff $PAR_PATH/io-files/result.txt $SEQ_PATH/io-files/result.txt | grep '^[1-9]'
    done
fi
