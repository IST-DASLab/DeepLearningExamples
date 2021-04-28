#!/bin/bash
NODES=2
BS=$(( 64 / $NODES ))
LOG_FILE="logs/adapt"
horovodrun -np $NODES python mnist.py --epochs 2 --log-interval 100 --batch-size $BS --quantization-bits 4  | tee $LOG_FILE