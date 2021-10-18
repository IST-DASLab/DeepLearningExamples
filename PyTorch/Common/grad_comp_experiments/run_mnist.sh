#!/bin/bash
NODES=2
BS=$(( 64 / $NODES ))
for noise_level in `seq -f "%f" 0.1 0.1 1.0`; do
  LOG_FILE="logs/noise-${noise_level}"
  horovodrun -np $NODES python mnist.py --epochs 20 --log-interval 100 --batch-size $BS  --eps-noise ${noise_level} #| tee $LOG_FILE
done