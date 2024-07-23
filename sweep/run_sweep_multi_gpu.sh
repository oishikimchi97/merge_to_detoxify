#!/bin/bash

SWEEP_ID=$1
multi_gpu=${2:-1}
num_gpu=${3:-$(nvidia-smi --list-gpus | wc -l)}

# Check if num_gpu is less than multi_gpu
if (( num_gpu < multi_gpu )); then
    echo "Error: num_gpu is less than multi_gpu"
    exit 1
fi

pids=()
j=0
for ((i=0; i<num_gpu; i+=multi_gpu)); do
    device_arr=($(seq $i $((i+multi_gpu-1))))
    device_map=$(echo "${device_arr[*]}" | tr ' ' ',')
    CUDA_VISIBLE_DEVICES=${device_map} wandb agent "${SWEEP_ID}" & pids[$j]=$!
    j=$((j+1))
done

wait "${pids[@]}"
