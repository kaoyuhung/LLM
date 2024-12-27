#!/bin/bash

nsys profile --cudabacktrace=true --capture-range=cudaProfilerApi --capture-range-end=stop-shutdown --sample=none -f true -o $0\
    python3 run_mistral.py --mode nsys_profile --model "Mistral-7B-Instruct-v0.3" --model_path "weights/Mistral-7B-Instruct-v0.3" --eval_nItrs 1