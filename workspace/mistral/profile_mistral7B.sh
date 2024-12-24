#!/bin/bash

nsys profile --cudabacktrace=true --capture-range=cudaProfilerApi --capture-range-end=stop-shutdown --sample=none -f true -o ./profile-mistral-7B python3 run_mistral.py --mode profile --model_path mistral_weights/Mistral-7B-Instruct-v0.3 --eval_nItrs 1