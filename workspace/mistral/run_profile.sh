#!/bin/bash

nsys profile --cudabacktrace=true --capture-range=cudaProfilerApi --capture-range-end=stop-shutdown --sample=none -f true -o ./profile python3 MistralTest.py --mode profile --eval_nItrs 1