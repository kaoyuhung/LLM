#!/bin/bash

if [ $# -lt 2 ]; then
  echo "Usage: $0 <server-addr> <port>"
  exit 1
fi

if [[ "`hostname -I`" == *"$1"* ]]; then  

    if [ -z "$SLURM_JOB_ID" ]; then
      iperf3 -s -1 -p $2
    else
      iperf3 -s -1 -p $2 >> "result/job$SLURM_JOB_ID/server.out"
      ib_send_bw >> "result/job$SLURM_JOB_ID/server.out"
    fi

else
    
    if [ -z "$SLURM_JOB_ID" ]; then
      iperf3 -c $1 -p $2
    else
      iperf3 -c $1 -p $2 >> "result/job$SLURM_JOB_ID/client.out"
      ib_send_bw $1 >> "result/job$SLURM_JOB_ID/client.out"
    fi
    
fi