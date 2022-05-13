#!/bin/bash

for order in 2 2; do
    for degree in $(seq 7 12); do
        julia --project=. batchfit_off.jl --rcut 10.0 10.0 --degree $degree --order $order > off_model_1106/out_off.$degree.$order 2>&1 &
    done
done

wait
