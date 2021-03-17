#!/bin/bash
mkdir -p figs
for i in $(ls data); do
    printf "starting file "$i" ..."
    python3 ../../../../snc/experiment/demo_gsm.py --path ./data --network $i --no-solution 
    mv "figs/"$i "figs/"${i%.csv}".graph"
done


