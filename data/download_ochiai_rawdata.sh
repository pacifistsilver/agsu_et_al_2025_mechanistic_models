#!/bin/bash

len_sra_id_file=$(wc -l < sra_ids.txt)

for (( i=1 ; i <= len_sra_id_file ; i++ )); do
    echo $i
done