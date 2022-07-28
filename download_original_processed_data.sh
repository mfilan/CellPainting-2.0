#!/bin/bash

# This script needs to be run from the root of the project
mkdir -p data/original_processed
cd data/original_processed

names=(
original_processed.zip
)
urls=(
"https://onedrive.live.com/download?cid=389519B65EF435AE&resid=389519B65EF435AE%2127008&authkey=AKFqcU_8OjJNFFo"
)
# Download all the files
for i in ${!names[@]}; do
    curl -L --retry 7 -C - -o "${names[$i]}" "${urls[$i]}"
done

# Unpack them
for f in "${names[@]}"; do
    jar xvf "$f"
done