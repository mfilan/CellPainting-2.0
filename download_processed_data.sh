#!/bin/bash

# This script needs to be run from the root of the project
mkdir -p data
cd data

names=(
#splitted_processed_0_25_50.zip,
#splitted_processed_patched.zip,
#merged_test_images.zip
processed_base.zip
)

urls=(
#"https://onedrive.live.com/download?cid=389519B65EF435AE&resid=389519B65EF435AE%2126961&authkey=ACoIKEV-MSBQ-us",
#"https://onedrive.live.com/download?cid=389519B65EF435AE&resid=389519B65EF435AE%2127000&authkey=AMNmXIBP_tjMeRo",
#"https://onedrive.live.com/download?cid=389519B65EF435AE&resid=389519B65EF435AE%2127001&authkey=AJdkPhg7kZpdA_E"
"https://onedrive.live.com/download?cid=389519B65EF435AE&resid=389519B65EF435AE%2120604&authkey=ABjTniFybb4T6VA"
)

# Download all the files
for i in ${!names[@]}; do
    curl -L --retry 7 -C - -o "${names[$i]}" "${urls[$i]}"
done

# Unpack them
for f in "${names[@]}"; do
    unzip "$f"
done
