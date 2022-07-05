#!/bin/bash

# This script needs to be run from the root of the project
mkdir -p data/raw
cd data/raw

names=(
HepG2_Exp3_Plate1_FX9__2021-04-08T16_16_48.zip
HepG2_Exp3_Plate2_FX9__2021-04-09T10_25_07.zip
HepG2_Exp4_Plate2_FX9__2021-05-20T19_14_17.zip
)
urls=(
"https://onedrive.live.com/download?cid=389519B65EF435AE&resid=389519B65EF435AE%2121079&authkey=AH8YikEgasFZjTI"
"https://onedrive.live.com/download?cid=389519B65EF435AE&resid=389519B65EF435AE%2121080&authkey=AGa3brt_KEI5BxQ"
"https://onedrive.live.com/download?cid=389519B65EF435AE&resid=389519B65EF435AE%2121078&authkey=ADqVemIbu6SAL4E"
)
sha256s=(
2a240f36c1412e6b764e13774b774f2ab652c496f7ae5eee46f7ec5337b08bee
9168fff5fa556e1a32869484ffa4377086ec52ad0494f8d4613b620763047121
ea55aaa08af10c376ae38b0d016eb1025b8ebae5c3fce3e30bac05100dda6b6f
)

# Download all the files
for i in ${!names[@]}; do
    if [[ -f "${names[$i]}" ]]
    then
        if printf "%s  %s" "${sha256s[$i]}" "${names[$i]}" | shasum --check
	then
            # Don't re-download ${names[$i]} - checksum is  correct
            continue
        else
            rm "${names[$i]}"
	fi
    fi
    curl -L --retry 7 -C - -o "${names[$i]}" "${urls[$i]}"
done

# Unpack them
for f in "${names[@]}"; do
    unzip "$f"
done
