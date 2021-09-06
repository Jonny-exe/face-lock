#!/bin/bash

FILENAME="$1"
R1="$2"
R2="$3"
R4="$4"
R5="$5"

R6=$(($R1+$R4))
R7=$(($R5+$R2))

convert "$FILENAME" -alpha on -fill "#00000000" -stroke black \
    -draw "rectangle $R1,$R2 $R6,$R7" "output_image.jpeg"
