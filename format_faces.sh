#!/bin/bash

DIRNAME="beaches"
for f in $(ls $DIRNAME)
do
 convert "$DIRNAME/$f" -resize "100x100!" "$DIRNAME/$f";
 echo "Done $f"
done

DIRNAME="faces"
mkdir newfaces
for f in $(ls $DIRNAME)
do
 
 R3=$(( $RANDOM % ($(ls -l beaches | wc -l) - 1) + 1 ))

 # Random size
 R4=$(( $RANDOM % 40 + 40))
 R5=$(( $RANDOM % 40 + 40))

 # Random offset
 R1=$(( $RANDOM % (100 - $R4) ))
 R2=$(( $RANDOM % (100 - $R5) ))
 convert "$DIRNAME/$f" -resize "${R4}x${R5}" "$DIRNAME/$f";
 convert -composite -geometry "+$R1+$R2" "beaches/$R3.jpg" "$DIRNAME/$f" "newfaces/${R1}x${R2}X${R4}x${R5}";
done
