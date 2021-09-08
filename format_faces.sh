#!/bin/bash

DIR_BEACHES="beaches"
DIR_FACES="faces"
DIR_NEWFACES="newfaces"
DIR_COLORS="colors"

echo "Resizing beaches..."
for f in $(ls $DIR_BEACHES)
do
 convert "$DIR_BEACHES/$f" -resize "100x100!" "$DIR_BEACHES/$f";
 echo "Done $f"
done
echo "Resized beaches!"

echo "Joining beaches and faces..."
mkdir newfaces
for f in $(ls $DIR_FACES)
do
 
 R3=$(( $RANDOM % ($(ls -l "$DIR_BEACHES" | wc -l) - 1) + 1 ))

 # Random size
 R4=$(( $RANDOM % 40 + 40))
 R5=$(( $RANDOM % 40 + 40))

 # Random offset
 R1=$(( $RANDOM % (100 - $R4) ))
 R2=$(( $RANDOM % (100 - $R5) ))
 convert "$DIR_FACES/$f" -resize "${R4}x${R5}" "$DIR_FACES/$f";
 convert -composite -geometry "+$R1+$R2" "$DIR_BEACHES/$R3.jpg" "$DIR_FACES/$f" "$DIR_NEWFACES/${R1}x${R2}X${R4}x${R5}";
done
echo "Joined beaches and faces!"

mkdir "$DIR_COLORS"
echo "Half done"
echo "Creating plain backgrounds..."
idx=0
for i in {0..255..15}; do
  for j in {0..255..15}; do
    for k in {0..255..15}; do
      convert -size 100x100 canvas:"rgb($i, $j, $k)" "$DIR_COLORS/$idx.png"
      idx=$(( idx+1 ))
    done
  done
done
echo "Created plain backgrounds!"

echo "Joining faces and backgrounds..."
for f in $(ls $DIR_FACES)
do
 
 R3=$(( $RANDOM % ($(ls -l "$DIR_COLORS" | wc -l) - 1) + 1 ))

 # Random size
 R4=$(( $RANDOM % 40 + 40))
 R5=$(( $RANDOM % 40 + 40))

 # Random offset
 R1=$(( $RANDOM % (100 - $R4) ))
 R2=$(( $RANDOM % (100 - $R5) ))
 convert "$DIR_FACES/$f" -resize "${R4}x${R5}" "$DIR_FACES/$f";
 convert -composite -geometry "+$R1+$R2" "$DIR_COLORS/$R3.png" "$DIR_FACES/$f" "$DIR_NEWFACES/${R1}x${R2}X${R4}x${R5}";
done
echo "Joined faces and backgrounds!"
