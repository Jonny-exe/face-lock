#!/bin/bash

FACE="custom_images/face.png"
BACKGROUND="custom_images/background.jpg"
DIR_NEWFACES="newsingle"
IDX=0

convert "$FACE" -resize "40x40" "$FACE";
mkdir "$DIR_NEWFACES"
echo "Joining faces and backgrounds..."
for R1 in {0..59..1}
do
  for R2 in {0..59..1}
  do
    convert -composite -geometry "+$R1+$R2" "$BACKGROUND" "$FACE" "$DIR_NEWFACES/${R1}x${R2}X40x40";
  done
done
echo "Joined faces and backgrounds!"
