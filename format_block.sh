#!/bin/bash

FACE="custom_images/white.jpg"
BACKGROUND="custom_images/black.jpg"
DIR_NEWFACES="newsingle"
IDX=0

convert "$FACE" -resize "60x60" "$FACE";
mkdir "$DIR_NEWFACES"
echo "Joining faces and backgrounds..."
for R1 in {0..39..1}
do
  for R2 in {0..39..1}
  do
    convert -composite -geometry "+$R1+$R2" "$BACKGROUND" "$FACE" "$DIR_NEWFACES/${R1}x${R2}X$((60 + $R1))x$((60 + $R2))";
  done
done
echo "Joined faces and backgrounds!"
