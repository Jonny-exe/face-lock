#!/bin/bash

DIRNAME="faces"
mkdir $DIRNAME
for i in {0..10000..1}
do
  wget -q -O tmp.jpeg https://thispersondoesnotexist.com/image
  convert tmp.jpeg -resize 80x80! "$DIRNAME/$i"
  #convert "$DIRNAME/$i" -bordercolor white -border 20x20 "$DIRNAME/$i"
  echo "Wrote $i / $1"
done

mkdir beaches && cd beaches && wget -q -w 3 https://thisbeachdoesnotexist.com/data/seeds-075/{1..9999}.jpg
