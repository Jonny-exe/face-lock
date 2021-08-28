#!/bin/bash

mkdir images
for i in {0..10000..1}
do
  wget -q -O tmp.jpeg https://thispersondoesnotexist.com/image
  convert tmp.jpeg -resize 400x400! "images/$i.jpeg"
  echo "Wrote $i / $1"
done
