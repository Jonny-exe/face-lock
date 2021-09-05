# face-lock

# Plan

## Process
1. Recognize what is the face
2. Recognize if the face passes
3. Go through with unlocking


## 1

- [ ] Make a CNN for recognizing which part of an image is a face
  - [ ] Somehow generate trainnning data for CNN. 
  (Images with faces with a square)
  - [ ] Train a model with data gotten from a webcam video

Notes: 
  - Learning rate has to be way higher than normal or recommended
  - Learning rate scheduler steps must be higher than usual two
  - Bigger batch sizes make generalizing easier
  - Data amount is not that important

## 2
- [ ] Make a CNN to recognize if the face is the same as the users
  - [ ] Crop the image from the user
  - [ ] Create a CNN for comparing faces.

## 3
- [ ] Find a way to unlock login or sudo access

