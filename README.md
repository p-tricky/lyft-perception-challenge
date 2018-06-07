# Lyft Perception Challenge

Code for 33rd place lyft perception challenge.

## Overview

The trained model performs pixel-wise semantic segmentation of roads and cars.  It was trained on data from the 
[Carla Simulator](https://github.com/carla-simulator/carla). You can generate your own training data by running the
simulator in semantic segmentation mode. The file in model/best.pt achieves a F<sub>2</sub>
score of .8155 oncars and an F<sub>.5</sub> score of .9856 on roads.  It runs at roughly 9 FPS

The model architectures and loss functions were borrowed from a 
[surgical instrument segmentation challenge](https://github.com/ternaus/robot-surgery-segmentation).

I settled on LinkNet34 for my final model architecture.  It was the only model that approached the target frame rate of 10.

## Install
This project uses PyTorch 0.4 and requires a GPU.  It was tested on a Tesla K80.
To install dependencies, setup [anaconda or miniconda](https://conda.io/docs/user-guide/install/download.html) 
and run $conda create --name unets --file unets.txt.

## Intersting take aways from the competition
Overall, my results are unremarkable; however, there was one fascinating bug that I observed during training. 
When I first trained the model, I randomly horizontally flipped the training images to reduce overfitting 
of the model.  This is a pretty standard technique, but I had messed it up by adding the random flips only to the
model inputs and not to the targets.  This means that every other data point in the training
set was garbage.  The crazy thing is that this monumental mess up had a pretty small impact on the final
performance of the model.  IIRC, the buggy model still achieved a .82 combined F score.  Fixing the bug only 
boosted the combined F score to .90.  It's so bizzare to me that these models are that robust to noisy data.
