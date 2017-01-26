# Udacity Self-Driving Car Nanodegree, Behavioural Cloning Project (Project 3)

## Introduction:

*The object of the project is to train a model to drive a car autonomously on a simulated track. 
The ability of the model to drive the car is gotten from cloning the behaviour of a human driver.
Training data is gotten from examples of a human driving in the simulator. The training data is then fed into a deep learning network which learns the response (steering angle) for every encountered frame in the simulation. In other words, the model is trained to predict an appropriate steering angle for every frame while driving. The model is then validated on a new track to check for generalization of the features learnt by the model for performing steering angle prediction.*

This project is influenced by [nvidia paper](https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf), [comma.ai paper](https://arxiv.org/pdf/1608.01230v1.pdf) and [vivek's blog](https://chatbotslife.com/using-augmentation-to-mimic-human-driving-496b569760a9#.5dpi87xzi) which I consulted while working on my solution. The [Keras Deep Learning library](https://keras.io/) was used with [Tensorflow](https://www.tensorflow.org/) backend, to perform deep learning operations.

## Outline
**
1. Data Recording

2. Data Pre-processing

3. Training

4. Model Testing

5. Discussion

6. Conclusion
**

## 1. Data Recording

The simulator has two modes - Training mode and Autonomous mode. Training mode is used to collect training data by driving through the tracks and recording the driving data in a folder. The Autonomous mode is used to test a trained model. 

Simulator splash screen: ![Simulator splash screen](/media/simulator_splash.png "Simulator Splash Screen")

Udacity provided a set of training data (24,108 datasets) which can be downloaded with the simulator. I thought the the Udacity might not be enough so I recorded my own training data (104,145 datasets) and would use the Udacity data for validation. Plotting an histogram of a sample of the training data shows that the data from track 1 has more 0  and left steering angles because of the nature of the track, so our pre-processing step will also include data augmentation.


## 2. Data Pre-processing

Data pre-processing is done to allow our model to be able to easily work with raw data for training. In this project, the data pre-processing is built into a generator (keras fit generator) to allow for real-time pre-processing of the data. The advantage here is that, in the case that we are working with a very large amount of data, the whole dataset is not needed to be loaded into memory; We can work with a manageable batch of data at a time. Hence the generator is run in parallel to the model, for efficiency.

The following are the pre-processing steps carried on the data:

1. Translate image and compensate for steering angle

## Heading 2

### Heading 3

**Bold sub-head**

link:
[github profile](https://github.com/toluwajosh)

- bullet point

1. Numbered line

code snippet: 

`conda install -c https://conda.anaconda.org/menpo opencv3`

block code snippet:

```
git clone https://github.com/udacity/CarND-Traffic-Signs
cd CarND-Traffic-Signs
jupyter notebook Traffic_Signs_Recognition.ipynb
```