# Human Activity Recognition using UCI's Human Activity Dataset

The notebook contains a Convolutional Neural Network and 2 LSTM models (One Normal and one Bi-Directional) to find out which model performs best in predicting the action (Walking, Standing, Sitting, etc.) done by the participant using time series accelerometer data recorded from a smartphone.

Each model is trained on two seperate instances 
1. Raw accelerometer data
2. Continuous Wavelet Transform of the raw data using Pywt 

to find out which model yields the best performance.

For the CWT we are using Morlet wavelets.

Here are some resources to learn about them:
1. https://web.ma.utexas.edu/users/davis/reu/ch3/cwt/manual.pdf
2. https://youtu.be/7ahrcB5HL0k

## References
### Using Dataset from this git repo:
https://github.com/jeandeducla/ML-Time-Series


### Original Dataset:
https://archive.ics.uci.edu/ml/datasets/Human+Activity+Recognition+Using+Smartphones

The dataset in the github repo is better formatted and labeled correctly as I think the Original UCI dataset is missing some labels although the github repo dataset is missing gyroscope data.

## Tensorboard

After running the entire notebook, you can open up the tensorboard analytics to check the model performance using the command ```tensorboard -logdir logs/fit/``` or to specify what model you want to see ```tensorboard -logdir logs/fit/cnn/``` for cnn 

The models performance are stored by date and time for everytime the model runs

Note: Running tensorboard directly on the notebook is possible but might not display correctly in the IPython output. Recommended to run it directly from the terminal.

## To run this notebook
### This notebook was run using Python version 3.9.7 and all other version of Python are untested
1. Clone this repo
2. (Optional) Create a virtual env
3. Download package dependancies using ```pip install -r requirements.txt```
4. (Optional) Download CUDA Toolkit and CUDNN for GPU acceleration
5. Open Model.ipynb