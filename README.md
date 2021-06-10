[![Build Status](https://travis-ci.com/JohnsonZhouUoa/AutoDDM.svg?branch=main)](https://travis-ci.com/JohnsonZhouUoa/AutoDDM)
# AutoDDM

Implementation for the paper "Automated Drift Detector Tuning for Data Streams".

> Developing accurate online machine learning models is a difficult task that usually requires the models to adapt when changes occur in the data. Concept drift detectors have been designed to detect these changes. Most of the detectors are non-parametric and do not assume a distribution within the data. However, these detectors have parameters, which should be tuned, such as drift tolerance levels. Tuning these parameters is non-trivial and necessary to cater for changes in drift behaviour. We propose AutoDDM that adjusts the drift thresholds based on prior information. We exploit the periodicity in the data stream when it exists, such that it is more sensitive to true concept drifts while reducing false-positive detections. The experiments show an accuracy improvement of the models by an average of 1.45\%.

### Requirements

Make sure the following dependencies are installed:

* Python &ge; 3.6


### Installing

```bash
git clone https://github.com/JohnsonZhouUoa/AutoDDM.git --recursive
pip install -r requirements.txt
```

### Example
Run `AutoDDMTestWeather` or `AutoTestElectricityData` for an example of running AutoDDM.

It should give you results look like the following:

![covtype results](https://scikit-ika.github.io/scikit-ika/_images/ErrorRate.PNG)

##### Real World Dataset

The original datasets are available on MOA and Kaggle, see references in the paper.

##### Synthetic Dataset

The project uses synthetic data generators from [Scikit-Ika](https://github.com/scikit-ika/scikit-ika/tree/master/skika/data). 
