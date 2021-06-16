# Inductive Bias Experiment
inductive bias experiment for deepnet paper

# Log

## 6/16/21
- change the color scheme for rotational line plot

## 5/24/21
- fix the colorbar range (equalize to `range(0,1)`)
- produce line plot over 0-180 degree
- match the exact coordinates between human and machine

## 5/13/21 
- assess estimated posteriors over a line drawn at 135 degree
- MTurk simulator written in jupyter

## 5/5/21
- retrain ML models at the same sample size as human experiment
- human true posteriors have been fixed and hellinger distances were recalculated
- smooth_gaussian_distance can now return the same sized grid and uses the single nearest neighbor for interpolation followed by multidirectional 1-D guassian smoothing

## 4/19/21
- Corrected gaussian smoothing experiment
    - method of smoothing changed from convolving circle + 2D gaussian to nearest neighbor + multidimentional 1D gaussian (scipy) 

## 4/14/21
- Update the true posterior tutorial
- Remove the outdated methods from dataset_generator and add brief docstring for each method

## 3/30/21
- Recode the whole project in an object-oriented manner 
    - implement the dataset loader
    - implement the dataset trainer
    - implement the dataset generator
    - implement the tool for analysis
    - implement the model trainer
- Retrain ML models with lesser sample points of simulation datasets (750->100) to match that of human experiment
- fix jagged spiral posterior
    - resolved by increasing the number of spiral center
- Explore mean and variance of posterior and hellinger distance with gaussian smoothing
- implement training class
    - wrote a subclass to interact with the base class
- Simulate human behavioral experiment setting on ML experiment

## 3/16/21
- extract the estimated posterior along the line at 135 degree
- draw line plot of the mean estimated posterior along the line
- plot pre-activation & post-activation of the MLP model (i.e. activation of the last output layer)
- draw line plot of the mean estimated posterior along the line (between pre-post activation of MLP)

## 12/30/20
- implement spiral posterior

<!-- 12/20/20 <br>

1. plot on the top row: class 1 likelihood, sample data, class 1 posterior
2. plot on bottom row: 3 estimated posteriors
3. make all the plots circular with radius 4
4. top row: show class 1 posterior curves
5. bottom row: show class 1 hellinger distance curves
6. for the posterior estimates, label with alg name & their mean hellinger distance -->

## 12/14/20
- remove KNN and XGBoost from the figure

## 12/08/20
- generate circular bbox posterior/hellinger plots with extended range (0,3) 
- remove SVC and rename nuSVC as SVM

## 12/05/20
- produce radial posterior/hellinger plots (hellinger vs euclidean; posterior vs euclidean), for in/outside of the circular bbox, to see which algo is more confident outside the range of the training data 

## 11/22/20
- implement circular bbox