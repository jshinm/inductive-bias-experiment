# Inductive Bias Experiment
inductive bias experiment for deepnet paper

<!-- 
TODO:
- N/A 
-->

5/5/21 <br>
- [x] retrain ML models at the same sample size as human experiment
- [x] human true posteriors have been fixed and hellinger distances were recalculated
- [x] smooth_gaussian_distance can now return the same sized grid and uses the single nearest neighbor for interpolation followed by multidirectional 1-D guassian smoothing

4/19/21 <br>
- [x] Corrected gaussian smoothing experiment
    - method of smoothing changed from convolving circle + 2D gaussian to nearest neighbor + multidimentional 1D gaussian (scipy) 

4/14/21 <br>
- [x] Update the true posterior tutorial
- [x] Remove the outdated methods from dataset_generator and add brief docstring for each method

3/30/21 <br>
- [x] Recode the whole project in an object-oriented manner 
    - implement the dataset loader
    - implement the dataset trainer
    - implement the dataset generator
    - implement the tool for analysis
    - implement the model trainer
- [x] Retrain ML models with lesser sample points of simulation datasets (750->100) to match that of human experiment
- [x] fix jagged spiral posterior
    - resolved by increasing the number of spiral center
- [x] Explore mean and variance of posterior and hellinger distance with gaussian smoothing
- [x] implement training class
    - wrote a subclass to interact with the base class
- [x] Simulate human behavioral experiment setting on ML experiment

3/16/21 <br>
- [x] extract the estimated posterior along the line at 135 degree
- [x] draw line plot of the mean estimated posterior along the line
- [x] plot pre-activation & post-activation of the MLP model (i.e. activation of the last output layer)
- [x] draw line plot of the mean estimated posterior along the line (between pre-post activation of MLP)

12/30/20 <br>
- [x] implement spiral posterior

<!-- 12/20/20 <br>

1. plot on the top row: class 1 likelihood, sample data, class 1 posterior
2. plot on bottom row: 3 estimated posteriors
3. make all the plots circular with radius 4
4. top row: show class 1 posterior curves
5. bottom row: show class 1 hellinger distance curves
6. for the posterior estimates, label with alg name & their mean hellinger distance -->

12/14/20 <br>
- [x] remove KNN and XGBoost from the figure

12/08/20 <br>
- [x] generate circular bbox posterior/hellinger plots with extended range (0,3) 
- [x] remove SVC and rename nuSVC as SVM

12/05/20 <br>
- [x] produce radial posterior/hellinger plots (hellinger vs euclidean; posterior vs euclidean), for in/outside of the circular bbox, to see which algo is more confident outside the range of the training data 

11/22/20 <br>
- [x] implement circular bbox