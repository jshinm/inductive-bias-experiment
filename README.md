# Inductive Bias Experiment
inductive bias experiment for deepnet paper

TODO:
- N/A

3/xx/21 <br>
- [ ] translate codes in an object-oriented manner
- [ ] implement training class

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