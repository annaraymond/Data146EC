# Liberia: Analyzed
## A deep dive into the analysis of Liberia's Humanitarian Data
This project was performed almost entirely on jupyter-hub as provided access by The College of William and Mary's Data Science Department. 
Python was used to complete these regressions and the following version was run: 
The following pacakges would be required to implement the same analysis as I have: Numpy, Matplotlib as imported from pyplot, pandas, and sklearn
My full script will be provided below for replication purposes, comments will provide as useful guides to the operations I am performing, if you have any questions reach out to me. 
## An Initial Description
### What the data set looks like, where we're going
This data set is a sociopolitical and wealth descriptor set of data surrounding Liberia. It contains six columns of information: location, size, wealth, gender, age, and education. We will be using the former five columns to predict the latter column- education. This can be referred to as the target. 

I produced an initial pairplot, using the Python package seabornplot, to look at the relationships before attempting any classification methods because it is important to do a quick scan of the covariates to detemine if there are any already visible and clear relationships. Since pairplots tend to be quite computationally expensive, to replicate this analysis I would only recommend doing so on a publicly or privately provided compuational network such as William and Mary's Jupyter-Hub. The pairplot is provided below, it demonstrates some strong correlative relationships which indicate that an analysis on this data set will most likely be fruitful. 
#### Image 1: Pairplot of Liberia's Covariates
<img src="Paiplot.png" width="500">

I also produced an initial heatmap of the data set in order to vizualize possible and existing correlations, simply in a different manner. An important note, on this heatmap the diagonal line sloping downwards and right is not indicative of a strong correlation between two separate covariates. It is simply the heatmap comparing the variable to itself. Other than that, the more intense the color, the stronger the correlation. If the color is red, it signifies a positive correlation. If the color is blue, it signifies a negative correlation. Both of these relationships can be helpful in the ultimate analysis of this dataset.
#### Image 2: Heatmap of Liberia's Covariates
<img src="Heatmap.png" width = "500">

In addition to these, my final method of viewing the data pre-regression was to produce a histogram with all the different covariates layered on to look for any other visible trends. I began, initially, with simply plotting the dataset. I had to modify the number of bins, the width of the bars, the legends, and the opacity of each of the layers to produce a readible plot. 
#### Image 3: Histogram of Liberia's Covariates
<img src = "Unscaled_Histogram.png" width = "500">

This plot, however, was largely overtaken by the existance of the 'age' column which has a strong tendancy to pull the graph right due to the temporal aspect of this particular covariate. To combat this, and to partially visualize what the standardized data which I will be using later in the analysis will look like, I standardized that dataset, and replotted it onto the histogram. I removed an additional outlier to the right at around x = 12 (which is still related to the age's temporal aspect) and adjusted the bins accordingly for a hopefully clearer histogram.
#### Image 4: Scaled Histogram of Liberia's 40 
<img src = "Scaled_Histogram.png" width = "500">

## Logistic Regression - the initial classifer 
To begin to attempt to classify educaiton through the existance of the other features, I used a logistic regression. Simply running the regression without any changes in the standardization of the data or the hyperparameters of the logistic regression gives us an accuracy of .57025. This is not a bad jumping off point and is highly indicative of the trends seen above in the pre analysis analysis. 
 
### Standard Scaler 
I, again ran the logistic regression, except this time on the standard scaler feature available from the sklearn.preprocessing library. This yielded a slightly better accuracy of .57030. While this is better, it is nowhere near enough of a positive increase in accuracy to deem this form of standardization useful 

### MinMax Scaler 
Once again, I ran the logistic regression on the same data but using the MinMaxScaler feature also available from the sklearn.preprocessing library. This created a worse outpyt of .5701 accuracy therby indicating that I could retire the MinMaxScaler from use for the rest of the analysis on this dataset

### Normalizer 
I ran the logistic regression with a normalizer as well, also available from the sklearn.preprocessing library. This attempt at preprocessing the data yielded an accuracy of .64652, the highest so far by a LOT considering the only aspect of the regression that I changed was the preprocessing. 

### Robust Scaler 
Finally, I tested the logistic regression with a robust scaler and it produced around the same accuracy as the other preprocessing methods. Even though these different scalers provided different accuracies, I still tested all of them for other models because different preprocessing methods function better for different models

## KNN Classification 
As a second attempt to classify this dataset I ran a K nearest neighbors model validation on this dataset with all the scalers mentioned above. To accomplish this I tested a range of K neighbors to find the best fit for the scaled and unscaled dataset. While looping through, I appended the accuracies and found the maximum accuracy for whichever argument of neighbor that was. The best KNN accuracy was .709. This was achieved through using a MinMaxScaler and 40 neighbors. 
### K- Fold Validation 
I ran a K Fold validation on this same instance of the best KNN accuracy, and I got an accuracy of .72293 which is higher than the KNN classification by itself and the highest accuracy I have generated so far

## Decision Tree Classifier 
As a third attempt to classify this dataset I ran a Decision Tree classifier on this dataset with all the preprocessing scalers I have been using. I generally found my models to be more accurate when I modified the max depth. The highest accuracy I got was .72094 on the unscaled dataset with K fold validation of 15 and a max depth of 5. Scaling datasets does not appear to affect decision trees that much (simply because of the binary search-like structure of the tree). I also created a graph of the these max depths and the accuracies to demonstrate which max depth I picked and why. 
#### Image 5: Max Depth Accuracies  
 <img src = "Max_depthh.png" width = "500">
 















