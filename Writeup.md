![SDAIA_Academy](https://user-images.githubusercontent.com/20911835/136685524-fda5c7dd-6f97-480b-bb69-4ae1ad02c460.jpeg)

# Adult Salary Prediction By Using Classification

Ghanim AlGhanim

Omran Fallatah

### Abstract

The goal of this project was to use classification to predict the salary bracket of adults(>$50k or <$50k). We worked with the data from <https://www.census.gov/> , leveraging feature Selection, feature engineering, dummy features, SMOTE. Then, we built  Logistic Regression, KNN, Decision Tree and Random Forest models. We concluded by comparing between the accuracy of each model

### Design

<!-- Add this
     Refer to the success guide --> 
### Data

Adults Salary dataset contains 48,000 data points and 15 features for each data point. A few feature highlights include Age, Gender, Work Class, Education and Marital Status. Our target feature is salary.

After cleaning the data, removing outliers, applying feature engineering, replacing NaN's with mode and dummy variables we ended up with 48,000 data points and 28 features.

### Algorithms

###### Data manipulation and cleaning.

-   Removed outliers.

-   Mapped categorical features into numerical.

-   Replaced NaN values with mode.

-   Applied dummy variables on Occupation feature.

-   Dropped unnecessary columns.


#### Model Evaluation and Selection

We split into 80/20 train and test respectively. The training dataset has 39073 data points and the test dataset has 9769 data points after the test/train split.
<!-- START HERE
     Add data for each model + results
     Get Acc, Pre etc for each model
     Insert plot for each model?
-->
###### Logistic Regression
###### SMOTE
###### K-Nearest Neighbor
###### Decision Tree
###### Random Forest

| Algorithm | Accuracy  | Precision | Recall | F-1 Score | ROC-AUC Score |
| ------------- | ------------- | ------------- | ------------- | ------------- | ------------- |
| SMOTE | 0.891 | 0.889  | 0.892  | 0.891 | 0.889  |
| Logistic Regression | 0.889  | 0.892  | 0.891 | 0.889  |
| K-Nearest Neighbor| 0.892  | 0.891  | 0.891 | 0.889  |
| Decision Tree | 0.891 | 0.889  | 0.892  | 0.891 | 0.889  |
| Random Forest | 0.891 | 0.889  | 0.892  | 0.891 | 0.889  |

<!-- Insert ROC CURVE PLOT -->

### Tools

-   Data manipulation and cleaning : Pandas , Numpy.
-   Plotting : Seaborn, Plotly and Matplotlib.
-   Modeling : Scikit-learn.

### Communication

In addition to the slides and the visuals included in the presentation, we will submit our code and proposal.
