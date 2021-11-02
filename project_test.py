# %% codecell
import pandas as pd
import numpy as np
import matplotlib.pyplot as pplt
import seaborn as sns
from pandas.plotting import scatter_matrix
# %% codecell
df = pd.read_csv('adult.csv')
# %% codecell
df.describe()
# %% codecell
df.info()
# %% codecell
df.head()
# %% codecell
# find all ? values
df.isin(['?']).sum(axis=0)
# %% codecell
df.rename(columns={'capital-gain': 'capital gain', 'capital-loss': 'capital loss', 'native-country': 'country','hours-per-week': 'hours per week','marital-status': 'marital'}, inplace=True)
# %% codecell
df
# %% codecell
df['country'].unique()
# %% codecell
#replace all ? with NaN
df = df.replace('?', np.NaN)
# %% codecell
df.isna().sum()
# %% codecell
for c in df.columns:
    print ("---- %s ---" % c)
    print (df[c].value_counts())
# %% codecell
df.dropna(how='any',inplace=True)
df.isin(['?']).sum(axis=0)
# %% codecell
for c in df.columns:
    print ("---- %s ---" % c)
    print (df[c].value_counts())
# %% codecell
df.workclass.value_counts()
# %% codecell
df.drop(['educational-num','age', 'hours per week', 'fnlwgt', 'capital gain','capital loss', 'country'], axis=1, inplace=True)
# %% markdown
# # Mapping
# %% codecell
df['income'] = df['income'].map({'<=50K': 0, '>50K': 1}).astype(int)
df['gender'] = df['gender'].map({'Male': 0, 'Female': 1}).astype(int)
df['race'] = df['race'].map({'Black': 0, 'Asian-Pac-Islander': 1, 'Other': 2, 'White': 3,
                                             'Amer-Indian-Eskimo': 4}).astype(int)
df['marital'] = df['marital'].map({'Married-spouse-absent': 0, 'Widowed': 1,
                                                             'Married-civ-spouse': 2, 'Separated': 3, 'Divorced': 4,
                                                             'Never-married': 5, 'Married-AF-spouse': 6}).astype(int)
df['workclass'] = df['workclass'].map({'Self-emp-inc': 0, 'State-gov': 1,
                                                             'Federal-gov': 2, 'Without-pay': 3, 'Local-gov': 4,
                                                             'Private': 5, 'Self-emp-not-inc': 6}).astype(int)
df['education'] = df['education'].map({'Some-college': 0, 'Preschool': 1,
                                                        '5th-6th': 2, 'HS-grad': 3, 'Masters': 4,
                                                        '12th': 5, '7th-8th': 6, 'Prof-school': 7,
                                                        '1st-4th': 8, 'Assoc-acdm': 9,
                                                        'Doctorate': 10, '11th': 11,
                                                        'Bachelors': 12, '10th': 13,
                                                        'Assoc-voc': 14,
                                                        '9th': 15}).astype(int)
df['occupation'] = df['occupation'].map({ 'Farming-fishing': 1, 'Tech-support': 2,
                                          'Adm-clerical': 3, 'Handlers-cleaners': 4,
                                         'Prof-specialty': 5,'Machine-op-inspct': 6,
                                         'Exec-managerial': 7,
                                         'Priv-house-serv': 8,
                                         'Craft-repair': 9,
                                         'Sales': 10,
                                         'Transport-moving': 11,
                                         'Armed-Forces': 12,
                                         'Other-service': 13,
                                         'Protective-serv': 14}).astype(int)
df['relationship'] = df['relationship'].map({'Not-in-family': 0, 'Wife': 1,
                                                             'Other-relative': 2,
                                                             'Unmarried': 3,
                                                             'Husband': 4,
                                                             'Own-child': 5}).astype(int)

# %% codecell
df.head()
# %% codecell
df.groupby('education').income.mean().plot(kind='bar')
# %% codecell
df.groupby('occupation').income.mean().plot(kind='bar')
# %% codecell
df.groupby('relationship').income.mean().plot(kind='bar')
# %% codecell
df.groupby('gender').income.mean().plot(kind='bar')
# %% codecell
corrmat = df.corr()
f, ax = pplt.subplots(figsize=(12, 9))
k = 8 #number of variables for heatmap
cols = corrmat.nlargest(k, 'income')['income'].index
cm = np.corrcoef(df[cols].values.T)
sns.set(font_scale=1.25)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
pplt.show()

# %% codecell
df.hist(figsize=(12,9))
pplt.show()
# %% codecell
df_x = pd.DataFrame(df)
df_x = pd.DataFrame(np.c_[df['relationship'], df['education'], df['race'],df['occupation'],df['gender'],df['marital'],df['workclass']],
                    columns = ['relationship','education','race','occupation','gender','marital','workclass'])
y = pd.DataFrame(df.income)
# %% codecell
# Import scikit_learn module for the algorithm/model: Linear Regression
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
# Import scikit_learn module to split the dataset into train.test sub-datasets
from sklearn.model_selection import train_test_split

# Import scikit_learn module for k-fold cross validation
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

# import the metrics class
from sklearn import metrics

# %% codecell
# Train, Test Split
from sklearn.model_selection import train_test_split
#split dataset into train and test data
X_train, X_test, y_train, y_test = train_test_split(df_x, y['income'], test_size=0.25, random_state=1, stratify=y)
# %% codecell
# Build the Classification model using KNN
from sklearn.neighbors import KNeighborsClassifier
# Create KNN classifier
knn = KNeighborsClassifier(n_neighbors = 3)
# Fit the classifier to the data
knn.fit(X_train,y_train)
# %% codecell
# Testing the model
knn.predict(X_test).reshape(-1,1)
# %% codecell
knn.score(X_test, y_test)

# %% codecell
knn.score(X_train, y_train)
# %% codecell
from sklearn.model_selection import cross_val_score
#create a new KNN model
KNN_CV = KNeighborsClassifier(n_neighbors=2)
#train model with CV of 5
cv_scores = cross_val_score(KNN_CV, df_x, y['income'], cv=5)
#print each CV score (accuracy) and find the average
print('The scores are:', cv_scores)
print(f'The mean score is {np.mean(cv_scores)}')
# %% codecell
from sklearn.model_selection import GridSearchCV
# Create a new KNN model
KNN_2 = KNeighborsClassifier()
# Create a dictionary of all values we want to test for n_neighbors
param_grid = {'n_neighbors': np.arange(1, 25)}
# use gridsearch to test all values for n_neighbors
KNN_gscv = GridSearchCV(KNN_2, param_grid, cv=5)

#fit model to data
KNN_gscv.fit(df_x, y['income'])

# %% codecell
# Check top performing n_neighbors value
KNN_gscv.best_params_
# %% codecell
KNN_gscv.best_estimator_
# %% codecell
# Check mean score for the top performing value of n_neighbors
KNN_gscv.best_score_
# %% codecell
# Build the Classification model using KNN with the best parameters
from sklearn.neighbors import KNeighborsClassifier
# Create KNN classifier
knn = KNeighborsClassifier(n_neighbors = 24)
# Fit the classifier to the data
knn.fit(X_train,y_train)
# %% codecell
#check accuracy of our model on the test data
knn.score(X_train, y_train)
# %% codecell
#check accuracy of our model on the test data
knn.score(X_test, y_test)
