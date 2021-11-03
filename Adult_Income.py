import pandas as pd
import numpy as np
import matplotlib.pyplot as pplt
import seaborn as sns
from pandas.plotting import scatter_matrix

df = pd.read_csv('adult.csv',na_values='?')

df.describe()

df

sns.countplot(x='income', data=df)

df.rename(columns={'capital-gain': 'capital gain', 'capital-loss': 'capital loss', 'native-country': 'country','hours-per-week': 'hours per week','marital-status': 'marital'}, inplace=True)

df

df['country'].unique()

print(f"{df[df['capital gain'] == 99999].shape[0]} outlier in the capital-gain")
print(f"{df[df['hours per week'] == 99].shape[0]} outlier in the hours-per-week")

df['capital gain'].replace(99999, np.mean(df['capital gain'].values), inplace=True)
df['hours per week'].replace(99, np.mean(df['hours per week'].values), inplace=True)

df["income"].value_counts()/len(df["income"])

# Mapping

df["gender"] = df["gender"].map({'Female':0, 'Male':1})

df["race"] = df["race"].map({'White':0, 'Black':1, 'Asian-Pac-Islander':2, 'Amer-Indian-Eskimo':3})

df["marital"] = df["marital"].map({'Widowed':0, 'Divorced':1, 'Separated':2,'Never-married':3,
                                                 'Married-civ-spouse':4, 'Married-spouse-absent':5, 'Married-AF-spouse':6})

df["relationship"] = df["relationship"].map({'Not-in-family':0, 'Unmarried':0, 'Own-child':0, 'Other-relative':0,
                                             'Husband':1, 'Wife':1})

df['workclass'] = df['workclass'].map({'?':0, 'Private':1, 'State-gov':2, 'Federal-gov':3,
                                       'Self-emp-not-inc':4, 'Self-emp-inc': 5, 'Local-gov': 6,
                                       'Without-pay':7, 'Never-worked':8})

df["income"] = df["income"].map({'<=50K':0, '>50K': 1})

#df.isna().sum()
df.fillna(df.mode().iloc[0], inplace=True)


df["occupation"].unique()

df['occupation'] = df['occupation'].apply(lambda x: x.replace('?', 'Unknown'))

occupation_df = pd.get_dummies(df["occupation"])
df = pd.concat([df, occupation_df], axis=1)
df.drop(["occupation"], axis=1, inplace=True)
df

df['country'] = df['country'].apply(lambda x: 1 if x.strip() == "United-States" else 0)
df

df[['education', 'educational-num']].groupby(['education'], as_index=False).mean().sort_values(by='educational-num', ascending=False)

df.drop(["education"], axis=1, inplace=True)
df

df.info()

df.isnull().sum()

X = df.drop(["income"], axis=1)
y = df["income"]


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=0, stratify=y)

print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

from sklearn.model_selection import GridSearchCV

from sklearn.metrics import classification_report

def evaluation(model, X_train, y_train, X_test, y_test, is_NN=False):
    y_pred = model.predict(X_test)

    if is_NN:
        y_pred = np.array(list(map(int, y_pred > 0.5)))

    print("Best model esitmator", model)
    print(classification_report(y_test, y_pred, target_names = ['Under 50k', 'Over 50k']))

    if not is_NN:
        print('Training Set Accuracy Score: {:.2f}'.format(model.score(X_train, y_train)))
        print('Testing Set Accuracy Score: {:.2f}'.format(model.score(X_test, y_test)))

    return y_pred

from sklearn.metrics import confusion_matrix

def confusion(y_true, y_pred):

    confusion_mat = confusion_matrix(y_true, y_pred)
    confusion_df = pd.DataFrame(confusion_mat)
    pplt.figure(figsize=(8,8))
    sns.heatmap(confusion_df, annot=True)

from sklearn.linear_model import LogisticRegression

def logistic_regression(X_train, y_train):
    logreg = LogisticRegression()
    grid_values = {'penalty' : ['l1', 'l2'],
                   'C': [0.01, 0.1, 1, 10, 100],
                    'solver' : ['liblinear']}

    grid_lr_rec = GridSearchCV(logreg, param_grid = grid_values, scoring ='accuracy', cv=5)
    grid_lr_rec.fit(X_train, y_train)

    return grid_lr_rec.best_estimator_

# Log Reg

logreg = logistic_regression(X_train, y_train)
y_pred = evaluation(logreg, X_train, y_train, X_test, y_test, is_NN=False)
confusion(y_test, y_pred)

# Random Forest

from sklearn.ensemble import RandomForestClassifier

def random_forest(X_train, y_train):
    clf = RandomForestClassifier(random_state=0)

    grid_values = {'max_depth': np.arange(1,11,2),
                   'max_features': np.arange(1,11,2)}

    grid_clf = GridSearchCV(clf, param_grid = grid_values, scoring ='accuracy', cv=5)
    grid_clf.fit(X_train, y_train)


    return grid_clf.best_estimator_

random_for = random_forest(X_train, y_train)
y_pred = evaluation(random_for, X_train, y_train, X_test, y_test, is_NN=False)
confusion(y_test, y_pred)

# KNN

from sklearn.neighbors import KNeighborsClassifier

def knn(X_train, y_train):
    clf = KNeighborsClassifier()

    grid_values = {'n_neighbors': [23, 25, 35]}

    grid_clf = GridSearchCV(clf, param_grid = grid_values, scoring ='accuracy', cv=5)
    grid_clf.fit(X_train, y_train)


    return grid_clf.best_estimator_

knn_clr = knn(X_train, y_train)
y_pred = evaluation(knn_clr, X_train, y_train, X_test, y_test, is_NN=False)
confusion(y_test, y_pred)

# SVM

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=True, test_size=0.2, random_state=0)

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

from sklearn.svm import SVC

def svm(X_train, y_train):
    clf = SVC(class_weight={1: 3})

    #grid_values = {'kernel': ['rbf', 'linear'],
    #               'C': [1, 0.8],
    #               'degree': [1, 2]}
    # take much time
    grid_values = {}

    grid_clf = GridSearchCV(clf, param_grid = grid_values, scoring ='accuracy', cv=5)
    grid_clf.fit(X_train, y_train)


    return grid_clf.best_estimator_

svm_clr = svm(X_train, y_train)
y_pred = evaluation(svm_clr, X_train, y_train, X_test, y_test, is_NN=False)
confusion(y_test, y_pred)

# SMOTE

from imblearn.over_sampling import SMOTE

sm = SMOTE(random_state=0)
X_up_train, y_up_train = sm.fit_resample(X_train, y_train)

y_up_train.value_counts()

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
X_up_train = scaler.fit_transform(X_up_train)
X_test = scaler.transform(X_test)

from sklearn.ensemble import AdaBoostClassifier

def ada_boost(X_train, y_train):
    clf = AdaBoostClassifier()

    grid_values = {'n_estimators': [50, 70,100]}

    grid_clf = GridSearchCV(clf, param_grid = grid_values, scoring ='accuracy', cv=5)
    grid_clf.fit(X_train, y_train)


    return grid_clf.best_estimator_

ada_boost_clr = ada_boost(X_up_train, y_up_train)
y_pred = evaluation(ada_boost_clr, X_up_train, y_up_train, X_test, y_test, is_NN=False)
confusion(y_test, y_pred)

# Gradient Boost

from sklearn.ensemble import GradientBoostingClassifier

def gradient_boost(X_train, y_train):
    clf = GradientBoostingClassifier()

    grid_values = {'n_estimators': [50, 70,100]}

    grid_clf = GridSearchCV(clf, param_grid = grid_values, scoring ='accuracy', cv=5)
    grid_clf.fit(X_train, y_train)


    return grid_clf.best_estimator_

grad_boost_clr = gradient_boost(X_up_train, y_up_train)
y_pred = evaluation(grad_boost_clr, X_up_train, y_up_train, X_test, y_test, is_NN=False)
confusion(y_test, y_pred)


grad_boost_clr = GradientBoostingClassifier(n_estimators=200, learning_rate=0.1, min_samples_split=500, min_samples_leaf=50,
                                            max_depth=8, max_features='sqrt', subsample=0.8, random_state=42)

grad_boost_clr.fit(X_up_train, y_up_train)
y_pred = evaluation(grad_boost_clr, X_up_train, y_up_train, X_test, y_test, is_NN=False)
confusion(y_test, y_pred)

# Decision Tree

from sklearn.tree import DecisionTreeClassifier

def decision_tree(X_train, y_train):
    clf = DecisionTreeClassifier()

    grid_values = {}

    grid_clf = GridSearchCV(clf, param_grid = grid_values, scoring ='accuracy', cv=5)
    grid_clf.fit(X_train, y_train)


    return grid_clf.best_estimator_

rad_boost_clr = decision_tree(X_up_train, y_up_train)
y_pred = evaluation(grad_boost_clr, X_up_train, y_up_train, X_test, y_test, is_NN=False)
confusion(y_test, y_pred)

from sklearn.tree import DecisionTreeClassifier
dtree = DecisionTreeClassifier()
dtree.fit(X_train,y_train)
predictions = dtree.predict(X_test)
accuracy_score = (y_test,predictions)
print(classification_report(y_test,predictions))

# cross val?


### cross validation
from sklearn.model_selection import KFold #for K-fold cross validation
from sklearn.model_selection import cross_val_score #score evaluation
from sklearn.model_selection import cross_val_predict #prediction
kfold = KFold(n_splits=10) # k=10, split the data into 10 equal parts
xyz=[]
accuracy=[]
std=[]
classifiers=['Logistic Regression','Decision Tree','KNN','Random Forest']
models=[LogisticRegression(),DecisionTreeClassifier(),
        KNeighborsClassifier(n_neighbors=24),RandomForestClassifier(n_estimators=100)]
for i in models:
    model = i
    cv_result = cross_val_score(model,X,y, cv = kfold,scoring = "accuracy")
    cv_result=cv_result
    xyz.append(cv_result.mean())
    std.append(cv_result.std())
    accuracy.append(cv_result)
models_dataframe=pd.DataFrame({'CV Mean':xyz,'Std':std},index=classifiers)
models_dataframe

# ROC comparison


from sklearn.metrics import plot_roc_curve
disp = plot_roc_curve(logreg, X_test, y_test)
plot_roc_curve(dtree, X_test, y_test, ax=disp.ax_)
plot_roc_curve(knn_clr, X_test, y_test, ax=disp.ax_)
plot_roc_curve(random_for, X_test, y_test, ax=disp.ax_)

pplt.show()
