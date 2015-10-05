__author__ = 'pranavagarwal'

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# import Linear Regression
from sklearn.linear_model import LinearRegression, LogisticRegression
# Import for Cross Validation
from sklearn.cross_validation import KFold
from sklearn import cross_validation
from sklearn.ensemble import RandomForestClassifier

titanic = pd.read_csv("/Users/pranavagarwal/Downloads/train.csv")
titanic["Age"] = titanic["Age"].fillna(titanic["Age"].median())
titanic.loc[titanic["Sex"] == "male", "Sex"] = 0
titanic.loc[titanic["Sex"] == "female", "Sex"] = 1
titanic["Embarked"] = titanic["Embarked"].fillna("S")
titanic.loc[titanic["Embarked"] == "S", "Embarked"] = 0
titanic.loc[titanic["Embarked"] == "C", "Embarked"] = 1
titanic.loc[titanic["Embarked"] == "Q", "Embarked"] = 2
# The columns we'll use to predict the target
predictors = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
#predictors = ["Pclass", "Sex", "Age"]
'''
groupByEmbark = titanic.groupby(['Embarked', 'Survived'])
titanic["Sex"].hist(by=titanic["Survived"])
plt.show()
print(groupByEmbark['Survived'].count())

# Initialize our algorithm class

alg = LinearRegression()

# Generate cross validation folds for the titanic dataset.  It return the row indices corresponding to train and test.
# We set random_state to ensure we get the same splits every time we run this. ie breaking the data set in 3
kf = KFold(titanic.shape[0], n_folds=3, random_state=1)
predictions = []

for train, test in kf:
    # The predictors we're using the train the algorithm.  Note how we only take the rows in the train folds.
    train_predictors = (titanic[predictors].iloc[train, :])
    # print(train_predictors)
    # The target we're using to train the algorithm.
    train_target = titanic["Survived"].iloc[train]
    # print(train_target)
    # Training the algorithm using the predictors and target.
    alg.fit(train_predictors, train_target)
    # We can now make predictions on the test fold
    test_predictions = alg.predict(titanic[predictors].iloc[test, :])
    predictions.append(test_predictions)

# The predictions are in three separate numpy arrays.  Concatenate them into one.
# We concatenate them on axis 0, as they only have one axis.
predictions = np.concatenate(predictions, axis=0)

# Map predictions to outcomes (only possible outcomes are 1 and 0)
predictions[predictions > .5] = 1
predictions[predictions <=.5] = 0
#predictions[predictions == titanic["Survived"]]
print(type(predictions))
print(type(titanic["Survived"]))
pro = []
pro = titanic["Survived"].values
accuracy = sum(predictions[predictions == pro]) / len(predictions)

#accuracy = sum(predictions[predictions == titanic["Survived"]]) / len(predictions)

print(accuracy)
'''
# Initialize our algorithm LogisticRegression
'''
alg = LogisticRegression(random_state=1)
# Compute the accuracy score for all the cross validation folds.  (much simpler than what we did before!)
scores = cross_validation.cross_val_score(alg, titanic[predictors], titanic["Survived"], cv=3)
# Take the mean of the scores (because we have one for each fold)
print(scores.mean())
'''
alg = RandomForestClassifier(random_state=1, n_estimators=150, min_samples_split=4, min_samples_leaf=2)

# Compute the accuracy score for all the cross validation folds.  (much simpler than what we did before!)
scores = cross_validation.cross_val_score(alg, titanic[predictors], titanic["Survived"], cv=3)

# Take the mean of the scores (because we have one for each fold)
print(scores.mean())

