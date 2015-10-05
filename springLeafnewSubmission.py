__author__ = 'pranavagarwal'

import pandas as pd
from sklearn import cross_validation
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif

# Columns with almost same value
mixCol = [8,9,10,11,12,18,19,20,21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 38, 39, 40, 41, 42, 43, 44, 45,
          73, 74, 98, 99, 100, 106, 107, 108, 156, 157, 158, 159, 166, 167, 168, 169, 176, 177, 178, 179, 180,
          181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 202, 205, 206, 207,
          208, 209, 210, 211, 212, 213, 214, 215, 216, 218, 219, 220, 221, 222, 223, 224, 225, 240, 371, 372, 373, 374,
          375, 376, 377, 378, 379, 380, 381, 382, 383, 384, 385, 386, 387, 388, 389, 390, 391, 392, 393, 394, 395,
          396, 397, 398, 399, 400, 401, 402, 403, 404, 405, 406, 427, 428, 429, 430, 431, 432, 433, 434, 435, 436,
          437, 438, 439, 440, 441, 442, 443, 444, 445, 446, 447, 448, 449, 450, 451, 452, 453, 454, 455, 456, 457,
          458, 459, 460, 461, 462, 463, 464, 465, 466, 467, 468, 469, 470, 471, 472, 473, 474, 475, 476, 477, 478,
          479, 480, 481, 492, 493, 494, 495, 496, 497, 498, 499, 500, 501, 502, 503, 504, 505, 506, 507, 508, 509,
          510, 511, 512, 513, 518, 519, 520, 521, 522, 523, 524, 525, 526, 527, 528, 529, 530, 840]

#Columns with Places as entries
placeCol = [200, 274, 237, 342]

#Columns with timestamps
dtCol = [75, 204, 217]

selectColumns = []
rmCol = mixCol+placeCol+dtCol
for i in range(1, 1935):
    if i not in rmCol:
        selectColumns.append(i)

cols = [str(n).zfill(4) for n in selectColumns]
strColName = ['VAR_' + strNum for strNum in cols]

# Read train.csv for the file with panda and apply nan if any value has empty space

train_data = pd.read_csv("/Users/pranavagarwal/Desktop/springleaf project/train.csv", usecols=strColName)
train_data = train_data.applymap(lambda x: np.nan if isinstance(x, basestring) and x.isspace() else x)
train_target = pd.read_csv("/Users/pranavagarwal/Desktop/springleaf project/train.csv", usecols=['target'])
test_id = pd.read_csv("/Users/pranavagarwal/Desktop/springleaf project/test.csv", usecols=['ID'])
test = pd.read_csv("/Users/pranavagarwal/Desktop/springleaf project/test.csv", usecols=strColName)

train_data.drop_duplicates()
test.drop_duplicates()

train_data.loc[train_data["VAR_0001"] == "H", "VAR_0001"] = 0
train_data.loc[train_data["VAR_0001"] == "R", "VAR_0001"] = 1
train_data.loc[train_data["VAR_0001"] == "Q", "VAR_0001"] = 2
train_data.loc[train_data["VAR_0005"] == "C", "VAR_0005"] = 0
train_data.loc[train_data["VAR_0005"] == "B", "VAR_0005"] = 1
train_data.loc[train_data["VAR_0005"] == "N", "VAR_0005"] = 2
train_data.loc[train_data["VAR_0005"] == "S", "VAR_0005"] = 3
train_data.loc[train_data["VAR_0226"] == "False", "VAR_0226"] = 0
train_data.loc[train_data["VAR_0226"] == "True", "VAR_0226"] = 1
train_data.loc[train_data["VAR_0230"] == "False", "VAR_0230"] = 0
train_data.loc[train_data["VAR_0230"] == "True", "VAR_0230"] = 1
train_data.loc[train_data["VAR_0232"] == "False", "VAR_0232"] = 0
train_data.loc[train_data["VAR_0232"] == "True", "VAR_0232"] = 1
train_data.loc[train_data["VAR_0232"] == "False", "VAR_0236"] = 0
train_data.loc[train_data["VAR_0232"] == "True", "VAR_0236"] = 1
train_data.loc[train_data["VAR_0232"] == "False", "VAR_0283"] = 0
train_data.loc[train_data["VAR_0232"] == "True", "VAR_0283"] = 1
train_data.loc[train_data["VAR_0305"] == "False", "VAR_0305"] = 0
train_data.loc[train_data["VAR_0305"] == "True", "VAR_0305"] = 1
train_data.loc[train_data["VAR_0325"] == "False", "VAR_0325"] = 0
train_data.loc[train_data["VAR_0325"] == "True", "VAR_0325"] = 1
train_data.loc[train_data["VAR_0352"] == "False", "VAR_0352"] = 0
train_data.loc[train_data["VAR_0352"] == "True", "VAR_0352"] = 1
train_data.loc[train_data["VAR_0353"] == "False", "VAR_0353"] = 0
train_data.loc[train_data["VAR_0353"] == "True", "VAR_0353"] = 1
train_data.loc[train_data["VAR_0354"] == "False", "VAR_0354"] = 0
train_data.loc[train_data["VAR_0354"] == "True", "VAR_0354"] = 1
train_data.loc[train_data["VAR_1934"] == "IAPS", "VAR_1934"] = 0
train_data.loc[train_data["VAR_1934"] == "RCC", "VAR_1934"] = 1
train_data.loc[train_data["VAR_1934"] == "BRANCH", "VAR_1934"] = 2
train_data.loc[train_data["VAR_1934"] == "MOBILE", "VAR_1934"] = 3
train_data.loc[train_data["VAR_1934"] == "CSC", "VAR_1934"] = 4



test.loc[test["VAR_0001"] == "H", "VAR_0001"] = 0
test.loc[test["VAR_0001"] == "R", "VAR_0001"] = 1
test.loc[test["VAR_0001"] == "Q", "VAR_0001"] = 2
test.loc[test["VAR_0005"] == "C", "VAR_0005"] = 0
test.loc[test["VAR_0005"] == "B", "VAR_0005"] = 1
test.loc[test["VAR_0005"] == "N", "VAR_0005"] = 2
test.loc[test["VAR_0005"] == "S", "VAR_0005"] = 3
test.loc[test["VAR_0226"] == "False", "VAR_0226"] = 0
test.loc[test["VAR_0226"] == "True", "VAR_0226"] = 1
test.loc[test["VAR_0230"] == "False", "VAR_0230"] = 0
test.loc[test["VAR_0230"] == "True", "VAR_0230"] = 1
test.loc[test["VAR_0232"] == "False", "VAR_0232"] = 0
test.loc[test["VAR_0232"] == "True", "VAR_0232"] = 1
test.loc[test["VAR_0232"] == "False", "VAR_0236"] = 0
test.loc[test["VAR_0232"] == "True", "VAR_0236"] = 1
test.loc[test["VAR_0232"] == "False", "VAR_0283"] = 0
test.loc[test["VAR_0232"] == "True", "VAR_0283"] = 1
test.loc[test["VAR_0305"] == "False", "VAR_0305"] = 0
test.loc[test["VAR_0305"] == "True", "VAR_0305"] = 1
test.loc[test["VAR_0325"] == "False", "VAR_0325"] = 0
test.loc[test["VAR_0325"] == "True", "VAR_0325"] = 1
test.loc[test["VAR_0352"] == "False", "VAR_0352"] = 0
test.loc[test["VAR_0352"] == "True", "VAR_0352"] = 1
test.loc[test["VAR_0353"] == "False", "VAR_0353"] = 0
test.loc[test["VAR_0353"] == "True", "VAR_0353"] = 1
test.loc[test["VAR_0354"] == "False", "VAR_0354"] = 0
test.loc[test["VAR_0354"] == "True", "VAR_0354"] = 1
test.loc[test["VAR_1934"] == "IAPS", "VAR_1934"] = 0
test.loc[test["VAR_1934"] == "RCC", "VAR_1934"] = 1
test.loc[test["VAR_1934"] == "BRANCH", "VAR_1934"] = 2
test.loc[test["VAR_1934"] == "MOBILE", "VAR_1934"] = 3
test.loc[test["VAR_1934"] == "CSC", "VAR_1934"] = 4

train_data = train_data.loc[:, train_data.apply(pd.Series.nunique) != 1]
test = test.loc[:, test.apply(pd.Series.nunique) != 1]

numeric = train_data._get_numeric_data()
numeric_test_set = test._get_numeric_data()

for col in numeric:
    numeric[col] = numeric[col].fillna(numeric[col].median())
    numeric_test_set[col] = numeric_test_set[col].fillna(numeric[col].median())

selector_predictors = numeric[:5000]
selector_target = train_target[:5000]
# make a list of the predictors column name, the number of list
colllist_predictors = numeric.columns.tolist()
no_of_Columns_in_predictors = len(colllist_predictors)

selector = SelectKBest(f_classif, k=5)
selector.fit(selector_predictors, selector_target)

# Get the raw p-values for each feature, and transform from p-values into scores

scores = -np.log10(selector.pvalues_)
topScorer = []
topPredictor = []

for x in range(0, no_of_Columns_in_predictors):
    if scores[x] > 25:
        topScorer.append(scores[x])
        topPredictor.append(colllist_predictors[x])

alg = RandomForestClassifier(random_state=1, n_estimators=150, min_samples_split=4, min_samples_leaf=2)

scores = cross_validation.cross_val_score(alg, numeric[topPredictor], train_target['target'], cv=3)

# Take the mean of the scores (because we have one for each fold)
print(scores.mean())

alg.fit(numeric[topPredictor], train_target['target'])

# Make predictions using the test set.
predictions = alg.predict(numeric_test_set[topPredictor])

submission = pd.DataFrame({
        "Id": test_id['ID'],
        "target": predictions})

submission.to_csv("springleaf_With_String_Random_Forest.csv", index=False)

'''

plt.bar(range(len(topPredictor)), topScorer)
plt.xticks(range(len(topPredictor)), topPredictor, rotation='vertical')
plt.show()
for col in train_data:
    print(col + str(train_data[col].unique()))

'''




