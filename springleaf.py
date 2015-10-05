from numpy.distutils.system_info import numarray_info

__author__ = 'pranavagarwal'
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import cross_validation
from sklearn.ensemble import RandomForestClassifier
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

#Columns with logical datatype
alphaCol = [283, 305, 325, 352, 353, 354, 1934]

#Columns with Places as entries
placeCol = [200, 274, 342]

#Columns with timestamps
dtCol = [75, 204, 217]

selectColumns = []
rmCol = mixCol+alphaCol+placeCol+dtCol
for i in range(1, 1935):
    if i not in rmCol:
        selectColumns.append(i)

cols = [str(n).zfill(4) for n in selectColumns]
strColName = ['VAR_' + strNum for strNum in cols]


data = pd.read_csv("/Users/pranavagarwal/Desktop/springleaf project/train.csv", usecols=strColName, nrows=5000)
data = data.applymap(lambda x: np.nan if isinstance(x, basestring) and x.isspace() else x)
target = pd.read_csv("/Users/pranavagarwal/Desktop/springleaf project/train.csv", usecols=['target'], nrows=5000)
#Train_data = data[:5000]
#Train_target = target[:5000]
# print(data.dtypes)
# data.loc[:, data.dtypes != object]
# print(len(data.columns))
data.drop_duplicates()
data = data.loc[:, data.apply(pd.Series.nunique) != 1]

numeric = data._get_numeric_data()

for col in numeric:
    numeric[col] = numeric[col].fillna(numeric[col].median())

colllist = numeric.columns.tolist()

predictors = numeric[colllist[0:1694]]
colllist_predictors = predictors.columns.tolist()
no_of_Columns_in_predictors = len(colllist_predictors)


selector = SelectKBest(f_classif, k=5)
selector.fit(predictors, target)

# Get the raw p-values for each feature, and transform from p-values into scores

scores = -np.log10(selector.pvalues_)
topScorer = []
topPredictor = []
for x in range(0, no_of_Columns_in_predictors):
    if scores[x] > 10:
        topScorer.append(scores[x])
        topPredictor.append(colllist_predictors[x])

alg = RandomForestClassifier(random_state=1, n_estimators=150, min_samples_split=4, min_samples_leaf=2)

# Compute the accuracy score for all the cross validation folds.  (much simpler than what we did before!)
scores = cross_validation.cross_val_score(alg, predictors[topPredictor], target['target'], cv=3)

# Take the mean of the scores (because we have one for each fold)
print(scores.mean())


'''

scores = -np.log10(selector.pvalues_)

plt.bar(range(no_of_Columns_in_predictors), scores)
plt.xticks(range(no_of_Columns_in_predictors), colllist_predictors, rotation='vertical')
plt.show()

for col in predictors:
    print(col + " : " + str(predictors[col].unique()) + " median: " + str(predictors[col].median()))

for col in numeric:
    print(col + " : " + str(numeric[col].unique()) + " Count:  " + str(numeric[col].count()))
#print(len(data.columns))
#print(data.columns.values)
#print(len(numeric.columns))
#print(numeric.columns.values)
#print(numeric.dtypes)
for col in numeric:
    print(col + " Count:  " + str(numeric[col].count()))

charecterdata = data._get_object_data()

print(len(charecterdata.columns))
print(charecterdata.columns.values)
data.loc[:, (data != data.ix[0]).any()]
data.drop_duplicates(cols='Id', take_last=True)
print(len(data.columns))
for col in data:
    if data[col].dtypes == object:
        #data[col].fillna(data[col].median())
        print (str(data[col].count()) + " :" + str(data[col].mode()))


for col in data:
    if col.dtypes == object:
        print(col.unique())

for col in data:
    print data[col].unique()

data["VAR_0001"].hist(by=data["target"])
plt.xlabel('var_0001')
plt.ylabel('Target')
plt.show()
'''