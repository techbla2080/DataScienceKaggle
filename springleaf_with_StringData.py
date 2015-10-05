__author__ = 'pranavagarwal'

import pandas as pd
import numpy as np

# Columns with almost same value
mixCol = [8, 9, 10, 11, 12, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 38, 39, 40, 41, 42, 43, 44, 45,
          73, 74, 98, 99, 100, 106, 107, 108, 156, 157, 158, 159, 166, 167, 168, 169, 176, 177, 178, 179, 180,
          181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 202, 205, 206, 207,
          208, 209, 210, 211, 212, 213, 214, 215, 216, 218, 219, 220, 221, 222, 223, 224, 225, 240, 371, 372, 373, 374,
          375, 376, 377, 378, 379, 380, 381, 382, 383, 384, 385, 386, 387, 388, 389, 390, 391, 392, 393, 394, 395,
          396, 397, 398, 399, 400, 401, 402, 403, 404, 405, 406, 427, 428, 429, 430, 431, 432, 433, 434, 435, 436,
          437, 438, 439, 440, 441, 442, 443, 444, 445, 446, 447, 448, 449, 450, 451, 452, 453, 454, 455, 456, 457,
          458, 459, 460, 461, 462, 463, 464, 465, 466, 467, 468, 469, 470, 471, 472, 473, 474, 475, 476, 477, 478,
          479, 480, 481, 492, 493, 494, 495, 496, 497, 498, 499, 500, 501, 502, 503, 504, 505, 506, 507, 508, 509,
          510, 511, 512, 513, 518, 519, 520, 521, 522, 523, 524, 525, 526, 527, 528, 529, 530, 840]

# Columns with logical datatype
alphaCol = [226, 230, 232, 236, 283, 305, 325, 352, 353, 354, 1934]

# Columns with Places as entries
placeCol = [200, 274, 342]

# Columns with timestamps
dtCol = [75, 204, 217]

selectColumns = []
rmCol = mixCol+alphaCol+placeCol+dtCol
for i in range(1, 1935):
    if i not in rmCol:
        selectColumns.append(i)

cols = [str(n).zfill(4) for n in selectColumns]
strColName = ['VAR_' + strNum for strNum in cols]

# Read train.csv for the file with panda and apply nan if any value has empty space

train_data = pd.read_csv("/Users/pranavagarwal/Desktop/springleaf project/train.csv", usecols=strColName)
train_data = train_data.applymap(lambda x: np.nan if isinstance(x, basestring) and x.isspace() else x)
# train_target = pd.read_csv("/Users/pranavagarwal/Desktop/springleaf project/train.csv", usecols=['target'], nrows=5000)


# drop duplicate data and also columns having a single value

train_data.drop_duplicates()
train_data = train_data.loc[:, train_data.apply(pd.Series.nunique) != 1]

# numeric = train_data._get_numeric_data()
object_df = train_data.loc[:, train_data.dtypes == object]

object_df["VAR_0237"] = object_df["VAR_0237"].fillna("CA")

object_df.loc[object_df["VAR_0001"] == "H", "VAR_0001"] = 0
object_df.loc[object_df["VAR_0001"] == "R", "VAR_0001"] = 1
object_df.loc[object_df["VAR_0001"] == "Q", "VAR_0001"] = 2
object_df.loc[object_df["VAR_0005"] == "C", "VAR_0005"] = 0
object_df.loc[object_df["VAR_0005"] == "B", "VAR_0005"] = 1
object_df.loc[object_df["VAR_0005"] == "N", "VAR_0005"] = 2
object_df.loc[object_df["VAR_0005"] == "S", "VAR_0005"] = 3

print(object_df.describe())

'''
print("train data: " + str(len(train_data.columns)))
print("numeric data : " + str(len(numeric.columns)))
print("String data : " + str(len(object_df.columns)))
'''