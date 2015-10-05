__author__ = 'pranavagarwal'
import matplotlib.pyplot as plt
import pandas as pd

'''
month = [1, 1, 2, 2, 4, 5, 5, 7, 8, 10, 10, 11, 12]
temperature = [32, 15, 40, 35, 50, 55, 52, 80, 85, 60, 57, 45, 35]
# We tell matplotlib to draw the scatterplot with this command.
plt.scatter(month, temperature)

plt.show()
'''

train = pd.read_csv("/Users/pranavagarwal/Desktop/springleaf project/train.csv", nrows=500)
object_df = train.loc[:, train.dtypes == object]
print(object_df.describe())