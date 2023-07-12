import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('https://raw.githubusercontent.com/mohitgupta-omg/Kaggle-SMS-Spam-Collection-Dataset-/master/spam.csv', encoding='latin-1')

data.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis=1, inplace=True)

data.columns = ['label', 'text']

# check for missing values
print(data.isna().sum())

# check data shape => tuple with dim of matrix
print(data.shape)

#check target balance
data['label'].value_counts(normalize=True).plot.bar()
plt.show()
