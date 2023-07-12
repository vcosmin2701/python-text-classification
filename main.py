import pandas as pd
from pandas import DataFrame
import matplotlib.pyplot as plt

data = pd.read_csv('https://raw.githubusercontent.com/mohitgupta-omg/Kaggle-SMS-Spam-Collection-Dataset-/master/spam.csv', encoding='latin-1')

data.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis=1, inplace=True)

data.columns = ['label', 'text']

import nltk
import re
# nltk.download('all')

text = list(data['text'])

# preprocessing loop
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

corpus = []

for i in range(len(text)):
    r = re.sub('^a-zA-Z', ' ', text[i])
    r = r.lower()
    r = r.split()
    r = [word for word in r if word not in stopwords.words('english')]
    r = [lemmatizer.lemmatize(word) for word in r]
    r = ' '.join(r)
    corpus.append(r)

data['text'] = corpus

df = DataFrame(data.head())

X = data['text']
y = data['label']

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=123)

print('Training Data: ', X_train.shape)
print('Testing Data: ', X_test.shape)

# Bag of Words model

from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer()

X_train_cv = cv.fit_transform(X_train)
print(X_train_cv.shape)

from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()
lr.fit(X_train_cv, y_train)

X_test_cv = cv.transform(X_test)

# generate predictions
predictions = lr.predict(X_test_cv)
print(predictions)


#confusion matrix
from sklearn import metrics
df1 = pd.DataFrame(metrics.confusion_matrix(y_test, predictions), index=['ham', 'spam'], columns=['ham', 'spam'])

with pd.ExcelWriter('output.xlsx') as writer:
    df.to_excel(writer, sheet_name='preprocessing')
    df1.to_excel(writer, sheet_name='confusionmatrix')
