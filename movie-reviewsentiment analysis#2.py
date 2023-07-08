import pickle
import itertools
from random import shuffle


import nltk
from nltk.collocations import BigramCollocationFinder
from nltk.metrics import BigramAssocMeasures
from nltk.probability import FreqDist, ConditionalFreqDist

import sklearn
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.metrics import accuracy_score
import pandas as pd 
data = pd.read_csv("Process_train.csv")
data.head()
train_data = pd.read_csv('Train_reviews.csv')
train_data.info()
train_data = train_data.drop('Unnamed: 0',axis=1)
train_data.info()
a = train_data.sample(10)
j=1
for i,k in zip(a.Text_Review,a.Sentiment):
    print(f'{j}) {(i)} - {k}')
    j+=1
def cln_txt(text):
    text = text[2:-2]
    text = re.sub('<br />','',text)
    text = text.lower()
    return text
train_data.Text_Review = train_data['Text_Review'].apply(cln_txt)
a = train_data.sample(10)
print(a)