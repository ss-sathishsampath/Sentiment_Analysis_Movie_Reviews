# -*- coding: utf-8 -*-
"""
Sentiment Analysis-Movie Reviews using NLTK

@author: Sathish Sampath(ss.sathishsampath@gmail.com)
Developed as part of  Microsoft's NLP MOOC(https://www.edx.org/course/natural-language-processing-nlp)


"""

# movie reviews / sentiment analysis 
import nltk
from nltk.corpus import movie_reviews as reviews
import random

# Input Documents
docs = [(list(reviews.words(id)), cat)  for cat in reviews.categories() for id in reviews.fileids(cat)]

# Shuffle the input
random.shuffle(docs)


fd = nltk.FreqDist(word.lower() for word in reviews.words())
topKeys = [ key for (key,value) in fd.most_common(2000)]


def review_features(doc):
    docSet = set(doc)
    features = {}
    
    for word in topKeys:
        features[word] = (word in docSet)
        
    return features


data = [(review_features(doc), label) for (doc,label) in docs]

dataCount = len(data)
trainCount = int(.8*dataCount)

trainData = data[:trainCount]
testData = data[trainCount:]
bayes2 = nltk.NaiveBayesClassifier.train(trainData)

print("train accuracy=", nltk.classify.accuracy(bayes2, trainData))
print("test accuracy=", nltk.classify.accuracy(bayes2, testData))

# Show Best Features
bayes2.show_most_informative_features(20)