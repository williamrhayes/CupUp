#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pickle
import pandas as pd
import numpy as np
import re
import os
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer

class FlavorModel:
    
    def __init__(self, count_vectorizer, xgb_file):
        # Load in our count vectorizer and model objects 
        with open(count_vectorizer, "rb") as cv_obj:
            self.cv  = pickle.load(cv_obj)
            
        with open(xgb_file, "rb") as xgb_obj:
            self.xgb_model = pickle.load(xgb_obj)
            
    def predict(self, flavor_str):
        # Clean and process the flavor string
        flavors = re.sub('[^a-zA-Z]', ' ', flavor_str)
        flavors = flavors.lower()
        flavors = flavors.split()
        ps = PorterStemmer()
        all_stopwords = stopwords.words('english')
        flavors = [ps.stem(word) for word in flavors if not word in set(all_stopwords)]
        flavors = ' '.join(flavors)
        # Use the Count Vectorizer to create the
        # Bag of Words
        X = self.cv.transform([flavors]).toarray()
        # Predict score based on user input
        return self.xgb_model.predict(X)[0]

