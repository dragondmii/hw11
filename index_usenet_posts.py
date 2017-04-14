#!/usr/bin/python

##################################
# module: index_usenet_posts.py
# description: indexing usenet posts
#
# sample run:
# $ python index_usenet_posts.py -feat_mat usenet_feat_mat.pck -vectorizer usenet_vectorizer.pck
# number of posts: 11314, number of features: 110992
# indexing finished
#
# bugs to vladimir dot kulyukin at usu dot edu
##################################

import os
import sys
import sklearn.datasets
import scipy as sp
from sklearn.feature_extraction.text import CountVectorizer
import cPickle as pickle
import argparse

## These are the groups
## ['alt.atheism', 'comp.graphics', 'comp.os.ms-windows.misc',
## 'comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware', 'comp.windows.x',
## 'misc.forsale', 'rec.autos', 'rec.motorcycles', 'rec.sport.baseball',
## 'rec.sport.hockey', 'sci.crypt', 'sci.electronics', 'sci.med',
## 'sci.space', 'soc.religion.christian', 'talk.politics.guns',
## 'talk.politics.mideast', 'talk.politics.misc', 'talk.religion.misc']

## load the usenet data
usenet_data = sklearn.datasets.fetch_20newsgroups()

import nltk.stem
english_stemmer = nltk.stem.SnowballStemmer('english')
class StemmedCountVectorizer(CountVectorizer):
    def build_analyzer(self):
        analyzer = super(StemmedCountVectorizer, self).build_analyzer()
        return lambda doc: (english_stemmer.stem(w) for w in analyzer(doc))

stemmed_vectorizer = StemmedCountVectorizer(min_df=1, stop_words='english')
usenet_data_feat_mat = stemmed_vectorizer.fit_transform(usenet_data.data)
num_samples, num_features = usenet_data_feat_mat.shape
print('#number of posts: %d, number of features: %d' % (num_samples, num_features))

if __name__ == '__main__':
  ap = argparse.ArgumentParser()
  ap.add_argument('-feat_mat', '--feat_mat', required=True, help='pickle feat mat file')
  ap.add_argument('-vectorizer', '--vectorizer', required=True, help='pickle vectorizer file')
  args = vars(ap.parse_args())
  with open(args['feat_mat'], 'wb') as feat_mat_pck:
      pickle.dump(usenet_data_feat_mat , feat_mat_pck)
  with open(args['vectorizer'], 'wb') as vectorizer_pck:
      pickle.dump(stemmed_vectorizer, vectorizer_pck)
  print('indexing finished')
