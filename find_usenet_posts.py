#!/usr/bin/python

#######################################
# module: find_usenet_posts.py
# description: finding usenet posts to user queries
# Robert Epstein and A01092594

# sample run:
#$ python find_usenet_posts.py -feat_mat usenet_feat_mat.pck -query 'is fuel injector cleaning necessary?' -top_n 5 -vectorizer usenet_vectorizer.pck > fuel_injector_query.txt
# 
#
# bugs to vladimir dot kulyukin at usu dot edu
#######################################

import os
import sys
import sklearn.datasets
import scipy as sp
import cPickle as pickle
from sklearn.feature_extraction.text import CountVectorizer
import argparse
import nltk.stem

## define the stemmer
english_stemmer = nltk.stem.SnowballStemmer('english')
class StemmedCountVectorizer(CountVectorizer):
    def build_analyzer(self):
        analyzer = super(StemmedCountVectorizer, self).build_analyzer()
        return lambda doc: (english_stemmer.stem(w) for w in analyzer(doc))

## define two distances
def euclid_dist(v1, v2):
    diff = v1 - v2
    return sp.linalg.norm(diff.toarray())

def norm_euclid_dist(v1, v2):
    v1_normalized = v1/sp.linalg.norm(v1.toarray())
    v2_normalized = v2/sp.linalg.norm(v2.toarray())
    diff = v1_normalized - v2_normalized
    return sp.linalg.norm(diff.toarray())

## load the texts of usenet newsgroups
print('Loading Usenet data')
usenet_data = sklearn.datasets.fetch_20newsgroups()
print('Usenet data loaded...')

## find the closest posts
def find_top_n_closest_posts(vectorizer, user_query, doc_feat_mat, dist_fun, top_n=10):
    #vectorize user_query
    #dist_fun(vectorized(user_query),doc_feat_mat)
    vectored = vectorizer.fit_transform([user_query]).getrow(0).toarray()
    answer = []
    for x in doc_feat_mat:
        temp = dist_fun(vectored, doc_feat_mat.getrow(x).toarray())
        answer.append([x,temp])
    return sorted(answer, key=lambda x: x[1],reverse=True)[:topn]
    pass

## command line arguments
if __name__ == '__main__':
  ap = argparse.ArgumentParser()
  ap.add_argument('-feat_mat', '--feat_mat', required=True, help='pickle index file')
  ap.add_argument('-query', '--query', required=True, help='user query text')
  ap.add_argument('-vectorizer', '--vectorizer', required=True, help='pickle vectorizer file')
  ap.add_argument('-top_n', '--top_n', required=True, help='top n results to retrieve', type=int)
  args = vars(ap.parse_args())
  usenet_data_feat_mat = None
  stemmed_vectorizer = None
  ## unpickle the feature matrix
  with open(args['feat_mat'], 'rb') as feat_mat_pck:
      usenet_data_feat_mat = pickle.load(feat_mat_pck)
  ## unpickle the vectorizer
  with open(args['vectorizer'], 'rb') as vectorizer_pck:
      stemmed_vectorizer = pickle.load(vectorizer_pck)
  print('#num_posts: %d, #features: %d' % usenet_data_feat_mat.shape)
  ## find the top n closest posts that print them.
  print stemmed_vectorizer.get_feature_names()
  
  matches = find_top_n_closest_posts(stemmed_vectorizer,
                                         args['query'],
                                         usenet_data_feat_mat,
                                         norm_euclid_dist,
                                         top_n=args['top_n'])
#  for i, d in matches:
#      print('Post #%d, matching distance=%f' % (i, d))
#      print('Post text:')
#      print(usenet_data.data[i])
#      print('--------------------')
