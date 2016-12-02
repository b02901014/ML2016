import numpy as np
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, TfidfTransformer, HashingVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import Normalizer
from sklearn.pipeline import make_pipeline
from sklearn.cluster import KMeans, MiniBatchKMeans, AgglomerativeClustering
from sklearn import feature_extraction
import matplotlib.pyplot as plt
import numpy as np
import sys
import csv
import string
from numpy import bincount

reload(sys)
sys.setdefaultencoding('utf8')
corpus = []
f = open(sys.argv[1]+'title_StackOverflow.txt', 'r')
lines = f.readlines()
for title in lines:
    corpus.append(title)

vectorizer = TfidfVectorizer(stop_words='english', max_df=0.5, min_df=2, max_features=None)
tfidf = vectorizer.fit_transform(corpus)
print tfidf.shape


svd = TruncatedSVD(n_components=20, n_iter=10, random_state=None)
normalizer = Normalizer(copy=False)
lsa = make_pipeline(svd, normalizer)
tfidf_svd = lsa.fit_transform(tfidf)

finalLabel = np.zeros((20000,1))

n_cluster = [100, 100, 100]
km = KMeans(n_clusters=n_cluster[0], init='k-means++', max_iter=100, n_init=10)
km.fit(tfidf_svd)
label = np.asarray(km.labels_)
center = km.cluster_centers_
clusterSize = np.bincount(label)
finalLabel = label
'''
km = KMeans(n_clusters=n_cluster[1], init='k-means++', max_iter=100, n_init=10)
km.fit(center)
label2 = np.asarray(km.labels_)
center2 = km.cluster_centers_
clusterSize2 = np.zeros((n_cluster[1],1))
for i in range(len(label2)):
    clusterSize2[label2[i]]+=clusterSize[i]
    index = ((label==i)*1).nonzero()
    finalLabel[index] = label2[i]
''' 
testFile = open(sys.argv[1]+'check_index.csv','r')
reader = csv.reader(testFile)
myData = np.asarray(list(reader))
myData = np.delete(myData,0,0)
myData = myData.astype("int32")

file_dir = sys.argv[2]
print "start testing"
with open(file_dir, 'wb') as ans:
    ans.write("ID,Ans\n")
    for j in range(myData.shape[0]):
        result = 0
        a = myData[j][1]
        b = myData[j][2]
        if finalLabel[a] == finalLabel[b]:
            result = 1
        ans.write(str(myData[j][0])+","+str(result)+"\n")



