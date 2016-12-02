import numpy as np
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, TfidfTransformer, HashingVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import Normalizer
from sklearn.pipeline import make_pipeline
from sklearn.cluster import KMeans, MiniBatchKMeans, AgglomerativeClustering
from sklearn.metrics.pairwise import euclidean_distances, cosine_similarity
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
sim = cosine_similarity(tfidf_svd, tfidf_svd)

testFile = open(sys.argv[1]+'check_index.csv', 'r')
reader = csv.reader(testFile)
myData = np.asarray(list(reader))
myData = np.delete(myData,0,0)
myData = myData.astype("int32")
print myData.shape

check = np.zeros((20000,1))
cluster = []
count = 0 

with open('aaa.csv', 'wb') as ans:
    ans.write("ID,size of cluster\n")
    for j in range (20000):
        if check[j,0]==0: 
            count +=1
            index = np.argwhere(sim[j,:]>=0.9)
            check[j,0] = count
            check[index] = count
print count
means = np.zeros((count,20))

for i in range(count):
    index = tfidf_svd[np.argwhere(check == i+1)]
    index = np.asarray(index)
    temp = np.zeros((1,20))
    for j in range(len(index)):
        temp = np.add(temp, index[j])
    means[i,:] = (temp[0,:]/len(index))
means = np.asarray(means)
means = np.reshape(means, (means.shape[0],20))
print count
print means.shape
print means


km = KMeans(n_clusters = 20, init='k-means++', max_iter=100, n_init=10)
km.fit(means)
label = np.asarray(km.labels_)
size =np.bincount(label)
print size



file_dir = sys.argv[2]
print "start testing"
with open(file_dir, 'wb') as ans:
    ans.write("ID,Ans\n")
    for j in range(myData.shape[0]):
        result = 0
        if sim[myData[j][1],myData[j][2]] >= 0.9:
            result = 1

        ans.write(str(myData[j][0])+","+str(result)+"\n")


