#!--encoding=utf-8
'''
Created on Apr 17, 2017

@author: binlin
'''
from __future__ import print_function
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer,\
    TfidfTransformer
from sklearn.feature_extraction.text import HashingVectorizer
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans,MiniBatchKMeans
from prohandle import *
from cPickle import load
from mailcap import show
import codecs
from sklearn.preprocessing.tests.test_data import n_features


def loadDataset(srcfile=PRE_OUTPUT_FILE):
    "导入文本数据集,此处把每一行当成数据集中的一个条目，可以根据需要来设定条目"
    f = codecs.open(srcfile,'r','utf-8')
    dataset = []
    for line in f.readlines():
        dataset.append(line)
    f.close()
    return dataset

def transform(dataset,n_features=10):
    "将数据集中的每个条目转换成向量，向量的维度数量由特征点数量决定，即向量维度＝特征点数量"
    #===========================================================================
    # for item in vectorizer.vocabulary_:
    #     print(item)
    # print(X.todense())
    # vectorizer指的是特征点数（此处缺省为10维向量）的词向量，这些词汇是10个高频词汇“有人，指示，网络，中国，系统，无法，打不开，央视，重要，习近平”
    # X 为数据集中每个数据条目的向量（已归一化为1)
    #===========================================================================
    #max_df有些词语，他们出的频率太高了（如果一个词语在所有文档出现了，还需要文本分类吗？不需要），我们设定一个阀值，如果是float＝0.5表示，如果50%以上的文档出现，就不划为特证词，改为临时停用词
    #如果是整数10，表现这个词如果在数据集超过10的文本都出现了，那么我们把它列为临时停用词
    #max_df反之，如果有些词语，出现频率太低了，我们也把它作为停用词 
    #TfidfVectorizer＝ TfidTransformer + CountVectorizer 
    vectorizer = TfidfVectorizer(max_df=0.5,max_features=n_features,min_df=1,use_idf=True)
    #获取词频，特征向量。Ｘ.vacabulary_ 和 X.vacabulary 是一样的。
    X = vectorizer.fit_transform(dataset)
    return X,vectorizer

def train(X,vectorizer, true_k=10,minibatch=False,showLable =False):
    "输入：数据集的向量，多少簇，输出：Ｋ族分类"
    if minibatch:
        km = MiniBatchKMeans(n_clusters=true_k,init='k-means++',n_init=1,
                             init_size=1000,batch_size=1000,verbose=False)
    else:
        km = KMeans(n_clusters=true_k,init='k-means++',max_iter=300,n_init=1,verbose=False)
    km.fit(X)
    
    if showLable:
        print("Top terms per cluster:")
        order_centroids= km.cluster_centers_.argsort()[:,::-1]
        terms =vectorizer.get_feature_names()
        print(vectorizer.get_stop_words())
        for i in range(true_k):
            print("Cluster %d:" % i ,end='')
            for ind in order_centroids[i,:10]:
                print(' %s' % terms[ind],end='')
            print()
    result = list(km.predict(X))
    print('Cluster distribution:')
    print(dict([(i,result.count(i)) for i in result]))
    return -km.score(X)


def test():
    "测试，寻找最优特征点"
    dataset = loadDataset()
    print("%d documents" % len(dataset))
    X, vectorizer = transform(dataset, n_features=100)
    true_ks =[]
    scores =[]
    for i in xrange(3,28,1):
        score = train(X,vectorizer,true_k=i)/len(dataset)
        print(i,score)
        true_ks.append(i)
        scores.append(score)
    plt.figure(figsize=(16,8))
    plt.plot(true_ks, scores, label="error",color="red",linewidth=1)
    plt.xlabel("n_features")
    plt.ylabel("error")
    plt.legend()
    plt.show()
    
def out():
    "输出最终结果"
    dataset = loadDataset()
    X, vectorizer = transform(dataset, n_features=50)
    for item in vectorizer.vocabulary_:
        print(item)
    print(X.todense())
    score = train(X,vectorizer,minibatch=False, true_k=4,showLable=True)/len(dataset)
    print(score)

def testtfidf():
    "测试如何将一段文本转化为数据向量，以及词频等"
    #将文本中的词语转换为词频矩阵，矩阵元素a[i][j] 表示j词在i类文本下的词频 
    vectorizer=CountVectorizer() #创建词袋数据结构
    #创建hash向量词袋
    # vectorizer = HashingVectorizer(stop_words =stpwrdlst,n_features = 10000) #设置停用词词表,设置最大维度10000
 
    #统计每个词语的tf-idf权值
    #TfidTransformer(norm=u'l2',use_idf=True,smooth_idf=True,sublinear_tf=)
    #norm=u'l1',u'l2',None
    transformer = TfidfTransformer()
    #第一个fit_transform是计算tf-idf，第二个fit_transform是将文本转为词频矩阵
    tfidf = transformer.fit_transform(vectorizer.fit_transform(loadDataset()))  
    wordlist = vectorizer.get_feature_names()#获取词袋模型中的所有词  
    # tf-idf矩阵 元素a[i][j]表示j词在i类文本中的tf-idf权重
    weightlist = tfidf.toarray()  
    #打印每类文本的tf-idf词语权重，第一个for遍历所有文本，第二个for便利某一类文本下的词语权重
    for i in range(len(weightlist)):  
         print("－－－－－－这里输出第"+str(i)+"类文本的词语tf-idf权重--------") 
         for j in range(len(wordlist)):  
             print(str(j)+" "+wordlist[j].encode('utf-8')+" "+ str(weightlist[i][j]))
    

if __name__ == '__main__':
    "主函数入口"
    #pass
    #print(line.encode("utf-8") for line in loadDataset())
    #===========================================================================
    # d = loadDataset()
    # print(type(d))
    # print(len(d))
    # for item in d: 
    #     print(item)
    #===========================================================================
    #out()
    #test()
    testtfidf()
    
    
    
    
    
    
    
    