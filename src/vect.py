import gensim
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer



def bow_vect(df_train,df_test):
    bow_vectorizer=CountVectorizer(min_df=1,max_features=500,stop_words="english")
    
    xtrain_bow=bow_vectorizer.fit_transform(df_train["text"])
    xtest_bow=bow_vectorizer.transform(df_test["text"])

    pickle.dump((xtrain_bow.toarray(),xtest_bow.toarray()),open("/bow_vect.pkl",'wb'))
    
    #print(xtrain_bow)
    #print(xtrain_bow.shape)
    
    #print(xtest_bow)
    #print(xtest_bow.shape)
    
    #print(bow_vectorizer.get_feature_names())
    
    return xtrain_bow.toarray(),xtest_bow.toarray()
    
def tfidf_vect(df_train,df_test):
    tfidf_vectorizer=TfidfVectorizer(max_features=500,stop_words="english")
    
    xtrain_tfidf=tfidf_vectorizer.fit_transform(df_train["text"])
    xtest_tfidf=tfidf_vectorizer.transform(df_test["text"])
    
    pickle.dump((xtrain_tfidf.toarray(),xtest_tfidf.toarray()),open("tfidf_vect.pkl",'wb'))
    
    #print(xtrain_tfidf)
    #print(xtrain_tfidf.shape)
    
    #print(xtest_tfidf)
    #print(xtest_tfidf.shape)
    
    #print(tfidf_vectorizer.get_feature_names())
    
    return xtrain_tfidf.toarray(),xtest_tfidf.toarray()

def word2v_vect(df_train,df_test):
    tok_tweet=df_train["text"].apply(strsplit)
    model_w2v = gensim.models.Word2Vec(tok_tweet,min_count=2,sg = 1,negative = 10,workers= 2,seed=37,vector_size=200)
    model_w2v.train(tok_tweet, total_examples= df.shape[0], epochs=20)
    
    xtrain_w2v=df_train["text"].apply(avg_vect,args=(model_w2v,))
    xtest_w2v=df_test["text"].apply(avg_vect,args=(model_w2v,))

    xtrain_array=np.array([xtrain_w2v.values[i] for i in range(xtrain_w2v.shape[0])],"float64")
    xtest_array=np.array([xtest_w2v.values[i] for i in range(xtest_w2v.shape[0])],"float64")

    pickle.dump((xtrain_array,xtest_array),open("w2v_vect.pkl",'wb'))
    
    return

    #print(model_w2v.wv.most_similar(positive="pfizer"))

def avg_vect(str,model):
    if str=="":
        return np.random.uniform(-10,10,(200,)).astype("float64")

    str_list=str.split()
    vect_list=[model.wv[str_i] if str_i in model.wv
                    else np.random.uniform(-10,10,(200,)).astype("float64")
                for str_i in str_list]
                
    return np.array(sum(vect_list)/len(vect_list),"float64")


def strsplit(str):
    return str.split()
