import re
import numpy as np
from collections import Counter
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

def sent_distr(df):
    neu= df[df.sentiment=="NEU"].shape[0]
    neg= df[df.sentiment=="NEG"].shape[0]
    pos= df[df.sentiment=="POS"].shape[0]
    
    plt.bar(("NEU","NEG","POS"),(neu,neg,pos))
    plt.title("Sentiment distribution")
    plt.show()
        
def retKey(tup):
    return tup[1]

def word_freq(df):
    allWords=[x.split() for x in df["text"]]
    allWords=sum(allWords,[])
    
    dictWords=dict(Counter(allWords).most_common(20))
    comWords=list(dictWords.items())
    comWords.sort(reverse=True,key=retKey)
    plt.bar([x[0] for x in comWords],[x[1] for x in comWords])
    plt.title("Most Common words")
    plt.show()
    
def word_freq(df,sent=None):
    if sent==None:
        allWords=[x.split() for x in df["text"]]
    else:
        allWords=[df["text"].iloc[i].split() for i in range(df.shape[0]) if df["sentiment"].iloc[i]==sent]
    allWords=sum(allWords,[])
    
    dictWords=dict(Counter(allWords).most_common(20))
    comWords=list(dictWords.items())
    comWords.sort(reverse=True,key=retKey)
    plt.bar([x[0] for x in comWords],[x[1] for x in comWords])
    plt.title("Most Common words in",sent,"tweet")
    plt.show()
    
    
def comp_astra_mrna(df):
    dfAstra=df[df.text.str.contains("astrazeneca")]
    dfMrna=df[df.text.str.contains("moderna|pfizer|biontech")]
    
    neuAstra= dfAstra[dfAstra.sentiment=="NEU"].shape[0]
    neuMrna= dfMrna[dfMrna.sentiment=="NEU"].shape[0]
    negAstra= dfAstra[dfAstra.sentiment=="NEG"].shape[0]
    negMrna= dfMrna[dfMrna.sentiment=="NEG"].shape[0]
    posAstra= dfAstra[dfAstra.sentiment=="POS"].shape[0]
    posMrna= dfMrna[dfMrna.sentiment=="POS"].shape[0]
    
    allAstra=neuAstra+negAstra+posAstra
    allMrna=neuMrna+negMrna+posMrna
    
    index = np.arange(3)
    
    fig, ax = plt.subplots()
    astra=ax.bar(index,100*np.array([neuAstra,negAstra,posAstra])/allAstra,0.3,label="Astrazeneca")
    mrna=ax.bar(index+0.3,100*np.array([neuMrna,negMrna,posMrna])/allMrna,0.3,label="Mrna(Pfizer and Moderna)")
    ax.set_xticks(index + 0.3 / 2)
    ax.set_xticklabels(["NEU","NEG","POS"])
    ax.set_title("Sentiment percentage distribution by vaccine")
    ax.legend()
    plt.show()

def month_distr(df):
    copyDf=df.copy()
    copyDf = copyDf.astype({"date": "datetime64"})
    copyDf=copyDf.sort_values("date")
    copyDf=copyDf.groupby(
        [copyDf['date'].dt.year.rename('year'),copyDf['date'].dt.month_name().rename('month')],sort=False
        ).size().plot(kind="bar")
    plt.title("Number of tweets by month")
    plt.tight_layout()
    plt.show()
    
def percent_positive_by_month(df):
    copyDf=df.copy()
    copyDf = copyDf.astype({"date": "datetime64"})
    copyDf=copyDf.sort_values("date")
    (copyDf[copyDf["sentiment"]=="NEG"].groupby(
        [copyDf['date'].dt.year.rename('year'),copyDf['date'].dt.month_name().rename('month')],sort=False
        ).size()/copyDf.groupby(
        [copyDf['date'].dt.year.rename('year'),copyDf['date'].dt.month_name().rename('month')],sort=False
        ).size()).plot(kind="bar")
    plt.title("Pecrentage of tweets that were negative by month")
    plt.tight_layout()
    plt.show()
    
def percent_retweets_by_sent(df):
    
    neu_perc= df[(df.sentiment=="NEU") & (df.retweets>0)].shape[0]/df[df.sentiment=="NEU"].shape[0]
    neg_perc= df[(df.sentiment=="NEG") & (df.retweets>0)].shape[0]/df[df.sentiment=="NEG"].shape[0]
    pos_perc= df[(df.sentiment=="POS") & (df.retweets>0)].shape[0]/df[df.sentiment=="POS"].shape[0]
    
    plt.bar(("NEU","NEG","POS"),(100*neu_perc,100*neg_perc,100*pos_perc))
    plt.ylim([0,100])
    plt.title("Pecrentage of tweets that were retweeted by sentiment")
    plt.show()

def analize_data(df):
    sent_distr(df)
    word_freq(df)
    comp_astra_mrna(df)
    month_distr(df)
    percent_positive_by_month(df)
    percent_retweets_by_sent(df)
    
    return df
    
    

