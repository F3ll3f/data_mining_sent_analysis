import matplotlib.pyplot as plt
from gensim.test.utils import common_corpus
from gensim.corpora.dictionary import Dictionary
from gensim.models.ldamodel import *
from gensim.models import CoherenceModel
#!pip install pyLDAvis
import pyLDAvis.gensim_models as gensimvis
import pyLDAvis
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

def lda(df,numTopics):
    tweets=[df["text"][i].split() for i in range(df.shape[0])]
    tweets_dictionary=Dictionary(tweets)
    tweets_corpus=[tweets_dictionary.doc2bow(tweet) for tweet in tweets]
    lda_model=LdaModel(tweets_corpus, num_topics=numTopics,id2word=tweets_dictionary)
    
    return lda_model,tweets_dictionary,tweets_corpus


def eval_lda(df):
    tweets=[df["text"][i].split() for i in range(df.shape[0])]
    tweets_dictionary=Dictionary(tweets)
    tweets_corpus=[tweets_dictionary.doc2bow(tweet) for tweet in tweets]
    
    coherences=[]
    
    for i in range(5,30):
        lda_model=LdaModel(tweets_corpus, num_topics=i,id2word=tweets_dictionary)
        coh_lda_model=CoherenceModel(model=lda_model,dictionary=tweets_dictionary,texts=tweets,coherence="c_v")
        coherence=coh_lda_model.get_coherence()
        coherences.append(coherence)
    
    plt.bar(list(range(5,30)),coherences,tick_label=[str(i) for i in range(5,30)])
    plt.title("Coherence score per number of topics")
    plt.rcParams['figure.figsize'] = [20, 8]
    plt.show()
        
    return coherences.index(max(coherences))

