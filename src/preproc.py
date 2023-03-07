import re
import numpy as np
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords
import nltk
from collections import Counter
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [20, 8]
nltk.download('stopwords')
nltk.download('wordnet')

#Clean each strin from emoticons, emojis, symbols, hashtags, links, stopwords etc.
def cleanString(str):
    str=str.lower()
    
    #Remove hashtags
    str=re.sub(r"#\S+","",str)

    #Remove links
    str=re.sub(r"http\S+","",str)
    
    
    emojis= re.compile("[" u"\U0001F600-\U0001F64F" u"\U0001F300-\U0001F5FF" u"\U0001F680-\U0001F6FF" u"\U0001F1E0-\U0001F1FF" u"\U000024C2-\U0001F251"  u"\U00010000-\U0010ffff" "]+",
                        flags=re.UNICODE )
    
    str=emojis.sub(r"",str)
    str=re.sub(r"[{}\[\]\|\\:;,.~`!@#\$%^&\*\(\)\+=\?/\'…’]+","",str)
    str=re.sub("\'\"","",str)
    str=re.sub(r"\s+"," ",str)
    str=re.sub(r" - "," ",str)
    
    #Lemmatization and removal of stop words
    lemmatizer = WordNetLemmatizer()
    str_list=[lemmatizer.lemmatize(word) for word in str.split() if word not in stopwords.words("english")]
    new_str=" ".join(str_list)
    str=new_str
    
    return str

#Clean the file
def cleanDf(file):
    df=pd.read_csv(file,sep="\t")
    df=df.astype({"text": str})
    df["text"]=df["text"].apply(cleanString)
    #Remove invalid tweets
    df=df.dropna(subset=["sentiment","retweets","date"])
    df=df.reset_index(drop=True)
    
    return df

df_train=cleanDf("/content/gdrive/MyDrive/Colab Notebooks/train.tsv")
df_test=cleanDf("/content/gdrive/MyDrive/Colab Notebooks/test.tsv")
df=pd.concat([df_train, df_test], ignore_index=True)
