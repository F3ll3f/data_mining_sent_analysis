import pickle
import pandas as pd
from sklearn.model_selection import train_test_split

def createTestAndTrain(infile, outTrain,outTest,percentage=1):
    df=pd.read_pickle(infile)
        
    df=df.sample(frac=percentage)
    (train_df,test_df)=train_test_split(df,test_size=0.2)
    train_df.to_csv(outTrain,sep="\t",index=False)
    test_df.to_csv(outTest,sep="\t",index=False)

    return df,train_df,test_df
