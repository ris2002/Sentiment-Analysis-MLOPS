import pandas as pd
import numpy as np
import re
import logging
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from typing import Tuple
from typing_extensions import Annotated

'''



Now in the below steps for feautre engineering we do the following
           1-We put the text to lower
           2- remove the user accounts in text like @elon_musk or @account
           3-we remove any urls
           4- we remove all puncuation marks and numbers
           5-we put a sentence in an array, like ex-
           i/p-I love Ferrari 
           o/p-['I','love','Ferrari;]
           6- we remove all the the common words and convert most words to root words
           7- WE can use WordNetLemmatizer() instead of port stemmer and that is efficient because it
           converts the word to theier dictionary roots where as POrtstemmer uses crude rules.But WordNetLemmatizer() is
           very slow.For this project we use Portstemmer
           ex-Word   Stemming  Lemmatization
            “studies”  studi ❌ study ✅
             “better”  better ❌    good ✅
           ex o/p-
                 0          upset cant updat facebook text might cri resul...
                 1               dive mani time ball manag save rest go bound
                 2                            whole bodi feel itchi like fire
                 3                                      behav im mad cant see
                 4                                                 whole crew
                                ...                       
                 1048567                              grandma make dinenr mum
                 1048568              midmorn snack time bowl chees noodl yum









'''
class Pre_Process_Strategies:
    def handle_data(self,df:pd.DataFrame)->pd.DataFrame:
        try:
            nltk.download('stopwords')
            stop_words=set(stopwords.words('english'))
            stemmer=PorterStemmer()
            df.drop(columns='id of the tweet',inplace=True)
            df.drop(columns=['query','user'],inplace=True)
            df.rename(columns={'polarity of tweet�': 'sentiment','date of the tweet': 'date','text of the tweet�': 'text'},inplace=True)
            '''
            It is to be noted that there are no null values and the main col is 'sentimennt' and 'text' so no other 
            '''
            
            df.drop(columns='date',inplace=True) 
           
            df['text']=df['text'].str.lower()
            df['text']=df['text'].str.replace(r'@\w+', '', regex=True)
            def pre_process_further(text):
                text=re.sub(r'http\S+|www\S+','',text)
                text=re.sub(r'[^a-z\s]','',text)
                words=text.split()
                words=[stemmer.stem(w) for w in words if  w not in stop_words]#stemmer.stem(w) for w in words if w not in stop_words
                return ' '.join(words)
            df['text']=df['text'].apply(pre_process_further)
            data=df
            return data
        except Exception as e:
            logging.error(f"Error while cleaning data: {e}")
            raise e  
   
    def split_test_train_and_feature_engineer(self,data:pd.DataFrame)->Tuple[
    Annotated[np.ndarray, "X_train"],
    Annotated[np.ndarray, "X_test"],#should be np.ndarray not no.array
    Annotated[pd.Series, "Y_train"],
    Annotated[pd.Series, "Y_test"]
]:
        try:
            X=data.drop(columns=['sentiment'])
            Y= data['sentiment']
            X_train_df, X_test_df, Y_train, Y_test =train_test_split(X, Y, test_size=0.2, random_state=2)#here 0.2 means 20% of taat will be for test.this methhod is used ot split the data into train and test
            tfid=TfidfVectorizer(max_features=5000,ngram_range=(1,2))
            X_train=tfid.fit_transform(X_train_df['text']).toarray()
            X_test=tfid.transform(X_test_df['text']).toarray()
            return X_train, X_test, Y_train, Y_test
        except Exception as e:
            logging.error(f"Error while dividing data or feature engineering of data: {e}")
            raise e


 













