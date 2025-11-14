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
import pickle

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

nltk.download('stopwords')
stop_words=set(stopwords.words('english'))#contains all common words in english
stemmer=PorterStemmer()
class Pre_Process_Strategies:
    def handle_data(self,df:pd.DataFrame)->pd.DataFrame:
        try:
            #nltk.download('stopwords')
           # stop_words=set(stopwords.words('english'))
           # stemmer=PorterStemmer()
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
            tfid=TfidfVectorizer(max_features=5000,ngram_range=(1,2))#max_features means it tells tfidf to only consider top 5000 words, n_grams=(1,2)means it tells it t conasider eeither single words like good,bad or 2 words like not good,nnot bad etc
            X_train=tfid.fit_transform(X_train_df['text']).toarray()
            X_test=tfid.transform(X_test_df['text']).toarray()
            # create a new filepath and open 'tfidf_vectorizer.pkl' 'wb' meams write in binary mode
            with open('tfidf_vectorizer.pkl','wb') as f:
                pickle.dump(tfid,f)#writes tfid to the file
                print("Saved successfully!")
                           
            return X_train, X_test, Y_train, Y_test
        except Exception as e:
            logging.error(f"Error while dividing data or feature engineering of data: {e}")
            raise e
    
    def clean_deployment_text(self,text:str)->np.ndarray:
        try:
           
            text = text.lower()
            text=re.sub(r'http\S+|www\S+','',text)
            text=re.sub(r'[^a-z\s]','',text)
            text=re.sub(r'@\w+','',text)
            #rb=read binary
            with open('tfidf_vectorizer.pkl','rb')as f:
                tfid=pickle.load(f)
            words=text.split()#splits the string on whitespace by default.
            words=[stemmer.stem(w) for w in words if w not in stop_words]
            cleaned_text = " ".join(words)
            
            pre_processed_words = tfid.transform([cleaned_text]).toarray()
            return pre_processed_words
        except Exception as e:
            logging.error(f'Error in cleaning the raw text for deployment')
            raise e









'''

Bag of words- It means count how many times each word appears in a document.
ex-Example:

Document	Text
D1	"I love dogs"
D2	"I love cats"

Vocabulary = ["I", "love", "dogs", "cats"]

Doc	I	love	dogs	cats
D1	1	1	1	0
D2	1	1	0	1

So each sentence becomes a vector of word counts.

🔸 Problem: common words ("I", "love") appear everywhere — not very informative.

TFIDF-TF-IDF (Term Frequency–Inverse Document Frequency)
make ccommon words less important and rare words important
IDF=log(N/(1+n(t)))
where N = total docs and n(t)=number of docs containing the term
it gives more weight to unique words
But doesnt cqpture synonyms.
We use this in the project because naive bayes and logistic regression do not work properly with word embeddings.
------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
fit_transform()- learns vocalblury and IDF weights and then it converts into TF-IDF vectors
transform()-Uses the same vocalblury and TF-IDF vectors from trrainig data aND just transforms the testing data to TF-IDF vectors 
''' 













