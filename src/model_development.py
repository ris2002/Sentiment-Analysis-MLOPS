import pandas as pd
from abc import ABC, abstractmethod
import logging
from sklearn.linear_model import LogisticRegression
import numpy as np
from sklearn.base import ClassifierMixin



class Model(ABC):
    @abstractmethod
    def train(self,X_Train: np.ndarray,Y_Train: pd.DataFrame):
       
        pass
class Logistic_Regression(Model):
    def train(self,X_Train: np.ndarray,Y_Train: pd.DataFrame,**kwargs):
        try:
            model=LogisticRegression(**kwargs)
            model.fit(X_Train,Y_Train)
            logging.info('Model has run  successfully')
            return model
            
        except Exception as e:
            logging.error(f'Error in training data in logistic regression{e}')
            raise e


 #     class Naive_Bayes(Model):
 #   def train(self,X_train,Y_train,**params):
 #       try:
  #          pass
 #       except Exception as e:
  #          logging.error('Error in training data in naive bayes regression ')
#



