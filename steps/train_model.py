from zenml import step
import pandas as pd
from src.model_development import Logistic_Regression
import numpy as np
import logging
from .config import Model_Config
from sklearn.linear_model import LogisticRegression
from typing_extensions import Annotated
from sklearn.base import ClassifierMixin
import mlflow
from zenml.client import Client
experiment_tracker=Client().active_stack.experiment_tracker.name

@step(experiment_tracker=experiment_tracker)
def train_model( X_Train: np.ndarray,Y_Train: pd.DataFrame)->ClassifierMixin:
                 
    
    try:
        config=Model_Config()
        if config.model_type=='Logistic Regression':
            model=Logistic_Regression()
            mlflow.sklearn.autolog()
            params=config.model_param[config.model_type]
            trained_model=model.train(X_Train,Y_Train,**params)
            return trained_model
        else:
            ValueError('modeel not assigned')

        
        
        
    except Exception as e:  
        logging.error(f"Error in training model: {e}")
        raise e
        

  