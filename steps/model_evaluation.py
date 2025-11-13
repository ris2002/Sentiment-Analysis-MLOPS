from zenml import step
import pandas as pd
from sklearn.base import RegressorMixin
import numpy as np
from typing_extensions import Annotated
from typing import Tuple
from src.data_evaluation import Classification_Report,Confusion_matrix,Accuracy_Score
import logging
from sklearn.linear_model import LogisticRegression
from sklearn.base import RegressorMixin,ClassifierMixin
import mlflow
from zenml.client import Client
experiment_tracker=Client().active_stack.experiment_tracker.name

#Annoted gives ore info about the o/p
@step(experiment_tracker=experiment_tracker)
def evaluate_model( X_Test: np.ndarray,Y_Test: pd.DataFrame,trained_model:ClassifierMixin)->Tuple[Annotated[float,'Accuracy_Score'],Annotated[np.ndarray,'Confusion_Matrix'],Annotated[str,'Classification Report']]:
    try:
        pred_val=trained_model.predict(X_Test)
        logging.info('Model Predicted')
        accuracy_score=Accuracy_Score()
        acc_score_result=accuracy_score.evaluate_model(Y_Test,pred_val)
        mlflow.log_metric('accuracy_score:',acc_score_result)
        
        logging.info("Successfully calculated Acc_Score: {}".format(acc_score_result))
        confusion_matrix=Confusion_matrix()
        confusion_matrix_result=confusion_matrix.evaluate_model(Y_Test,pred_val)
        cm = confusion_matrix_result
        mlflow.log_metric('true_negatives', int(cm[0][0]))
        mlflow.log_metric('false_positives', int(cm[0][1]))
        mlflow.log_metric('false_negatives', int(cm[1][0]))
        mlflow.log_metric('true_positives', int(cm[1][1]))

        logging.info("Successfully calculated Acc_Score: {}".format(confusion_matrix_result))
        classification_report=Classification_Report()
        report_results=classification_report.evaluate_model(Y_Test,pred_val)
        
        logging.info("Successfully calculated Acc_Score: {}".format(report_results))
       
        
        return acc_score_result,confusion_matrix_result,report_results
    except Exception as e:
        logging.error(f'Err in Evaluating the scores{e}')
        raise e