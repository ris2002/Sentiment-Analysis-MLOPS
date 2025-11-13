from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
import numpy as np
import logging
from abc import ABC,abstractmethod
import pandas as pd


class Evaluation(ABC):
    def evaluate_model(self,Y_Test:pd.DataFrame,Y_pred:pd.DataFrame):
        pass
class Accuracy_Score(Evaluation):
    try:
        def evaluate_model(self,Y_Test:pd.DataFrame,Y_pred:np.ndarray):
            acc_score=accuracy_score(Y_Test,Y_pred)
            if not acc_score:
                raise ValueError('Error in accuracy function ')
            logging.info("Acc_Score: {}".format(acc_score))
            return acc_score

    except Exception as e:
        logging.error(f'Error in calculating accuracy score {e}')
        raise e

class Confusion_matrix(Evaluation):
    try:
        def evaluate_model(self,Y_Test:pd.DataFrame,Y_pred:np.ndarray):
            confusion_matrix_score=confusion_matrix(Y_Test,Y_pred)
            if not confusion_matrix:
                raise ValueError('Error in confusion matrix function ')
            logging.info('Confusion_Matrix:{}'.format(confusion_matrix_score))
            return confusion_matrix_score
    except Exception as e:
        logging.error(f'Error in calculating confusin matrix score {e}')

class Classification_Report(Evaluation):
    try:
        def evaluate_model(self,Y_Test:pd.DataFrame,Y_pred:np.ndarray):
            report=classification_report(Y_Test,Y_pred)
            if not report:
                raise ValueError('Error in classifiaction report function ')
            logging.info('Report:{}'.format(report))
            return report
    except Exception as e:
        logging.error(f'Error in calculating classification report score {e}')


