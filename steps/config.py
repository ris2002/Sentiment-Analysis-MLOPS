#all model configurations will be stored herre

class Model_Config:
    def __init__(self):
        self.model_type='Logistic Regression'
        self.model_param={
            'Logistic Regression':{
            'solver':'saga','max_iter':100,'n_jobs':-1
            },
            'Naive Bayes':{

            },
            'KNN':{

            }
        }
        