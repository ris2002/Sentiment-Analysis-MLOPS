from src.data_cleaning import Pre_Process_Strategies
from fastapi import FastAPI
from pydantic import BaseModel#Base Model helps vaildating input data
import mlflow
import logging
app=FastAPI()
cleaning_strategies=Pre_Process_Strategies()
#pulls model  = mlflow.pyfunc.load_model("models:/sentiment_analysis_model/2") from local mlflow tracking directory not the regisstry
#so alternatively 

model = mlflow.pyfunc.load_model("/app/mlruns/516471548267291175/5aa8c63aac9d467a8ea9818ffaf0e584/artifacts/model")
#you have to mention the storage loccation ad etc from the yaml file gennerated from trraing and make surre that yyaml file is copied in to docker image

class InputText(BaseModel):
    text:str
@app.post('/predict')
def predict(input:InputText):
    
    try:
        
        cleaned_words=cleaning_strategies.clean_deployment_text(input.text)
        prediction=model.predict(cleaned_words)
        logging.info('Prediction Completed ')
        return{"prediction": prediction.tolist()}
    except Exception as e:
        logging.error(f'Prediction API not working:{e}')
        raise e






