from zenml import pipeline
from steps.ingest_data import ingest_data
from steps.clean_data import clean_data
from steps.train_model import train_model
from steps.model_evaluation import evaluate_model


@pipeline
def training_pipeline(csv_file):
    df=ingest_data(csv_file)
    cleaned_data=clean_data(df)
    model=train_model(cleaned_data)
    evaluation=evaluate_model(model)
    return evaluation


