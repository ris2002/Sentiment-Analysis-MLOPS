from zenml import pipeline
from steps.ingest_data import ingest_data
from steps.clean_data import clean_data
from steps.train_model import train_model
from steps.model_evaluation import evaluate_model
@pipeline
def training_pipeline(csv_file):
    df = ingest_data(csv_file)
    X_train, X_test, y_train, y_test = clean_data(df)
    model = train_model(X_train, y_train)
    acc_score_result, confusion_matrix_result, report_results = evaluate_model(X_test, y_test,model)

