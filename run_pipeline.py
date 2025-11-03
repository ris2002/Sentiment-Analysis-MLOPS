import kagglehub
from kagglehub import KaggleDatasetAdapter
from pipelines.training_pipeline import training_pipeline
output_csv_path = "sentiment140.csv"
data_file=""
if __name__ =="__main__":
    training_pipeline(data_file)