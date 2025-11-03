import kagglehub
from kagglehub import KaggleDatasetAdapter
from pipelines.training_pipeline import training_pipeline
output_csv_path = "sentiment140.csv"
file_path = ""
data_file = kagglehub.load_dataset(
    KaggleDatasetAdapter.PANDAS,
    "kazanova/sentiment140",  
    file_path,             
)
if __name__ =="__main__":
    training_pipeline(data_file)