
from pipelines.training_pipeline import training_pipeline
output_csv_path = "sentiment140.csv"
data_file=('/Users/rishilboddula/Desktop/MLOPS/Sentiment-Analysis-MLOPS/archive (13)/training.1600000.processed.noemoticon.csv')

training_pipeline(data_file)