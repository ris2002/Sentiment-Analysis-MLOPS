import pandas as pd
from zenml import step
import logging
@step
def ingest_data(data_path:pd.DataFrame)->None:
    pass