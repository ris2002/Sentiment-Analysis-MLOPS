import pandas as pd
from zenml import step
import logging
from zenml.materializers.pandas_materializer import PandasMaterializer
from pydantic import BaseModel





class IngestData:
    def __init__(self, data_path: str):
        self.data_path = data_path

    def getData(self):
        df = pd.read_csv(self.data_path)
        print('ingest dATA')
        return df


@step()
def ingest_data(data_path: str)->pd.DataFrame:

    try:
        loader = IngestData(data_path)
        df = loader.getData()
        
        return df

    except Exception as e:
        logging.error(f"Error while ingesting data: {e}")
        raise e
