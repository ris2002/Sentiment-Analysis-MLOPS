from zenml import step
import pandas as pd

@step
def train_model(dataset:pd.DataFrame)->None:
    pass