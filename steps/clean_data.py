from zenml import step
import pandas as pd
from src.data_cleaning import Pre_Process_Strategies
from typing import Tuple
import logging
import numpy as np
from typing_extensions import Annotated
@step()
def clean_data(df: pd.DataFrame) -> Tuple[
    Annotated[np.ndarray, "X_train"],
    Annotated[np.ndarray, "X_test"],
    Annotated[pd.Series, "Y_train"],
    Annotated[pd.Series, "Y_test"]
]:
    try:
        processor=Pre_Process_Strategies()
        cleaned_df=processor.handle_data(df)
        X_train,X_test,Y_train,Y_test=processor.split_test_train_and_feature_engineer(cleaned_df)
        logging.info("Data cleaning,feature_engineering and division successful")
        return X_train,X_test,Y_train,Y_test

    except Exception as e:


        logging.error(f"Error while cleaning data: {e}")
        raise e
