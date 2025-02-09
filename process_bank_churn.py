import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from typing import Dict, List, Tuple

def split_data(raw_df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Splits the raw dataframe into training and validation sets."""
    return train_test_split(raw_df, test_size=test_size, random_state=random_state, stratify=raw_df['Exited'])

def drop_unnecessary_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Drops unnecessary columns before preprocessing."""
    return df.drop(columns=['Surname'], errors='ignore')

def separate_features_and_target(df: pd.DataFrame, target_col: str = 'Exited') -> Tuple[pd.DataFrame, pd.Series]:
    """Separates the input features and target variable from the dataset."""
    input_cols = list(df.columns)[1:-1]
    return df[input_cols], df[target_col]

def scale_numeric_features(train_inputs: pd.DataFrame, val_inputs: pd.DataFrame, numeric_cols: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame, MinMaxScaler]:
    """Scales numeric features using MinMaxScaler."""
    scaler = MinMaxScaler()
    train_inputs[numeric_cols] = scaler.fit_transform(train_inputs[numeric_cols])
    val_inputs[numeric_cols] = scaler.transform(val_inputs[numeric_cols])
    return train_inputs, val_inputs, scaler

def encode_categorical_features(train_inputs: pd.DataFrame, val_inputs: pd.DataFrame, categorical_cols: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame, OneHotEncoder, List[str]]:
    """Encodes categorical features using OneHotEncoder."""
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    encoder.fit(train_inputs[categorical_cols])
    encoded_cols = list(encoder.get_feature_names_out(categorical_cols))
    train_inputs[encoded_cols] = encoder.transform(train_inputs[categorical_cols])
    val_inputs[encoded_cols] = encoder.transform(val_inputs[categorical_cols])
    return train_inputs, val_inputs, encoder, encoded_cols

def preprocess_data(raw_df: pd.DataFrame) -> Dict[str, object]:
    """Processes the raw dataframe by splitting, scaling, and encoding the data."""
    raw_df = drop_unnecessary_columns(raw_df)
    train_df, val_df = split_data(raw_df)
    train_inputs, train_targets = separate_features_and_target(train_df)
    val_inputs, val_targets = separate_features_and_target(val_df)
    
    numeric_cols = train_inputs.select_dtypes(include=np.number).columns.tolist()[:-1]
    categorical_cols = train_inputs.select_dtypes(include='object').columns.tolist()
    
    train_inputs, val_inputs, scaler = scale_numeric_features(train_inputs, val_inputs, numeric_cols)
    train_inputs, val_inputs, encoder, encoded_cols = encode_categorical_features(train_inputs, val_inputs, categorical_cols)
    
    X_train = train_inputs[numeric_cols + encoded_cols]
    X_val = val_inputs[numeric_cols + encoded_cols]
    
    return {
        "X_train": X_train,
        "train_targets": train_targets,
        "X_val": X_val,
        "val_targets": val_targets,
        "input_cols": X_train.columns.tolist(),
        "scaler": scaler,
        "encoder": encoder,
    }

def preprocess_new_data(new_data: pd.DataFrame, scaler: MinMaxScaler, encoder: OneHotEncoder) -> pd.DataFrame:
    """Preprocesses new data using the given scaler and encoder."""
    new_data = drop_unnecessary_columns(new_data)
    numeric_cols = new_data.select_dtypes(include=np.number).columns.tolist()[1:-1]
    categorical_cols = new_data.select_dtypes(include='object').columns.tolist()
    
    new_data[numeric_cols] = scaler.transform(new_data[numeric_cols])
    encoded_cols = list(encoder.get_feature_names_out(categorical_cols))
    new_data[encoded_cols] = encoder.transform(new_data[categorical_cols])
    
    return new_data