import pandas as pd
from src.io.imu_reader import read_imu_csv

def load_imu(path: str) -> pd.DataFrame:
    return read_imu_csv(path)
