import pandas as pd
from src.io.ins_reader import read_ins_csv

def load_ins(path: str) -> pd.DataFrame:
    return read_ins_csv(path)
