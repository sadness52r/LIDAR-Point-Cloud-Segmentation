import pandas as pd

def read_ins_csv(filename: str) -> pd.DataFrame:
    columns = [
        "timestamp", "latitude", "longitude", "height",
        "north_velocity", "east_velocity", "up_velocity",
        "roll", "pitch", "azimuth", "status",
    ]
    return pd.read_csv(filename, names=columns)
