import pandas as pd

def read_imu_csv(filename: str) -> pd.DataFrame:
    columns = [
        "timestamp", "qx", "qy", "qz", "qw",
        "eul_x", "eul_y", "eul_z",
        "gyr_x", "gyr_y", "gyr_z",
        "acc_x", "acc_y", "acc_z",
        "mag_x", "mag_y", "mag_z",
    ]
    return pd.read_csv(filename, names=columns)
