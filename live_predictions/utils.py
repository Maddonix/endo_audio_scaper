import pathlib
import time
from datetime import datetime as dt

import numpy as np
import paho.mqtt.client as paho
import pandas as pd


class prediction_logger:
    def __init__(self, classes, date_format_string="%Y-%m-%d %H:%M:%S"):
        self.classes = classes
        self.date_format_string = date_format_string

        self.path_prediction_log = pathlib.Path(
            f"pred_log_{dt.now().strftime('%y-%m-%d %H-%M')}.csv")
        if self.path_prediction_log.exists():
            pass
        else:
            cols = ["timestamp"] + classes
            pd.DataFrame(columns=cols).to_csv(
                self.path_prediction_log, index=False)

    def log_values(self, values: np.array) -> dt:
        assert values.shape[0] == len(self.classes)
        timestamp = dt.now()
        _ = [timestamp.strftime(self.date_format_string)]
        _.extend(values)
        _ = [str(_) for _ in _]
        with open(self.path_prediction_log, "a") as file:
            file.write(f'{",".join(_)}\n')
        return timestamp
