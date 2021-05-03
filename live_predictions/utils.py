import pandas as pd
from datetime import datetime as dt
import pathlib
import numpy as np
import paho.mqtt.client as paho
import time

class prediction_logger:
    def __init__(self, classes, date_format_string = "%Y-%m-%d %H:%M:%S"):
        self.classes = classes
        self.date_format_string = date_format_string

        self.path_prediction_log = pathlib.Path(f"pred_log_{dt.now().strftime('%y-%m-%d %H-%M')}.csv")
        if self.path_prediction_log.exists():
            pass
        else: 
            cols = ["timestamp"] + classes
            pd.DataFrame(columns = cols).to_csv(self.path_prediction_log, index = False)


    def log_values(self, values:np.array) -> dt:
        assert values.shape[0] == len(self.classes)
        timestamp = dt.now()
        _ = [timestamp.strftime(self.date_format_string)]
        _.extend(values)
        _ = [str(_) for _ in _]
        with open(self.path_prediction_log, "a") as file:
            file.write(f'{",".join(_)}\n')
        return timestamp

class mqtt_logger:
    def __init__(self, classes, date_format_string = "%Y-%m-%d %H:%M:%S"):
        self.classes = classes
        self.date_format_string = date_format_string
        self.connected = False
        self.HOST = "10.42.0.1"
        self.PORT = 1883
        self.KEEPALIVE = 60
        self.FLAG = True
        self.message_received = False
        self.topic = "test"

        # Create Client
        self.client = paho.Client()
        self.client.on_subscribe = self.on_subscribe
        self.client.on_unsubscribe = self.on_unsubscribe
        self.client.on_connect = self.on_connect
        self.client.on_message = self.on_message
        self.client.on_disconnect = self.on_disconnect

        self.response = self.client.connect(
            host = self.HOST,
            port = self.PORT,
            keepalive = self.KEEPALIVE
        )

        self.logdata = pd.DataFrame(columns = ["timestamp", "data"])
        # self.logdata["timestamp"] = self.logdata["timestamp"].astype("datetime64[ns]")
        
        self.client.loop_start()
        self.client.subscribe(topic = self.topic)
        # while True:
        #     time.sleep(0.1)

    def on_connect(self, client, userdata, flags, rc):
        self.connected = True
        print("Connected - rc:", rc)
    def on_message(self, client, userdata, message):
        msg = str(message.payload.decode("utf-8"))
        value = int(msg)
        row = {
            "timestamp": dt.now(),
            "data": value
        }
        self.logdata = self.logdata.append(row, ignore_index = True)
        if len(self.logdata) > 60:
            self.logdata = self.logdata.iloc[1:]
        print(self.logdata)
    def on_subscribe(self, client, userdata,mid, granted_qos):
        print("Subscribed:", str(mid), str(granted_qos))
    def on_unsubscribe(self, client, userdata,mid):
        print("Unsubscribed:", str(mid))
    def on_disconnect(self, client, userdata, rc):
        self.connected = False
        if rc!=0:
            print("Unexpected Disconnection:", rc)