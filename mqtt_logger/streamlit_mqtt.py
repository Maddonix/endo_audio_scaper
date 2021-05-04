import json
import pandas as pd
import paho.mqtt.client as paho
import time
import numpy as np
from datetime import datetime as dt
import streamlit as st
import pathlib
import plotly.express as px

with open("config.json", "r") as f:
    config = json.load(f)
    
date_format_string = config["date_format_string"]
HOST = config["HOST"]
PORT = config["PORT"]
KEEPALIVE = 60
topic = config["topic"]

message_received = False
connected = False

log_path = pathlib.Path(f"logs/{dt.now().strftime('%Y-%m-%d %H:%M')}.csv")


######## Streamlit Config
st.set_page_config(page_title="Sensor Log", page_icon=None, initial_sidebar_state='auto')
st.header("Sensor Log")

st.write(f"Host: {HOST}")
st.write(f"PORT: {PORT}")
#########################

def on_connect(client, userdata, flags, rc):
    global connected
    connected = True
    # conn_stat_display.text(f"Connected: {connected}")
    st.write(f"Connected: {connected}")
    
def on_message(client, userdata, message):
    global chart
    global logdata
    msg = str(message.payload.decode("utf-8"))
    value = float(msg)
    row = {
        "timestamp": dt.now(),
        "value": value
    }
    logdata =logdata.append(row, ignore_index = True)
    fig = px.line(logdata, x = "timestamp", y = "value")
    chart.write(fig)
    if len(logdata) > 60:
        logdata = logdata.iloc[1:]
    
    
def on_subscribe(client, userdata,mid, granted_qos):
    st.write(f"Subscribed: {topic}")
    
def on_unsubscribe(client, userdata,mid):
    pass    
def on_disconnect(client, userdata, rc):
    global connected
    connected = False
    if rc!=0:
        print("Unexpected Disconnection:", rc)
        
        
logdata = pd.DataFrame(data = [[dt.now(),0]], columns= ["timestamp", "value"])
chart = st.empty()

def main():
    client=paho.Client()
    client.on_subscribe = on_subscribe
    client.on_unsubscribe = on_unsubscribe
    client.on_connect = on_connect
    client.on_message = on_message
    client.on_disconnect = on_disconnect
    response = client.connect(
        host = HOST,
        port = PORT,
        keepalive = KEEPALIVE
    )
    

    time.sleep(4)
    client.subscribe(topic = topic)
    client.loop_forever()

if __name__ == "__main__":
    main()