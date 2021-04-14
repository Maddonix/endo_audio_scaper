import pathlib
from typing import List
from datetime import datetime as dt
import scaper
from config_soundscape_generator import ScapeConfig
import pandas as pd

date_format = r"%Y %m %d %Hh %Mmin"
distribution_choices = [
        "uniform",
        "normal",
        "truncnorm",
        "const"
    ]

def get_labels_of_path(path:pathlib.Path) -> List[str]:
    return [_.name for _ in path.iterdir() if _.is_dir()] 

def get_export_folder_path(path:pathlib.Path) -> pathlib.Path:
    return path.joinpath(dt.now().strftime(date_format))

def make_export_folder(path:pathlib.Path) -> pathlib.Path:
    path = get_export_folder_path(path)
    if not path.exists():
        path.mkdir()

    return path

def reset_scaper(sc:scaper.Scaper) -> scaper.Scaper:        
    sc.reset_bg_event_spec()
    sc.reset_fg_event_spec()
    
    return sc

def add_bg(sc:scaper.Scaper) -> scaper.Scaper: 
    sc.add_background(
        label=("choose", []),
        source_file = ("choose", []),
        source_time = ("const", 0)
    )
    return sc

def make_slider_widget(streamlit_element, label, _min, _max, value, step = 1):
    return streamlit_element.slider(label, _min, _max, value, step)

def make_widget_key(base:str, suffix:str):
    return f"{base}_{suffix}"

def make_distribution_select_container(streamlit_element, label: str, dist_dict: {}, cfg, expandable: bool = False, key: str = None):
    # get dist from cfg
    # important to set new stuff with widgets
    dist = dist_dict["dist"]
    widget_dict = {
        "container": None,
        "select_dist_type": None,
        "select_min": None, 
        "select_max": None,
        "select_mean": None,
        "select_std": None
    }
    index_default_dist = distribution_choices.index(dist)

    container = streamlit_element.beta_expander(label, expanded = expandable).beta_container()
    widget_dict["container"] = container
    select_dist = container.selectbox("Distribution Type", distribution_choices, index = index_default_dist, key = make_widget_key(key, "select_dist"))
    widget_dict["dist"] = select_dist

    if select_dist == "truncnorm":
        widget_dict = make_trunc_norm_dist_widget(container, dist_dict, widget_dict, key = key)
    elif select_dist == "normal":
        widget_dict = make_norm_dist_widget(container, dist_dict, widget_dict, key = key)
    elif select_dist == "uniform":
        widget_dict = make_uniform_dist_widget(container, dist_dict, widget_dict, key = key)

    return widget_dict


def make_trunc_norm_dist_widget(streamlit_element, dist_dict, widget_dict, key:str):
    widget_dict["min"] = streamlit_element.slider("Select Min", dist_dict["input_range"][0],
        dist_dict["input_range"][1], dist_dict["min"], step = dist_dict["step"], key = make_widget_key(key, "min"))
    widget_dict["max"] = streamlit_element.slider("Select Max", dist_dict["input_range"][0],
        dist_dict["input_range"][1], dist_dict["max"], step = dist_dict["step"], key = make_widget_key(key, "max"))
    widget_dict["mean"] = streamlit_element.slider("Select Mean", dist_dict["input_range"][0],
        dist_dict["input_range"][1], dist_dict["mean"], step = dist_dict["step"], key = make_widget_key(key, "mean"))
    widget_dict["std"] = streamlit_element.slider("Select Std", dist_dict["input_range"][0],
        dist_dict["input_range"][1], dist_dict["std"], step = dist_dict["step"], key = make_widget_key(key, "std"))

    if widget_dict["min"] > widget_dict["max"]:
        streamlit_element.error("min must be <= max")
    if widget_dict["mean"] >= widget_dict["max"] or widget_dict["mean"] <= widget_dict["min"]:
        streamlit_element.error("mean must be < max and > min")

    return widget_dict

def make_norm_dist_widget(streamlit_element, dist_dict, widget_dict, key: str):
    widget_dict["mean"] = streamlit_element.slider("Select Mean", dist_dict["input_range"][0],
        dist_dict["input_range"][1], dist_dict["mean"], step = dist_dict["step"], key = make_widget_key(key, "mean"))
    widget_dict["std"] = streamlit_element.slider("Select Std", dist_dict["input_range"][0],
        dist_dict["input_range"][1], dist_dict["std"], step = dist_dict["step"], key = make_widget_key(key, "std"))

    return widget_dict

def make_uniform_dist_widget(streamlit_element, dist_dict, widget_dict, key: str):
    widget_dict["min"] = streamlit_element.slider("Select Min", dist_dict["input_range"][0],
        dist_dict["input_range"][1], dist_dict["min"], step = dist_dict["step"], key = make_widget_key(key, "min"))
    widget_dict["max"] = streamlit_element.slider("Select Max", dist_dict["input_range"][0],
        dist_dict["input_range"][1], dist_dict["max"], step = dist_dict["step"], key = make_widget_key(key, "max"))

    if widget_dict["min"] > widget_dict["max"]:
        streamlit_element.error("min must be <= max")

    return widget_dict

def parse_args_by_dist_type(dist_dict:{}):
    dist = dist_dict["dist"]
    if dist is "normal":
        args = (
            dist_dict["dist"],
            dist_dict["mean"],
            dist_dict["std"]
        )
    elif dist is "truncnorm":
        args = (
            dist_dict["dist"],
            dist_dict["mean"],
            dist_dict["std"],
            dist_dict["min"],
            dist_dict["max"]
        )
    elif dist is "uniform": 
        args = (
            dist_dict["dist"],
            dist_dict["min"],
            dist_dict["max"]
        )
    elif dist is "choose":
        args = (
            dist_dict["dist"],
            dist_dict["value"]
        )
    elif dist is "const":
        args = (
            dist_dict["dist"],
            dist_dict["value"]
        )
    else:
        print(f"'{dist}' is not a valid distribution key")
        args = ()
    
    return args

def add_event(sc:scaper.Scaper, cfg: ScapeConfig) -> scaper.Scaper:
    sc.add_event(
        label=("choose", []),
        source_file = ("choose", []),
        source_time = (
            cfg.source_time["dist"],
            cfg.source_time["value"]
        ),
        event_time=(
            cfg.event_time["dist"],
            cfg.event_time["mean"],
            cfg.event_time["std"],
            cfg.event_time["min"],
            cfg.event_time["max"]
        ),
        event_duration=(
            cfg.event_duration["dist"],
            cfg.event_duration["min"],
            cfg.event_duration["max"]
        ),
        snr=(
            cfg.snr["dist"],
            cfg.snr["min"],
            cfg.snr["max"]
        ),
        pitch_shift=(
            cfg.pitch["dist"],
            cfg.pitch["min"],
            cfg.pitch["max"]
        ),
        time_stretch=(
            cfg.time_stretch["dist"],
            cfg.time_stretch["min"],
            cfg.time_stretch["max"]
    ))
    return sc

def make_export_filepaths(n:int, path:pathlib.Path) -> List[pathlib.Path]:
    audiofile = path.joinpath(f"soundscape_unimodal{n}.wav").as_posix()
    jamsfile = path.joinpath(f"soundscape_unimodal{n}.jams").as_posix()
    txtfile = path.joinpath(f"soundscape_unimodal{n}.txt").as_posix()

    return [audiofile, jamsfile, txtfile]

def read_event_table_for_path(name:str,path:pathlib.Path) -> pd.DataFrame:
    name = pathlib.Path(name).with_suffix(".txt")
    path = path.joinpath(name)
    df = pd.read_csv(path, sep = "\t", index_col = None, names = ["event_start", "event_end", "event_label"])
    return df