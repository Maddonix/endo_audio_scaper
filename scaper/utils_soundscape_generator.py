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
    """Iterates over given directory and returns the names of folders as list. Not recursive.

    Args:
        path (pathlib.Path): Path to iterate over

    Returns:
        List[str]: List of strings containing foldernames
    """
    return [_.name for _ in path.iterdir() if _.is_dir()] 

def get_export_folder_path(path:pathlib.Path) -> pathlib.Path:
    """Generates the path to a folder with a current timestamp as extension of the given folder

    Args:
        path (pathlib.Path): path to expand

    Returns:
        pathlib.Path: path to new folder
    """
    return path.joinpath(dt.now().strftime(date_format))

def make_export_folder(path:pathlib.Path) -> pathlib.Path:
    """Checks if path exists. If path does not exist, it is created as directory.

    Args:
        path (pathlib.Path): path to create

    Returns:
        pathlib.Path: path of new folder
    """
    path = get_export_folder_path(path)
    if not path.exists():
        path.mkdir()

    return path

def reset_scaper(sc:scaper.Scaper) -> scaper.Scaper: 
    """Resets foreground and background event specifications of given scaper object

    Args:
        sc (scaper.Scaper): Object to reset

    Returns:
        scaper.Scaper: Object with fore/background specs reset
    """           
    sc.reset_bg_event_spec()
    sc.reset_fg_event_spec()
    
    return sc

def add_bg(sc:scaper.Scaper, cfg) -> scaper.Scaper:
    """Add Background to scaper object

    Args:
        sc (scaper.Scaper): scaper object to add a background event to
        cfg (?): Configuration object

    Returns:
        scaper.Scaper: Scaper object with added background event
    """
    sc.add_background(
        label=("choose", cfg.bg_labels_used),
        source_file = ("choose", []),
        source_time = ("const", 0)
    )
    return sc

def make_slider_widget(streamlit_element, label:str, _min:float, _max:float, value:float, step:float = 1):
    return streamlit_element.slider(label, _min, _max, value, step)

def make_widget_key(base:str, suffix:str):
    return f"{base}_{suffix}"

def make_distribution_select_container(streamlit_element, label: str, dist_dict: {}, cfg: ScapeConfig, expandable: bool = False, key: str = ""):
    """Adds a distribution settings container to given streamlit element.

    Args:
        streamlit_element (streamlit element): Object to add the container to.
        label (str): label of the new container
        dist_dict (dict): Dictionary containing the necessary information to create the widget.
        cfg (ScapeConfig): cfg object
        expandable (bool, optional): If true, container is expanded at startup. Defaults to False.
        key (str, optional): key for the new widget. Defaults to "".

    Returns:
        dict: dictionary of widgets inside of the newly generated container
    """
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
    elif select_dist == "const":
        widget_dict = make_const_numeric_input_widget(container, dist_dict, widget_dict, key = key)

    return widget_dict


def make_trunc_norm_dist_widget(streamlit_element, dist_dict:{}, widget_dict: {}, key:str):
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

def make_norm_dist_widget(streamlit_element, dist_dict:{}, widget_dict:{}, key: str):
    widget_dict["mean"] = streamlit_element.slider("Select Mean", dist_dict["input_range"][0],
        dist_dict["input_range"][1], dist_dict["mean"], step = dist_dict["step"], key = make_widget_key(key, "mean"))
    widget_dict["std"] = streamlit_element.slider("Select Std", dist_dict["input_range"][0],
        dist_dict["input_range"][1], dist_dict["std"], step = dist_dict["step"], key = make_widget_key(key, "std"))

    return widget_dict

def make_uniform_dist_widget(streamlit_element, dist_dict:{}, widget_dict:{}, key: str):
    widget_dict["min"] = streamlit_element.slider("Select Min", dist_dict["input_range"][0],
        dist_dict["input_range"][1], dist_dict["min"], step = dist_dict["step"], key = make_widget_key(key, "min"))
    widget_dict["max"] = streamlit_element.slider("Select Max", dist_dict["input_range"][0],
        dist_dict["input_range"][1], dist_dict["max"], step = dist_dict["step"], key = make_widget_key(key, "max"))

    if widget_dict["min"] > widget_dict["max"]:
        streamlit_element.error("min must be <= max")

    return widget_dict

def make_const_numeric_input_widget(streamlit_element, dist_dict:{}, widget_dict:{}, key:str):
    widget_dict["value"] = streamlit_element.slider("Value", dist_dict["input_range"][0],
        dist_dict["input_range"][1], dist_dict["value"], step = dist_dict["step"], key = make_widget_key(key, "const"))
    
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
        label=("choose", cfg.fg_labels_used),
        source_file = ("choose", []),
        source_time = parse_args_by_dist_type(cfg.source_time),
        event_time=parse_args_by_dist_type(cfg.event_time),
        event_duration=parse_args_by_dist_type(cfg.event_duration),
        snr=parse_args_by_dist_type(cfg.snr),
        pitch_shift=parse_args_by_dist_type(cfg.pitch),
        time_stretch=parse_args_by_dist_type(cfg.time_stretch))
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