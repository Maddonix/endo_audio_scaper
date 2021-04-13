import pathlib
from typing import List
from datetime import datetime as dt
import scaper
from config_soundscape_generator import ScapeConfig
import pandas as pd

date_format = r"%Y %m %d %Hh %Mmin"

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

def make_slider_widget(streamlit_element, label, _min, _max, value):
    return streamlit_element.slider(label, _min, _max, value)

def add_event(sc:scaper.Scaper, cfg: ScapeConfig) -> scaper.Scaper:
    sc.add_event(
        label=("choose", []),
        source_file = ("choose", []),
        source_time = (
            cfg.source_time["dist"],
            cfg.source_time["time"]
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