from typing import List
from collections import namedtuple
import json

dist_params = namedtuple("dist_params", ["param", "param_type"])

distributions_dict = {
    "const": [dist_params("value", "const")], 
    "choose": [dist_params("list", "list")], 
    "uniform": [dist_params("min", "number"), dist_params("max", "number")], 
    "normal": [dist_params("mean", "number"), dist_params("std", "number")],
    "truncnorm": [
        dist_params("mean", "number"),
        dist_params("std", "number"),
        dist_params("min", "number"), 
        dist_params("max", "number")
    ]
}

class ScapeConfig():
    def __init__(self, cfg_dict = None):
        if cfg_dict:
            self.set_defaults()
        else:
            self.set_defaults()

    def load_from_json(self, cfg_json):
        cfg_json = cfg_json.read().decode("utf-8")
        cfg_json = json.loads(cfg_json)

        self.seed = cfg_json["seed"]
        self.n_soundscapes = cfg_json["n_soundscapes"]
        self.ref_db = cfg_json["ref_db"]
        self.min_events = cfg_json["min_events"]
        self.max_events = cfg_json["max_events"]
        self.event_time = cfg_json["event_time"]
        self.event_duration =cfg_json["event_duration"]
        self.source_time = cfg_json["source_time"]
        self.snr = cfg_json["snr"]
        self.pitch = cfg_json["pitch"]
        self.time_stretch = cfg_json["time_stretch"]
        self.soundscape_type = cfg_json["soundscape_type"]
        

    def make_num_dist_default_value_dict(
        self,
        dist:str,
        value: float,
        mean: float,
        std: float,
        _min: float,
        _max: float,
        input_range: List[float],
        step: float):
        _dict = {
            "dist": dist,
            "value": float(value),
            "mean": float(mean),
            "std": float(std),
            "min": float(_min),
            "max": float(_max),
            "step": float(step),
            "input_range": [float(_) for _ in input_range]
        }

        return _dict

    def set_defaults(self):
        # general
        self.seed = 42
        self.soundscape_type = "foreground"
        self.n_soundscapes = {"min": 10, "max":10000, "value": 100}
        self.ref_db = -20
        self.duration = 10.0

        self.fg_labels = []
        self.bg_labels = []
        self.fg_labels_used = []
        self.bg_labels_used = []

        self.protected_labels = []

        self.min_events = 1
        self.max_events = 9

        #To Do: Implement reverb distribution
        self.reverb = 0.1
        self.event_time = self.make_num_dist_default_value_dict("truncnorm", 0, 5, 2, 0, 10, [0, self.duration], 0.5)
        self.source_time = self.make_num_dist_default_value_dict("const", 0,5,2,0,10, [0, self.duration], 0.5)
        self.event_duration = self.make_num_dist_default_value_dict("uniform", 1,2,1,1,10, [0, self.duration], 0.5)
        self.snr = self.make_num_dist_default_value_dict("uniform", 1, 1, 0.2, 0.1, 3, [0.1, 10], 0.1)
        self.pitch = self.make_num_dist_default_value_dict("uniform", 0, 0, 2, -5, 5, [0,10], 0.5)
        self.time_stretch = self.make_num_dist_default_value_dict("uniform", 1,1,0.1, 0.5, 1, [0.5, 1.5], 0.1)

    def get_params(self):
        params = {
            "seed": self.seed,
            "n_soundscapes": self.n_soundscapes,
            "ref_db": self.ref_db,
            "duration": self.duration,
            "min_events": self.min_events,
            "max_events": self.max_events,
            "event_time": self.event_time,
            "event_duration": self.event_duration,
            "source_time": self.source_time,
            "snr": self.snr,
            "pitch": self.pitch,
            "time_stretch": self.time_stretch,
            "soundscape_type": self.soundscape_type,
            "fg_labels_used": self.fg_labels_used,
            "bg_labels_used": self.bg_labels_used
        }

        for key, value in params.items():
            if type(value) is dict:
                if "container" in value:
                    del params[key]["container"]


        return params