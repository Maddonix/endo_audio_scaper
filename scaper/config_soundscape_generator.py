import icecream as ic
from collections import namedtuple

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
            ic("cfg dicts not yet supported")
            self.set_defaults()
        else:
            self.set_defaults()


    def test_cfg(self):
        params = [
            ("n_soundscapes", int),
            ("ref_db", int),
            ("duration", float),
            ("min_events", int),
            ("max_events", int),
            ("event_time_dist", str),
            ("event_time_mean", float),
            ("event_time_std", float),
            ("event_time_min", float),
            ("event_time_max", float)
            ("source_time_dist", str),
            ("source_time", float),
            # TO BE DONE

        ]

    def make_distribution_dict(self, dist:str, defaults:{} = None):
        assert dist in list(distributions_dict.keys())
        if defaults == None:
            defaults = {
                "min": 1,
                "max": 100,
                "mean": 50,
                "std": 20,
                "value": 2,
                "list": []
            }

        dist_dict = {_.param: _.param_type for _ in distributions_dict[dist]}

        if dist == "const":
            dist_dict["value"] = defaults["value"]
        elif dist == "choose":
            dist_dict["list"] = defaults["list"]
        elif dist == "uniform":
            dist_dict["min"] = defaults["min"]
            dist_dict["max"] = defaults["max"]
        elif dist == "normal":
            dist_dict["mean"] = defaults["mean"]
            dist_dict["std"] = defaults["std"]
        elif dist == "truncnorm":
            dist_dict["min"] = defaults["min"]
            dist_dict["max"] = defaults["max"]
            dist_dict["mean"] = defaults["mean"]
            dist_dict["std"] = defaults["std"]

        return dist_dict


    def set_defaults(self):
        # general
        self.seed = 42
        self.n_soundscapes = {"min": 1, "max":100, "value": 5}
        self.ref_db = -50
        self.duration = 10.0

        self.fg_labels = []
        self.bg_labels = []
        self.protected_labels = []

        self.min_events = 1
        self.max_events = 9

        self.reverb = 0.1

        self.event_time = {
            "dist": "truncnorm",
            "mean": 5.0,
            "std": 2.0,
            "min": 0,
            "max": 10
        }

        self.source_time = {
            "dist": "const",
            "time": 0
        }

        self.event_duration = {
            "dist": "uniform",
            "min": 0.5, 
            "max": 4
        }

        self.snr = {
            "dist": "uniform",
            "min": 6, 
            "max": 30
        }

        self.pitch = {
            "dist": "uniform",
            "min": -3,
            "max": 3
        }

        self.time_stretch = {
            "dist": "uniform", 
            "min": 0.8,
            "max": 1.2
        }

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
            "time_stretch": self.time_stretch
        }
        return params