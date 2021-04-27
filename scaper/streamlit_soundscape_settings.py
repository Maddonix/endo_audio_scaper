import utils_soundscape_generator as uscg
from config_soundscape_generator import ScapeConfig
import json

def make_soundscape_settings_container(streamlit_element, cfg: ScapeConfig):
    """Generates the UI elements for soundscape settings.

    Args:
        streamlit_element (streamlit object): Streamlit UI object to which the elements will be added
        cfg (ScapeConfig): ScapeConfig object containing the specifications for the UI objects which will be generated
    """
    container = streamlit_element.beta_container()
    container.header("Settings")
    
    cfg_json = container.file_uploader("Load previously saved config.json", type=["json"], accept_multiple_files=False)
    if cfg_json is not None:
        cfg.load_from_json(cfg_json)

    col_1, col_2, col_3 = container.beta_columns(3)
    expanded = False

    # Col 1
    make_labels_container(col_1, cfg, expanded = expanded)
    make_event_time_distribution_container(col_1, cfg, expanded = expanded)
    make_event_duration_distribution_container(col_1, cfg, expanded = expanded)

    # Col 2
    make_general_settings_container(col_2, cfg, expanded = expanded)
    make_time_stretch_distribution_container(col_2, cfg, expanded = expanded)
    make_pitch_distribution_container(col_2, cfg, expanded = expanded)

    # Col 3
    make_source_time_distribution_container(col_3, cfg, expanded = expanded)
    make_snr_distribution_container(col_3, cfg, expanded = expanded)

def make_event_time_distribution_container(streamlit_element, cfg: ScapeConfig, expanded:bool = False):
    """Generates a container for the event time distribution settings and adds them to the given streamlit element.

    Args:
        streamlit_element (streamlit element): Object to which the container will be added
        cfg (ScapeConfig): Config object containing the specifications
        expanded (bool, optional): If True, the container will be expanded by default. Defaults to False.
    """
    event_time_dist_widget_dict = uscg.make_distribution_select_container(
        streamlit_element,
        label = "Event Time Distribution",
        dist_dict = cfg.event_time,
        cfg = cfg,
        expandable = False,
        key = "event_time_dist_widget"
    )
    for key, value in event_time_dist_widget_dict.items():
        if value:
            cfg.event_time[key] = value

def make_event_duration_distribution_container(streamlit_element, cfg: ScapeConfig, expanded:bool = False):
    event_duration_dist_widget_dict = uscg.make_distribution_select_container(
        streamlit_element,
        label = "Event Duration Distribution",
        dist_dict = cfg.event_duration,
        cfg = cfg,
        expandable = False,
        key = "event_duration_dist_widget"
    )

    for key, value in event_duration_dist_widget_dict.items():
        if value:
            cfg.event_duration[key] = value

def make_time_stretch_distribution_container(streamlit_element, cfg: ScapeConfig, expanded:bool = False):
    time_stretch_dist_widget_dict = uscg.make_distribution_select_container(
        streamlit_element,
        label = "Time Stretch Distribution",
        dist_dict = cfg.time_stretch,
        cfg = cfg,
        expandable = False,
        key = "time_stretch_dist_widget"
    )
    for key, value in time_stretch_dist_widget_dict.items():
        if value:
            cfg.time_stretch[key] = value

def make_pitch_distribution_container(streamlit_element, cfg: ScapeConfig, expanded:bool = False):
    pitch_dist_widget_dict = uscg.make_distribution_select_container(
        streamlit_element,
        label = "Pitch Distribution",
        dist_dict = cfg.pitch,
        cfg = cfg,
        expandable = False,
        key = "pitch_dist_widget"
    )
    for key, value in pitch_dist_widget_dict.items():
        if value:
            cfg.pitch[key] = value

def make_source_time_distribution_container(streamlit_element, cfg: ScapeConfig, expanded:bool = False):
    source_time_dist_widget_dict = uscg.make_distribution_select_container(
        streamlit_element,
        label = "Source Time Distribution",
        dist_dict = cfg.source_time,
        cfg = cfg,
        expandable = False,
        key = "source_time_dist_widget"
    )
    for key, value in source_time_dist_widget_dict.items():
        if value:
            cfg.source_time[key] = value

def make_snr_distribution_container(streamlit_element, cfg: ScapeConfig, expanded:bool = False):
    snr_dist_widget_dict = uscg.make_distribution_select_container(
        streamlit_element,
        label = "Signal-to-Noise Distribution",
        dist_dict = cfg.snr,
        cfg = cfg,
        expandable = False,
        key = "snr_dist_widget"
    )
    for key, value in snr_dist_widget_dict.items():
        if value:
            cfg.snr[key] = value

def make_general_settings_container(streamlit_element, cfg: ScapeConfig, expanded:bool = False):
    settings_general_container = streamlit_element.beta_expander("General", expanded = expanded).beta_container()
    cfg.n_soundscapes["value"] = uscg.make_slider_widget(
        settings_general_container, 
        "n soundscapes",
        cfg.n_soundscapes["min"],
        cfg.n_soundscapes["max"],
        cfg.n_soundscapes["value"],
        step = 10
    )

    cfg.ref_db = uscg.make_slider_widget(
        settings_general_container, 
        "ref db",
        -120,
        0,
        cfg.ref_db
    )

    cfg.duration = uscg.make_slider_widget(
        settings_general_container,
        "duration (s)",
        1.0,
        20.0,
        cfg.duration,
        step = 0.5
    )

    cfg.min_events = settings_general_container.slider("Min Events:", 0, 10, 0)
    cfg.max_events = settings_general_container.slider("Max Events:", 0, 10, 4)

    if cfg.min_events > cfg.max_events:
        settings_general_container.error("Min events must be <= max events ")

def make_labels_container(streamlit_element, cfg: ScapeConfig, expanded:bool = False):
    # Labels
    select_label_container = streamlit_element.beta_expander("Select Labels", expanded = expanded).beta_container()
    cfg.fg_labels_used = select_label_container.multiselect("Select from available foreground labels:", cfg.fg_labels, default = cfg.fg_labels)
    cfg.bg_labels_used = select_label_container.multiselect("Select from available background labels:", cfg.bg_labels, default = cfg.bg_labels)
