import streamlit as st
import numpy as np
import pathlib
import utils_soundscape_generator as uscg
import config_soundscape_generator
import scaper
import json
from icecream import ic

############
import importlib
importlib.reload(config_soundscape_generator)
#############
# global config in user/.streamlit/config.toml

cfg = config_soundscape_generator.ScapeConfig()
def make_scaper(cfg:config_soundscape_generator.ScapeConfig, fg_path:pathlib.Path, bg_path:pathlib.Path) -> scaper.Scaper:
    sc = scaper.Scaper(
        cfg.duration,
        fg_path.as_posix(),
        bg_path.as_posix(),
        cfg.protected_labels,
        cfg.seed
    )
    return sc


# Settings
data_path = pathlib.Path("data")
export_path = pathlib.Path("soundscapes")

# SIDEBAR
st.sidebar.subheader("Paths")

data_path = pathlib.Path(st.sidebar.text_input(label = 'Data Path:', value = data_path.as_posix()))
fg_path = data_path.joinpath("foreground")
bg_path = data_path.joinpath("background")

if fg_path.exists():
    cfg.fg_labels = uscg.get_labels_of_path(fg_path)
else: 
    cfg.fg_labels = []
    st.sidebar.error(f"Path '{fg_path}' does not exist!")
if bg_path.exists():
    cfg.bg_labels = uscg.get_labels_of_path(bg_path)
else: 
    cfg.bg_labels = []
    st.sidebar.error(f"Path '{bg_path}' does not exist!")

export_path = pathlib.Path(st.sidebar.text_input(label = "Export Path:", value = export_path.as_posix()))

sb_col_1, sb_col_2, sb_col_3 = st.sidebar.beta_columns(3)
st.sidebar.subheader("Soundscape Settings")
cfg.n_soundscapes["value"] = uscg.make_slider_widget(
    sb_col_1, 
    "n soundscapes",
    cfg.n_soundscapes["min"],
    cfg.n_soundscapes["max"],
    cfg.n_soundscapes["value"]
)
# sb_col_1.slider(
#     label = "n soundscapes",
#     min_value = cfg.n_soundscapes["min"],
#     max_value=cfg.n_soundscapes["max"],
#     value = cfg.n_soundscapes["value"],
#     # format = "%f"
    # )
cfg.ref_db = sb_col_1.slider(
    "ref db",
    min_value = -120,
    max_value = 0,
    value = cfg.ref_db
)
cfg.duration = sb_col_1.slider(
    label = "duration (s)",
    min_value = 1.0,
    max_value = 20.0,
    value = cfg.duration,
    step = 0.5
)

# MAIN
st.write(cfg.make_distribution_dict("normal"))


# Left Col
col_1, col_2 = st.beta_columns(2)
col_1.header("Create Soundscapes")
col_1.write("To generate new Soundscapes, choose the data and export path on the right. Then, use the options on the left to specify your settings.")
col_1.subheader("Labels")
col_1.multiselect("Select from available foreground labels:", cfg.fg_labels, default = cfg.fg_labels)
col_1.multiselect("Select from available background labels:", cfg.bg_labels, default = cfg.bg_labels)
col_1.write("When you are done, click 'Start!'")


if col_1.button("Start!"):
    col_1.write(cfg.n_soundscapes["value"])
    progress_generator = col_1.progress(0.0)
    generator_status_text = col_1.empty()
    col_1.write("0%")
    # Make Scaper
    sc = make_scaper(cfg, fg_path, bg_path)
    current_export_dir = uscg.make_export_folder(export_path)
    with open("test.json", "w") as f:
        json.dump(cfg.get_params(), f)

    for i in range(cfg.n_soundscapes["value"]):
        sc = uscg.reset_scaper(sc)
        sc = uscg.add_bg(sc)
        _n_events = np.random.randint(cfg.min_events, cfg.max_events+1)

        for n in range(_n_events):
            sc = uscg.add_event(sc, cfg)
            audiofile, jamsfile, txtfile = uscg.make_export_filepaths(n, current_export_dir)

            sc.generate(
                audiofile,
                jamsfile,
                allow_repeated_label=True,
                allow_repeated_source=True,
                reverb=cfg.reverb, # TO SETTINGS
                disable_sox_warnings=True,
                no_audio=False,
                txt_path=txtfile
            )

        percent_progress = (i+1)/cfg.n_soundscapes["value"]
        progress_generator.progress(percent_progress)
        generator_status_text.text(f"{percent_progress*100:.2f}%")

else:
    pass

# Right Col
col_2.subheader("Listen to your Soundscapes")
audio_file_dir = pathlib.Path(col_2.selectbox('First, Select Origin Folder', [_ for _ in export_path.iterdir() if _.is_dir()]))
audio_file = col_2.file_uploader(
    label = 'Choose Audio File:',
    accept_multiple_files = False,
    type = ["wav"]
)
if audio_file is not None:
    audio_file_name = audio_file.name
    event_df = uscg.read_event_table_for_path(audio_file_name, export_path.joinpath("2021 04 13 10h 30min"))
    col_2.audio(audio_file)
    col_2.write(event_df)

st.write(cfg.fg_labels)