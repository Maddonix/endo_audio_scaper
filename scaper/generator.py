import streamlit as st
import streamlit_sidebar as utils_sidebar
import streamlit_soundscape_settings as utils_settings
import numpy as np
import pathlib
import utils_soundscape_generator as uscg
import config_soundscape_generator
import scaper
import json
from icecream import ic

# global config in user/.streamlit/config.toml
st.set_page_config(page_title="Soundscape Generator", page_icon=None, layout='wide', initial_sidebar_state='auto')


cfg = config_soundscape_generator.ScapeConfig()
def make_scaper(cfg:config_soundscape_generator.ScapeConfig, fg_path:pathlib.Path, bg_path:pathlib.Path) -> scaper.Scaper:
    sc = scaper.Scaper(
        cfg.duration,
        fg_path.as_posix(),
        bg_path.as_posix(),
        cfg.protected_labels,
        cfg.seed
    )
    sc.ref_db = cfg.ref_db
    return sc


sidebar_dict = utils_sidebar.make_sidebar(st, cfg)
fg_path = sidebar_dict["fg_path"]
bg_path = sidebar_dict["bg_path"]
data_path = sidebar_dict["data_path"]
export_path = sidebar_dict["export_path"]

# MAIN
# Choose if BG or FG Generation
utils_settings.make_soundscape_settings_container(st, cfg)

st.header("Create Soundscapes")
st.write("To generate new Soundscapes, choose the data and export path on the right. Then, use the options on the left to specify your settings.")

# Start Generator
st.write("When you are done, click 'Start!'")

if st.button("Start!"):
    progress_generator = st.progress(0.0)
    generator_status_text = st.empty()
    # Make Scaper
    sc = make_scaper(cfg, fg_path, bg_path)
    current_export_dir = uscg.make_export_folder(export_path)
    if cfg.soundscape_type == "background":
        path_cfg_json = current_export_dir.parent.parent.joinpath(f"{current_export_dir.name}_cfg.json")
    elif cfg.soundscape_type == "foreground": 
        path_cfg_json = current_export_dir.parent.joinpath(f"{current_export_dir.name}_cfg.json")
    with open(path_cfg_json, "w") as f:
        json.dump(cfg.get_params(), f)

    for i in range(cfg.n_soundscapes["value"]):
        sc = uscg.reset_scaper(sc)
        if cfg.soundscape_type == "foreground":
            sc = uscg.add_bg(sc, cfg)

        _n_events = np.random.randint(cfg.min_events, cfg.max_events+1)
        for n in range(_n_events):
            sc = uscg.add_event(sc, cfg)
        audiofile, jamsfile, txtfile = uscg.make_export_filepaths(i, current_export_dir)
        if cfg.soundscape_type == "background":
            jamsfile = None
            txtfile = None

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
st.subheader("Listen to your Soundscapes")
soundscape_folders = [_ for _ in export_path.iterdir() if _.is_dir()]
if len(soundscape_folders) > 0:
    audio_file_dir = pathlib.Path(st.selectbox('First, Select Origin Folder', [_ for _ in export_path.iterdir() if _.is_dir()]))
    audio_file = st.file_uploader(
        label = 'Choose Audio File:',
        accept_multiple_files = False,
        type = ["wav"]
    )
    if audio_file is not None:
        audio_file_name = audio_file.name
        st.audio(audio_file)

        if cfg.soundscape_type == "foreground":
            event_df = uscg.read_event_table_for_path(audio_file_name, audio_file_dir)
            st.write(event_df)
else: 
    st.write("No soundscape folders found")