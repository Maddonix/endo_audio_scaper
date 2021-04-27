import pathlib
import utils_soundscape_generator as uscg

def make_sidebar(st, cfg):
    st.sidebar.subheader("Paths")
    cfg.soundscape_type = st.sidebar.selectbox("Type:", ["foreground", "background"], index = 0)

    data_path = pathlib.Path("data")
    if cfg.soundscape_type == "foreground":
        export_path = pathlib.Path("soundscapes")
        data_path = data_path.joinpath("foreground_generation")
    elif cfg.soundscape_type == "background":
        export_path = data_path.joinpath("foreground_generation/background")
        data_path = data_path.joinpath("background_generation")   

    data_path = pathlib.Path(st.sidebar.text_input(label = 'Data Path:', value = data_path.as_posix()))
    fg_path = data_path.joinpath("foreground")
    bg_path = data_path.joinpath("background")

    if cfg.soundscape_type == "foreground":
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
    elif cfg.soundscape_type == "background":
        if fg_path.exists():
            cfg.fg_labels = uscg.get_labels_of_path(fg_path)
        else: 
            cfg.fg_labels = []
            st.sidebar.error(f"Path '{fg_path}' does not exist!")
        cfg.bg_labels = []

    export_path = pathlib.Path(st.sidebar.text_input(label = "Export Path:", value = export_path.as_posix()))
    
    sidebar_dict = {
        "fg_path": fg_path,
        "bg_path": bg_path,
        "data_path": data_path,
        "export_path": export_path
    }

    return sidebar_dict