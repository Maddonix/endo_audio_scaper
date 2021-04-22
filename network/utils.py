import math, random
import torch
import torchaudio.transforms
from IPython.display import Audio
import pathlib
import pandas as pd
from tqdm.auto import tqdm
from sklearn.preprocessing import MultiLabelBinarizer
from torch.utils.data import Dataset

class SoundDataSet(Dataset):
    def __init__(self, df, data_path, duration, sample_rate, top_db, n_mels, hop_len, n_fft,**kwargs):
        self.df = df
        self.data_path = pathlib.Path(data_path)
        self.sr = sample_rate
        self.duration = duration,
        self.top_db = top_db
        self.n_mels = n_mels
        self.hop_len = hop_len
        self.n_fft = n_fft
        self.melspec = torchaudio.transforms.MelSpectrogram(
            sample_rate = sample_rate,
            n_fft = n_fft,
            win_length = n_fft//2,
            hop_length = hop_len,
            n_mels = n_mels,
            normalized = True
        )
        self.amp_to_db = torchaudio.transforms.AmplitudeToDB(stype = "power", top_db = top_db)
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.loc[idx, :]
        audiofile_path = self.data_path.joinpath(row["relative_path"])
        class_ids = row["class_ids"]
        
        sig, sr = torchaudio.load(audiofile_path)
        if sr is not self.sr:
            sig, sr = resample((sig, sr), self.sr)
        
        return sig, class_ids
        
    def get_index(self,idx):
        row = self.df.loc[idx, :]
        audiofile_path = self.data_path.joinpath(row["relative_path"])
        class_ids = row["class_ids"]
        
        sig, sr = torchaudio.load(audiofile_path)
        if sr is not self.sr:
            sig, sr = resample((sig, sr), self.sr)
        spec = self.melspec(sig)
        spec = self.amp_to_db(spec)

        return sig, spec, class_ids, sr, audiofile_path

def read_label_txt(path:pathlib.Path) -> pd.DataFrame:
    colnames = ["onset", "offset", "label"]
    df = pd.read_csv(path, sep = "\t", names = ["onset", "offset", "label"], index_col = False)
    return df

def get_dataset_dataframe(data_path):
    data_path = pathlib.Path(data_path)
    wav_paths = [_ for _ in data_path.iterdir() if _.suffix == ".wav"]
    label_paths = [_.with_suffix(".txt") for _ in wav_paths]
    labels = [read_label_txt(_)["label"].to_list() for _ in tqdm(label_paths)]
    mlb = MultiLabelBinarizer()
    label_array = mlb.fit_transform(labels)
    classes = mlb.classes_

    df = pd.DataFrame(data = {
        "relative_path": [_.name for _ in wav_paths],
        "class_ids": [_ for _ in label_array]
    })

    return df, classes

def get_mfcc(data, config):
    return torchaudio.transforms.MFCC(
            sample_rate = config["sample_rate"],
            n_mfcc = config["n_mfcc"],
            log_mels = True,
            melkwargs = {
                "n_mels": config["n_mels"],
                "n_fft": config["n_fft"],
                "normalized": True
            }
        )(data)[0,...]

def get_melspec(data, sample_rate, n_fft, hop_len, n_mels, top_db, **kwargs):
    spec = torchaudio.transforms.MelSpectrogram(
        sample_rate,
        n_fft,
        hop_len,
        n_mels,
        normalized = True
    )(data)
    return torchaudio.transforms.AmplitudeToDB(top_db)(spec)[0,...]

def resample(aud, newsr):
    sig, sr = aud

    if sr == newsr:
        # Nothing to do
        return aud

    num_channels = sig.shape[0]
    # Resample first channel
    resig = torchaudio.transforms.Resample(sr, newsr)(sig[:1, :])
    if num_channels > 1:
        # Resample the second channel and merge both channels
        retwo = torchaudio.transforms.Resample(sr, newsr)(sig[1:, :])
        resig = torch.cat([resig, retwo])

    return (resig, newsr)

def show_spectrogram(spec):
    '''
    Expects Tensor
    '''
    spec = spec.numpy()
    if len(spec.shape) == 3:
        spec = spec[0,...]
        
    librosa.display.specshow(spec)

# class AudioUtil:
#     # ----------------------------
#     # Load an audio file. Return the signal as a tensor and the sample rate
#     # ----------------------------
#     @staticmethod
#     def open(audio_file):
#         sig, sr = torchaudio.load(audio_file)
#         return (sig, sr)

#     # ----------------------------
#     # Convert the given audio to the desired number of channels
#     # ----------------------------
#     @staticmethod
#     def rechannel(aud, new_channel):
#         sig, sr = aud

#         if sig.shape[0] == new_channel:
#             # Nothing to do
#             return aud

#         if new_channel == 1:
#             # Convert from stereo to mono by selecting only the first channel
#             resig = sig[:1, :]
#         else:
#             # Convert from mono to stereo by duplicating the first channel
#             resig = torch.cat([sig, sig])

#         return (resig, sr)

#     # ----------------------------
#     # Since Resample applies to a single channel, we resample one channel at a time
#     # ----------------------------
#     @staticmethod
#     def resample(aud, newsr):
#         sig, sr = aud

#         if sr == newsr:
#             # Nothing to do
#             return aud

#         num_channels = sig.shape[0]
#         # Resample first channel
#         resig = torchaudio.transforms.Resample(sr, newsr)(sig[:1, :])
#         if num_channels > 1:
#             # Resample the second channel and merge both channels
#             retwo = torchaudio.transforms.Resample(sr, newsr)(sig[1:, :])
#             resig = torch.cat([resig, retwo])

#         return (resig, newsr)

#     # ----------------------------
#     # Pad (or truncate) the signal to a fixed length 'max_ms' in milliseconds
#     # ----------------------------
#     @staticmethod
#     def pad_trunc(aud, max_ms):
#         sig, sr = aud
#         num_rows, sig_len = sig.shape
#         max_len = sr // 1000 * max_ms

#         if sig_len > max_len:
#             # Truncate the signal to the given length
#             sig = sig[:, :max_len]

#         elif sig_len < max_len:
#             # Length of padding to add at the beginning and end of the signal
#             pad_begin_len = random.randint(0, max_len - sig_len)
#             pad_end_len = max_len - sig_len - pad_begin_len

#             # Pad with 0s
#             pad_begin = torch.zeros((num_rows, pad_begin_len))
#             pad_end = torch.zeros((num_rows, pad_end_len))

#             sig = torch.cat((pad_begin, sig, pad_end), 1)

#         return (sig, sr)

#     # ----------------------------
#     # Shifts the signal to the left or right by some percent. Values at the end
#     # are 'wrapped around' to the start of the transformed signal.
#     # ----------------------------
#     @staticmethod
#     def time_shift(aud, shift_limit):
#         sig, sr = aud
#         _, sig_len = sig.shape
#         shift_amt = int(random.random() * shift_limit * sig_len)
#         return (sig.roll(shift_amt), sr)

#     # ----------------------------
#     # Generate a Spectrogram
#     # ----------------------------
#     @staticmethod
#     def spectro_gram(aud, n_mels=64, n_fft=1024, hop_len=None):
#         sig, sr = aud
#         top_db = 80

#         # spec has shape [channel, n_mels, time], where channel is mono, stereo etc
#         spec = transforms.MelSpectrogram(
#             sr, n_fft=n_fft, hop_length=hop_len, n_mels=n_mels
#         )(sig)

#         # Convert to decibels
#         spec = transforms.AmplitudeToDB(top_db=top_db)(spec)
#         return spec

#     # ----------------------------
#     # Augment the Spectrogram by masking out some sections of it in both the frequency
#     # dimension (ie. horizontal bars) and the time dimension (vertical bars) to prevent
#     # overfitting and to help the model generalise better. The masked sections are
#     # replaced with the mean value.
#     # ----------------------------
#     @staticmethod
#     def spectro_augment(spec, max_mask_pct=0.1, n_freq_masks=1, n_time_masks=1):
#         _, n_mels, n_steps = spec.shape
#         mask_value = spec.mean()
#         aug_spec = spec

#         freq_mask_param = max_mask_pct * n_mels
#         for _ in range(n_freq_masks):
#             aug_spec = transforms.FrequencyMasking(freq_mask_param)(
#                 aug_spec, mask_value
#             )

#         time_mask_param = max_mask_pct * n_steps
#         for _ in range(n_time_masks):
#             aug_spec = transforms.TimeMasking(time_mask_param)(aug_spec, mask_value)

#         return aug_spec


# from torch.utils.data import DataLoader, Dataset, random_split
# import torchaudio

# # ----------------------------
# # Sound Dataset
# # ----------------------------
# class SoundDS(Dataset):
#     def __init__(self, df, data_path:pathlib.Path):
#         self.df = df
#         self.data_path = data_path
#         self.duration = 10000
#         self.sr = 44100
#         self.channel = 2
#         self.shift_pct = 0.4

#     # ----------------------------
#     # Number of items in dataset
#     # ----------------------------
#     def __len__(self):
#         return len(self.df)

#     # ----------------------------
#     # Get i'th item in dataset
#     # ----------------------------
#     def __getitem__(self, idx):
#         # Absolute file path of the audio file - concatenate the audio directory with
#         # the relative path
#         audio_file = self.data_path.joinpath(self.df.loc[idx, "relative_path"])
#         # Get the Class ID
#         class_id = self.df.loc[idx, "class_ids"]

#         aud = AudioUtil.open(audio_file)
#         # Some sounds have a higher sample rate, or fewer channels compared to the
#         # majority. So make all sounds have the same number of channels and same
#         # sample rate. Unless the sample rate is the same, the pad_trunc will still
#         # result in arrays of different lengths, even though the sound duration is
#         # the same.
#         reaud = AudioUtil.resample(aud, self.sr)
#         rechan = AudioUtil.rechannel(reaud, self.channel)

#         dur_aud = AudioUtil.pad_trunc(rechan, self.duration)
#         shift_aud = AudioUtil.time_shift(dur_aud, self.shift_pct)
#         sgram = AudioUtil.spectro_gram(shift_aud, n_mels=64, n_fft=1024, hop_len=None)
#         aug_sgram = AudioUtil.spectro_augment(
#             sgram, max_mask_pct=0.1, n_freq_masks=2, n_time_masks=2
#         )

#         return aug_sgram, class_id
