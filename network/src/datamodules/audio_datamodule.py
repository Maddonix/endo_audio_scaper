from typing import Optional, Tuple

from pytorch_lightning import LightningDataModule
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split
import os
import pathlib
import pandas as pd
from tqdm.auto import tqdm
from sklearn.preprocessing import MultiLabelBinarizer
from src.datamodules.datasets.audio_dataset import AudioDataset

from torch.utils.data import Dataset
import torchaudio

class AudioDataModule(LightningDataModule):
    """
    Example of LightningDataModule for MNIST dataset.

    A DataModule implements 5 key methods:
        - prepare_data (things to do on 1 GPU/TPU, not on every GPU/TPU in distributed mode)
        - setup (things to do on every accelerator in distributed mode)
        - train_dataloader (the training dataloader)
        - val_dataloader (the validation dataloader(s))
        - test_dataloader (the test dataloader(s))

    This allows you to share a full dataset without explaining how to download,
    split, transform and process the data

    Read the docs:
        https://pytorch-lightning.readthedocs.io/en/latest/extensions/datamodules.html
    """

    def __init__(
        self,
        classes:[],
        data_dir: str = "data/",
        train_val_test_split: Tuple[float, float, float] = (0.8, 0.1, 0.1),
        batch_size: int = 64,
        num_workers: int = 5,
        pin_memory: bool = False,
        sample_rate: int = 41000,
        duration: int = 10000, # in ms
        **kwargs,
    ):
        super().__init__()

        self.classes = classes
        self.data_path = data_dir
        self.sample_rate = sample_rate
        self.duration = duration
        self.train_val_test_split = train_val_test_split
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        # self.transforms = transforms.Compose(
        #     [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        # )

        # self.dims is returned when you call datamodule.size()
        self.dims = (1, self.duration * self.sample_rate)

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

    def setup(self, stage: Optional[str] = None):
        """Load data. Set variables: self.data_train, self.data_val, self.data_test."""
        dataset = AudioDataset(self.data_path, self.sample_rate, self.classes)
        self.classes = dataset.classes
        self.data_train, self.data_val, self.data_test = random_split(
            dataset, [int(_ * len(dataset)) for _ in self.train_val_test_split]
        )

    def train_dataloader(self):
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
        )


# class AudioDataset(Dataset):
#     """Dataclass to load wavfiles from folder."""

#     def __init__(self, dir: str, sr: int):
#         dir = pathlib.Path(dir)
#         assert dir.exists()
#         self.data_path = dir
#         self.df, self.classes = self.get_dataset_dataframe()

#     def __getitem__(self, idx: int):
#         row = self.df.loc[idx, :]
#         audiofile_path = self.data_path.joinpath(row["relative_path"])
#         class_ids = row["class_ids"]
        
#         sig, sr = torchaudio.load(audiofile_path)
#         if sr is not self.sr:
#             sig, sr = resample((sig, sr), self.sr)
        
#         return sig, class_ids

#     def __len__(self):
#         return len(self.df)


#     def get_dataset_dataframe(self):
#         wav_paths = [_ for _ in self.data_path.iterdir() if _.suffix == ".wav"]
#         label_paths = [_.with_suffix(".txt") for _ in wav_paths]
#         labels = [self.read_label_txt(_)["label"].to_list() for _ in tqdm(label_paths)]
#         mlb = MultiLabelBinarizer()
#         label_array = mlb.fit_transform(labels)
#         classes = mlb.classes_

#         df = pd.DataFrame(data = {
#             "relative_path": [_.name for _ in wav_paths],
#             "class_ids": [_ for _ in label_array]
#         })

#         return df, classes

#     def read_label_txt(self, path:pathlib.Path) -> pd.DataFrame:
#         colnames = ["onset", "offset", "label"]
#         df = pd.read_csv(path, sep = "\t", names = ["onset", "offset", "label"], index_col = False)
#         return df