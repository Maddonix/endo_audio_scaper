from torch import nn
import torchaudio.transforms

class AudioPreprocess(nn.Module):
    """Module for Audio Preprocessing. Generates Melspectrogram and turns it into db representation.
    """
    def __init__(self, hparams:dict):
        super().__init__()

        self.melspec = torchaudio.transforms.MelSpectrogram(
                    sample_rate = hparams["sr"],
                    n_fft = hparams["n_fft"],
                    win_length = hparams["n_fft"]//2,
                    hop_length = hparams["hop_len"],
                    n_mels = hparams["n_mels"],
                    normalized = True
                )

        self.amp_to_db = torchaudio.transforms.AmplitudeToDB(stype = "power", top_db = hparams["top_db"])
        self.model = nn.Sequential(
            self.melspec,
            self.amp_to_db
        )

    def forward(self, x):
        x = self.model(x)
        return x