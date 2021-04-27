import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.transforms

class SimpleConvNet(nn.Module):
    def __init__(self, hparams:dict):
        super().__init__()

        self.melspec = torchaudio.transforms.MelSpectrogram(
                    sample_rate = hparams["sample_rate"],
                    n_fft = hparams["n_fft"],
                    win_length = hparams["n_fft"]//2,
                    hop_length = hparams["hop_len"],
                    n_mels = hparams["n_mels"],
                    normalized = True
                )
        self.amp_to_db = torchaudio.transforms.AmplitudeToDB(stype = "power", top_db = hparams["top_db"])
        
        self.bn0 = nn.BatchNorm2d(1)
        self.conv0 = nn.Conv2d(
            hparams["conv0"][0],
            hparams["conv0"][1], 
           ( hparams["conv0"][2], 
            hparams["conv0"][3])          
            )
        self.conv1 = nn.Conv2d(
            hparams["conv1"][0],
            hparams["conv1"][1], 
            (hparams["conv1"][2], 
            hparams["conv1"][3])        
            )
        self.conv2 = nn.Conv2d(
            hparams["conv2"][0],
            hparams["conv2"][1], 
            (hparams["conv2"][2], 
            hparams["conv2"][3])        
            )
        self.fc = nn.Linear(hparams["fc"][0], hparams["fc"][1])
        
        
        
    def forward(self, input):
        x = self.melspec(input)
        x = self.amp_to_db(x)
        
        x = self.bn0(x)
        x = self.conv0(x)
        x = F.relu(x)
        x = F.max_pool2d(x, (2,2))
        x = nn.Dropout(p = 0.2)(x)
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, (3,3))
        x = nn.Dropout(p = 0.2)(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, (3,3))
        x = nn.Dropout(p = 0.2)(x)
        
        x = torch.flatten(x, 1)
        out = self.fc(x)
        
        return out