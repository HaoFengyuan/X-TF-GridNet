import os

import numpy as np
import soundfile as sf
import torch

MAX_INT16 = np.iinfo(np.int16).max


def write_wav(fname, samps, fs=16000):
    fdir = os.path.dirname(fname)
    if fdir and not os.path.exists(fdir):
        os.makedirs(fdir)

    sf.write(fname, samps, fs)


def read_wav(fname, return_rate=False):
    samps, samp_rate = sf.read(fname)
    if return_rate:
        return samp_rate, samps

    return samps


def get_comSTFT(egs, signal_configs, device):
    win_size = int(signal_configs['sr'] * signal_configs['win_size'])
    win_shift = int(signal_configs['sr'] * signal_configs['win_shift'])

    batch_mix_stft = torch.stft(egs['mix'],
                                n_fft=signal_configs['fft_num'],
                                hop_length=win_shift,
                                win_length=win_size,
                                return_complex=True,
                                window=torch.hann_window(win_size).to(device))  # (B,F,T,2)
    batch_mix_stft = torch.view_as_real(batch_mix_stft)
    batch_mix_mag, batch_mix_phase = torch.norm(batch_mix_stft, dim=-1) ** signal_configs['beta'], \
        torch.atan2(batch_mix_stft[..., -1], batch_mix_stft[..., 0])
    egs['mix'] = torch.stack((batch_mix_mag * torch.cos(batch_mix_phase),
                              batch_mix_mag * torch.sin(batch_mix_phase)), dim=-1).permute(0, 3, 2, 1)

    batch_aux_stft = torch.stft(egs['aux'],
                                n_fft=signal_configs['fft_num'],
                                hop_length=win_shift,
                                win_length=win_size,
                                return_complex=True,
                                window=torch.hann_window(win_size).to(device))  # (B,F,T,2)
    batch_aux_stft = torch.view_as_real(batch_aux_stft)
    batch_aux_mag, batch_aux_phase = torch.norm(batch_aux_stft, dim=-1) ** signal_configs['beta'], \
        torch.atan2(batch_aux_stft[..., -1], batch_aux_stft[..., 0])
    egs['aux'] = torch.stack((batch_aux_mag * torch.cos(batch_aux_phase),
                              batch_aux_mag * torch.sin(batch_aux_phase)), dim=-1).permute(0, 3, 2, 1)

    egs["aux_len"] = torch.tensor([(length - win_size + win_size) // win_shift + 1 for length in egs["aux_len"]],
                                  dtype=torch.int, device=device)  # center case
    egs['mix_scale'] = egs['mix_scale'][:, None]

    return egs


def parse_scripts(scp_path, value_processor=lambda x: x, num_tokens=2):
    scp_dict = dict()
    line = 0
    
    with open(scp_path, "r") as f:
        for raw_line in f:
            scp_tokens = raw_line.strip().split()
            line += 1
            if 2 <= num_tokens != len(scp_tokens) or len(
                    scp_tokens) < 2:
                raise RuntimeError("For {}, format error in line[{:d}]: {}".format(scp_path, line, raw_line))
            if num_tokens == 2:
                key, value = scp_tokens
            else:
                key, value = scp_tokens[0], scp_tokens[1:]
            if key in scp_dict:
                raise ValueError("Duplicated key \'{0}\' exists in {1}".format(key, scp_path))
            scp_dict[key] = value_processor(value)

    return scp_dict


class Reader(object):
    def __init__(self, scp_path, value_processor=lambda x: x):
        self.index_dict = parse_scripts(
            scp_path, value_processor=value_processor, num_tokens=2)
        self.index_keys = list(self.index_dict.keys())

    def _load(self, key):
        return self.index_dict[key]

    def __len__(self):
        return len(self.index_dict)

    def __contains__(self, key):
        return key in self.index_dict

    def __iter__(self):
        for key in self.index_keys:
            yield key, self._load(key)

    def __getitem__(self, index):
        if type(index) not in [int, str]:
            raise IndexError("Unsupported index type: {}".format(type(index)))

        if type(index) == int:
            num_utts = len(self.index_keys)
            if index >= num_utts or index < 0:
                raise KeyError("Interger index out of range, {:d} vs {:d}".format(index, num_utts))
            index = self.index_keys[index]

        if index not in self.index_dict:
            raise KeyError("Missing utterance {}!".format(index))

        return self._load(index)


class WaveReader(Reader):
    def __init__(self, wav_scp, sample_rate=None):
        super(WaveReader, self).__init__(wav_scp)
        self.samp_rate = sample_rate

    def _load(self, key):
        samp_rate, samps = read_wav(self.index_dict[key], return_rate=True)

        if self.samp_rate is not None and samp_rate != self.samp_rate:
            raise RuntimeError("SampleRate mismatch: {:d} vs {:d}".format(samp_rate, self.samp_rate))

        return samps
