import os
import random

import numpy as np
import torch
from torch.utils.data.dataloader import default_collate

from .audio import WaveReader


def make_dataloader(train=True,
                    path_kwargs=None,
                    num_workers=4,
                    chunk_length=4,
                    sample_rate=8000,
                    pin_memory=True,
                    batch_size=16):
    chunk_size = sample_rate * chunk_length

    dataset = Dataset(mix_scp=os.path.join(path_kwargs['scp_dir'], path_kwargs['mix_scp']),
                      ref_scp=os.path.join(path_kwargs['scp_dir'], path_kwargs['ref_scp']),
                      aux_scp=os.path.join(path_kwargs['scp_dir'], path_kwargs['aux_scp']),
                      ref_dur=path_kwargs['ref_dur'],
                      spk_list=path_kwargs['spk_list'],
                      sample_rate=sample_rate)

    return DataLoader(dataset,
                      train=train,
                      chunk_size=chunk_size,
                      batch_size=batch_size,
                      pin_memory=pin_memory,
                      num_workers=num_workers)


class Dataset(object):
    def __init__(self,
                 mix_scp=None,
                 ref_scp=None,
                 aux_scp=None,
                 ref_dur=None,
                 spk_list=None,
                 sample_rate=8000):
        self.sample_rate = sample_rate
        self.spk_list = self._load_spk(spk_list)

        self.mix = WaveReader(mix_scp, sample_rate=sample_rate)
        self.ref = WaveReader(ref_scp, sample_rate=sample_rate)
        self.aux = WaveReader(aux_scp, sample_rate=sample_rate)
        self.ref_dur = WaveReader(ref_dur, sample_rate=sample_rate)

    def _load_spk(self, spk_list_path):
        if spk_list_path is None:
            return []

        lines = open(spk_list_path).readlines()
        new_lines = []
        for line in lines:
            new_lines.append(line.strip())

        return new_lines

    def __len__(self):
        return len(self.mix)

    def __getitem__(self, index):
        key = self.mix.index_keys[index]
        mix = self.mix[key]
        ref = self.ref[key]
        aux = self.aux[key]

        target1_dur = self.ref_dur.index_dict[key.split('_')[0]]
        target2_dur = self.ref_dur.index_dict[key.split('_')[2]]
        end_idx = int(min(float(target1_dur), float(target2_dur)) * self.sample_rate)

        mix = mix[:end_idx].astype(np.float32)
        ref = ref[:end_idx].astype(np.float32)
        aux = aux.astype(np.float32)

        mix_std_, aux_std_ = np.std(mix), np.std(aux)
        mix, aux = mix / mix_std_, aux / aux_std_

        spk_idx = self.spk_list.index(key.split('_')[-1][0:3])

        return {"mix": mix,
                "ref": ref,
                "aux": aux,
                "aux_len": len(aux),
                "spk_idx": spk_idx,
                "mix_scale": mix_std_}


class ChunkSplitter(object):
    def __init__(self, chunk_size, train=True, least=16000):
        self.chunk_size = chunk_size
        self.least = least
        self.train = train

    def _make_chunk(self, eg, s):
        chunk = dict()
        chunk["mix"] = eg["mix"][s:s + self.chunk_size]
        chunk["ref"] = eg["ref"][s:s + self.chunk_size]
        chunk["aux"] = eg["aux"]
        chunk["aux_len"] = eg["aux_len"]
        chunk["valid_len"] = int(self.chunk_size)
        chunk["spk_idx"] = eg["spk_idx"]
        chunk["mix_scale"] = eg["mix_scale"]

        return chunk

    def split(self, eg):
        N = eg["mix"].size

        if N < self.least:
            return []
        chunks = []

        if N < self.chunk_size:
            P = self.chunk_size - N
            chunk = dict()
            chunk["mix"] = np.pad(eg["mix"], (0, P), "constant")
            chunk["ref"] = np.pad(eg["ref"], (0, P), "constant")
            chunk["aux"] = eg["aux"]
            chunk["aux_len"] = eg["aux_len"]
            chunk["valid_len"] = int(N)
            chunk["spk_idx"] = eg["spk_idx"]
            chunk["mix_scale"] = eg["mix_scale"]
            chunks.append(chunk)
        else:
            s = random.randint(0, N % self.least) if self.train else 0
            while True:
                if s + self.chunk_size > N:
                    break
                chunk = self._make_chunk(eg, s)
                chunks.append(chunk)
                s += self.least

        return chunks


class DataLoader(object):
    def __init__(self,
                 dataset,
                 num_workers=4,
                 chunk_size=32000,
                 batch_size=16,
                 train=True,
                 pin_memory=True):
        self.batch_size = batch_size
        self.train = train
        self.splitter = ChunkSplitter(chunk_size,
                                      train=train,
                                      least=chunk_size // 2)

        self.eg_loader = torch.utils.data.DataLoader(dataset,
                                                     batch_size=batch_size // 2,
                                                     num_workers=num_workers,
                                                     shuffle=train,
                                                     pin_memory=pin_memory,
                                                     collate_fn=self._collate)

    def _collate(self, batch):
        chunk = []
        for eg in batch:
            chunk += self.splitter.split(eg)

        return chunk

    def _pad_aux(self, chunk_list):
        lens_list = []
        for chunk_item in chunk_list:
            lens_list.append(chunk_item['aux_len'])
        max_len = np.max(lens_list)

        for idx in range(len(chunk_list)):
            P = max_len - len(chunk_list[idx]["aux"])
            chunk_list[idx]["aux"] = np.pad(chunk_list[idx]["aux"], (0, P), "constant")

        return chunk_list

    def _merge(self, chunk_list):
        N = len(chunk_list)
        if self.train:
            random.shuffle(chunk_list)
        blist = []
        for s in range(0, N - self.batch_size + 1, self.batch_size):
            batch = default_collate(self._pad_aux(chunk_list[s:s + self.batch_size]))
            blist.append(batch)
        rn = N % self.batch_size

        return blist, chunk_list[-rn:] if rn else []

    def __iter__(self):
        chunk_list = []
        for chunks in self.eg_loader:
            chunk_list += chunks
            batch, chunk_list = self._merge(chunk_list)
            for obj in batch:
                yield obj
