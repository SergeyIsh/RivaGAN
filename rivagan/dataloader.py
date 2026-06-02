import os
from glob import glob
from random import randint
import numpy as np

import cv2
import torch
from torch.utils.data import DataLoader, Dataset


class VideoDataset(Dataset):
    # (3, seq_len, H, W) clips from *.avi / *.mp4 under root_dir

    def __init__(self, root_dir, crop_size, seq_len, max_crop_size=(360, 480)):
        self.seq_len = seq_len
        self.crop_size = crop_size
        self.max_crop_size = max_crop_size

        self.videos = []
        for ext in ["avi", "mp4"]:
            for path in glob(os.path.join(root_dir, "**/*.%s" % ext), recursive=True):
                cap = cv2.VideoCapture(path)
                nb_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                self.videos.append((path, nb_frames))

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, idx):
        path, nb_frames = self.videos[idx]
        if nb_frames < self.seq_len + 1:
            start_idx = 0
        else:
            start_idx = randint(0, nb_frames - self.seq_len - 1)

        cap = cv2.VideoCapture(path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_idx)
        ok, frame = cap.read()
        H, W = frame.shape[0], frame.shape[1]

        dy, dx = self.crop_size if self.crop_size else (H, W)
        dy = min(dy, H, self.max_crop_size[0])
        dx = min(dx, W, self.max_crop_size[1])
        x = randint(0, max(0, W - dx))
        y = randint(0, max(0, H - dy))

        frames = []
        for i in range(self.seq_len):
            if i > 0:
                ok, frame = cap.read()
            frames.append(frame[y : y + dy, x : x + dx].astype(np.float32))

        cap.release()
        t = torch.from_numpy(np.array(frames, dtype=np.float32)).permute(3, 0, 1, 2)
        return t / 127.5 - 1.0


def load_train_val(seq_len, batch_size, num_workers=16, dataset="hollywood2"):
    train = DataLoader(VideoDataset(
        "%s/train" % dataset,
        crop_size=(160, 160),
        seq_len=seq_len,
    ), shuffle=True, num_workers=num_workers, batch_size=batch_size, pin_memory=True)
    val = DataLoader(VideoDataset(
        "%s/val" % dataset,
        crop_size=False,
        seq_len=seq_len,
    ), shuffle=False, batch_size=1, pin_memory=True)
    return train, val
