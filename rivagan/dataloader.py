import os
from glob import glob
from random import randint
import numpy as np

import cv2
import torch
from torch.utils.data import DataLoader, Dataset


class VideoDataset(Dataset):
    """
    Given a folder of *.avi video files organized as shown below, this dataset
    selects randomly crops the video to `crop_size` and returns a random
    continuous sequence of `seq_len` frames of shape.

        /root_dir
            1.avi
            2.avi

    The output has shape (3, seq_len, crop_size[0], crop_size[1]).
    """

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
        # Select time index
        path, nb_frames = self.videos[idx]
        
        # Защита от недостаточного количества кадров
        if nb_frames < self.seq_len + 1:
            start_idx = 0
        else:
            start_idx = randint(0, nb_frames - self.seq_len - 1)

        # Select space index
        cap = cv2.VideoCapture(path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_idx)
        
        # Проверка успешности чтения первого кадра
        ok, frame = cap.read()
        if not ok or frame is None:
            cap.release()
            # Возврат черного кадра в случае ошибки
            dummy_frame = np.zeros((self.crop_size[0], self.crop_size[1], 3), dtype=np.float32)
            frames = [dummy_frame.copy() for _ in range(self.seq_len)]
            x = torch.from_numpy(np.array(frames, dtype=np.float32))
            x = x.permute(3, 0, 1, 2)
            return x / 127.5 - 1.0
        
        H, W = frame.shape[0], frame.shape[1]
        
        # Вычисление безопасных координат обрезки
        dy, dx = self.crop_size if self.crop_size else (H, W)
        dy = min(dy, H, self.max_crop_size[0])
        dx = min(dx, W, self.max_crop_size[1])
        x = randint(0, max(0, W - dx))
        y = randint(0, max(0, H - dy))

        # Read frames and normalize to [-1.0, 1.0]
        frames = []
        last_good_frame = frame[y:y+dy, x:x+dx].astype(np.float32)
        
        for _ in range(self.seq_len):
            if len(frames) > 0:
                ok, frame = cap.read()
            
            # Проверка успешности чтения кадра
            if not ok or frame is None:
                # Использование последнего хорошего кадра при ошибке
                current_frame = last_good_frame.copy()
            else:
                # Обрезка с защитой от выхода за границы
                y_end = min(y + dy, frame.shape[0])
                x_end = min(x + dx, frame.shape[1])
                current_frame = frame[y:y_end, x:x_end].astype(np.float32)
                
                # Дополнение до нужного размера если кадр меньше
                if current_frame.shape[0] < dy or current_frame.shape[1] < dx:
                    pad_h = dy - current_frame.shape[0]
                    pad_w = dx - current_frame.shape[1]
                    current_frame = np.pad(current_frame, ((0, pad_h), (0, pad_w), (0, 0)), mode='edge')
                
                last_good_frame = current_frame.copy()
            
            frames.append(current_frame)
        
        cap.release()
        
        # Преобразование в тензор
        x = torch.from_numpy(np.array(frames, dtype=np.float32))
        x = x.permute(3, 0, 1, 2)
        return x / 127.5 - 1.0


def load_train_val(seq_len, batch_size, num_workers=16, dataset="hollywood2"):
    """
    This returns two dataloaders correponding to the train and validation sets. Each
    iterator yields tensors of shape (N, 3, L, H, W) where N is the batch size, L is
    the sequence length, and H and W are the height and width of the frame.

    The batch size is always 1 in the validation set. The frames are always cropped
    to (128, 128) windows in the training set. The frames in the validation set are
    not cropped if they are smaller than 360x480; otherwise, they are cropped so the
    maximum returned size is 360x480.
    """
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
