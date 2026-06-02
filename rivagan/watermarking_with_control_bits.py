import os
from typing import Iterable, Optional, Sequence, Union

import cv2
import numpy as np
import torch
from tqdm import tqdm


def _torch_load_model(path: str):
    return torch.load(path, weights_only=False)


def _model_for_data_dim(models: Sequence, data_dim: int):
    want = int(data_dim)
    matches = [m for m in models if int(m.data_dim) == want]
    if len(matches) != 1:
        found = [int(m.data_dim) for m in models]
        raise ValueError("data_dim=%d: expected one model, got %s" % (want, found))
    return matches[0]


def _validated_bch_code(data_dim: int, bch_n: int, bch_k: int, bch_t: int):
    import galois

    code = galois.BCH(int(bch_n), int(bch_k))
    if int(code.t) != int(bch_t):
        raise ValueError("bch_t=%d != BCH(%d,%d).t=%d" % (bch_t, bch_n, bch_k, code.t))
    if code.n > int(data_dim):
        raise ValueError("bch_n=%d > data_dim=%d" % (code.n, data_dim))
    return code


class RivaGanWithControlBits(object):

    def __init__(self, model_paths: Sequence[str]):
        if not model_paths:
            raise ValueError("model_paths is empty")
        self.models = []
        for p in model_paths:
            p = os.path.abspath(p)
            if not os.path.isfile(p):
                raise FileNotFoundError(p)
            self.models.append(_torch_load_model(p))

    def _codeword_to_payload(self, codeword_bits: np.ndarray, data_dim: int):
        cw = codeword_bits.astype(np.float32).ravel()
        if len(cw) > data_dim:
            raise ValueError("codeword too long")
        out = np.zeros((data_dim,), dtype=np.float32)
        out[: len(cw)] = cw
        return out

    @staticmethod
    def _decoder_soft_bits(model, frame: torch.Tensor, n: int) -> np.ndarray:
        logits = model.decoder(frame)[0]
        return torch.sigmoid(logits).detach().cpu().numpy()[:n]

    def encode_with_control_bits(
        self,
        video_in: str,
        message_bits: Union[Sequence[int], np.ndarray],
        video_out: str,
        data_dim: int,
        bch_n: Optional[int] = None,
        bch_k: Optional[int] = None,
        bch_t: Optional[int] = None,
        *,
        raw_payload: bool = False,
    ):
        msg = np.asarray(message_bits, dtype=int).ravel()
        if msg.size == 0 or not np.isin(msg, [0, 1]).all():
            raise ValueError("message_bits must be non-empty 0/1 vector")

        model = _model_for_data_dim(self.models, data_dim)
        dd = int(data_dim)

        if raw_payload:
            if msg.size > dd:
                raise ValueError("message too long for data_dim")
            payload = self._codeword_to_payload(msg, dd)
        else:
            if None in (bch_n, bch_k, bch_t):
                raise ValueError("need bch_n, bch_k, bch_t")
            import galois

            code = _validated_bch_code(dd, bch_n, bch_k, bch_t)
            k = code.k
            if msg.size > k:
                raise ValueError("message too long for BCH k=%d" % k)
            padded = np.zeros(k, dtype=int)
            padded[: msg.size] = msg
            cw = np.array(code.encode(galois.GF2(padded)), dtype=int)
            payload = self._codeword_to_payload(cw, dd)

        data = torch.FloatTensor([payload.tolist()]).cuda()
        enc = model.encoder
        enc.eval()

        cap_in = cv2.VideoCapture(video_in)
        w = int(cap_in.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap_in.get(cv2.CAP_PROP_FRAME_HEIGHT))
        nframes = int(cap_in.get(cv2.CAP_PROP_FRAME_COUNT))
        writer = cv2.VideoWriter(
            video_out, cv2.VideoWriter_fourcc(*"mp4v"), 20.0, (w, h)
        )

        with torch.no_grad():
            for _ in tqdm(range(nframes)):
                ok, frame = cap_in.read()
                if not ok:
                    break
                x = torch.FloatTensor([frame]) / 127.5 - 1.0
                x = x.permute(3, 0, 1, 2).unsqueeze(0).cuda()
                wm = torch.clamp(enc(x, data), min=-1.0, max=1.0)
                out = (
                    (wm[0, :, 0, :, :].permute(1, 2, 0) + 1.0) * 127.5
                ).detach().cpu().numpy().astype("uint8")
                writer.write(out)

        writer.release()
        cap_in.release()

    def decode_with_control_bits(
        self,
        video_in: str,
        data_dim: int,
        bch_n: Optional[int] = None,
        bch_k: Optional[int] = None,
        bch_t: Optional[int] = None,
        message_bits_len: Optional[int] = None,
        *,
        raw_payload: bool = False,
    ) -> Iterable[np.ndarray]:
        model = _model_for_data_dim(self.models, data_dim)
        dd = int(data_dim)

        if raw_payload:
            trim = message_bits_len if message_bits_len is not None else dd
            if trim > dd:
                raise ValueError("message_bits_len > data_dim")
            n_read = trim
            code = None
        else:
            if None in (bch_n, bch_k, bch_t):
                raise ValueError("need bch_n, bch_k, bch_t")
            import galois

            code = _validated_bch_code(dd, bch_n, bch_k, bch_t)
            trim = message_bits_len if message_bits_len is not None else code.k
            if trim > code.k:
                raise ValueError("message_bits_len > k")
            n_read = code.n

        cap_in = cv2.VideoCapture(video_in)
        length = int(cap_in.get(cv2.CAP_PROP_FRAME_COUNT))

        with torch.no_grad():
            for _ in tqdm(range(length)):
                ok, frame = cap_in.read()
                if not ok:
                    break
                x = torch.FloatTensor([frame]) / 127.5 - 1.0
                x = x.permute(3, 0, 1, 2).unsqueeze(0).cuda()
                soft = self._decoder_soft_bits(model, x, n_read)
                hard = (soft >= 0.5).astype(np.int64)
                if raw_payload:
                    yield hard[:trim].copy()
                else:
                    info = np.array(code.decode(galois.GF2(hard.astype(int))), dtype=int)
                    yield info[:trim].copy()

        cap_in.release()
