import json
import logging
import os
import random
import tempfile
import zlib
from glob import glob
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

_RAW_MODE_SEED_XOR = 0x51ED << 16


def setup_logging(log_file: Optional[str] = None) -> None:
    handlers = [logging.StreamHandler()]
    if log_file:
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=handlers,
    )


def list_videos(root_dir: str) -> List[str]:
    paths = []
    for ext in ("avi", "mp4"):
        paths.extend(
            sorted(glob(os.path.join(root_dir, "**/*.%s" % ext), recursive=True))
        )
    return paths


def dedupe_paths(model_paths: Sequence[str]) -> List[str]:
    out = []
    seen = set()
    for p in model_paths:
        ap = os.path.abspath(p)
        if ap in seen:
            continue
        seen.add(ap)
        out.append(ap)
    return out


def deterministic_start_index(
    video_path: str, num_frames: int, nb_frames: int
) -> int:
    if nb_frames <= 0:
        return 0
    h = zlib.adler32(video_path.encode("utf-8", errors="replace")) & 0xFFFFFFFF
    rng = random.Random(h)
    if nb_frames <= num_frames:
        return 0
    return rng.randint(0, nb_frames - num_frames)


def write_clip(
    src_path: str,
    dst_path: str,
    start_idx: int,
    num_frames: int,
    fps: float = 20.0,
) -> Tuple[int, int]:
    cap = cv2.VideoCapture(src_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_idx)
    writer = cv2.VideoWriter(
        dst_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    for _ in range(num_frames):
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        writer.write(frame)
    writer.release()
    cap.release()
    return width, height


def prepare_video_clip(
    video_path: str, tmp_dir: str, vi: int, num_frames: int, fps: float
) -> Tuple[str, int, int]:
    cap = cv2.VideoCapture(video_path)
    nb_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    start_idx = deterministic_start_index(video_path, num_frames, nb_frames)
    clip_in = os.path.join(tmp_dir, "clip_%d_in.mp4" % vi)
    write_clip(video_path, clip_in, start_idx, num_frames, fps=fps)
    return clip_in, start_idx, nb_frames


def find_bch_for_correctable_errors(
    data_dim: int, want_t: int
) -> Optional[Tuple[int, int, int]]:
    import galois

    want_t = int(want_t)
    if want_t < 1:
        return None
    best = None
    m = 2
    while True:
        n = (1 << m) - 1
        if n > data_dim:
            break
        for k in range(n - 1, 0, -1):
            try:
                c = galois.BCH(n, k)
            except (ValueError, LookupError):
                continue
            if int(c.t) != want_t:
                continue
            cand = (n, k, int(c.t))
            if best is None or (n, k) > (best[0], best[1]):
                best = cand
        m += 1
    if best is None:
        return None
    return int(best[0]), int(best[1]), int(best[2])


def master_message_bits(
    k: int,
    video_path: str,
    data_dim: int,
    bch_n: int,
    bch_k: int,
    bch_t: int,
    *,
    raw_mode: bool = False,
) -> np.ndarray:
    seed = (
        (zlib.adler32(video_path.encode("utf-8", errors="replace")) & 0xFFFFFFFF)
        ^ (int(data_dim) << 20)
        ^ (int(bch_n) << 12)
        ^ (int(bch_k) << 6)
        ^ int(bch_t)
        ^ (_RAW_MODE_SEED_XOR if raw_mode else 0)
    )
    rng = np.random.RandomState(seed)
    return rng.randint(0, 2, size=k, dtype=np.int64)


def bit_accuracy(decoded: np.ndarray, truth: np.ndarray) -> float:
    n = min(len(decoded), len(truth))
    if n == 0:
        return 0.0
    return float(np.mean(decoded[:n] == truth[:n]))


def full_message_match(decoded: np.ndarray, truth: np.ndarray) -> bool:
    truth = np.asarray(truth, dtype=np.int64).ravel()
    dec = np.asarray(decoded, dtype=np.int64).ravel()
    if len(dec) < len(truth):
        return False
    return bool(np.all(dec[: len(truth)] == truth))


def majority_vote_message(decoded_rows: Sequence[np.ndarray]) -> np.ndarray:
    if not decoded_rows:
        return np.array([], dtype=np.int64)
    mat = np.stack(
        [np.asarray(r, dtype=np.int64).ravel() for r in decoded_rows], axis=0
    )
    votes = mat.sum(axis=0).astype(np.int64)
    n = mat.shape[0]
    return (2 * votes > n).astype(np.int64)


def aggregate_decode_stats(
    dec_rows: Sequence[np.ndarray], truth: np.ndarray, msg_len: int
) -> Tuple[float, List[float], float, int, bool]:
    accs = []
    for row in dec_rows:
        r = np.asarray(row, dtype=np.int64).ravel()[:msg_len]
        accs.append(bit_accuracy(r, truth))
    n_used = len(accs)
    mean_acc = float(np.mean(accs)) if accs else 0.0
    full_count = sum(1 for row in dec_rows if full_message_match(row, truth))
    full_frac = float(full_count / n_used) if n_used else 0.0
    maj = majority_vote_message(dec_rows)
    maj_ok = (
        bool(len(maj) == msg_len and np.array_equal(maj, truth)) if n_used else False
    )
    return mean_acc, accs, full_frac, full_count, maj_ok


def summary_mean_bit_accuracy_by_dim_t_msglen(
    results: Sequence[Dict[str, Any]],
) -> Dict[str, float]:
    summary: Dict[str, Any] = {}
    for r in results:
        if r.get("status") != "ok" or r.get("mean_bit_accuracy") is None:
            continue
        key = "dim_%d_t%d_L%d" % (
            r["data_dim"],
            r["correctable_errors"],
            r["message_length"],
        )
        summary.setdefault(key, []).append(r["mean_bit_accuracy"])
    return {k: float(np.mean(v)) for k, v in summary.items()}


def write_json_atomic(
    output_json: str,
    out_doc: Dict[str, Any],
    tmp_prefix: str = "rivagan_eval_json_",
) -> None:
    out_dir = os.path.dirname(os.path.abspath(output_json))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    abs_path = os.path.abspath(output_json)
    fd, tmp_path = tempfile.mkstemp(
        prefix=tmp_prefix, suffix=".tmp", dir=out_dir or None
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(out_doc, f, indent=2, ensure_ascii=False)
        os.replace(tmp_path, abs_path)
    except BaseException:
        try:
            if os.path.isfile(tmp_path):
                os.remove(tmp_path)
        except OSError:
            pass
        raise


def _norm_paths(paths: Sequence[str]) -> List[str]:
    return sorted(
        os.path.normcase(os.path.normpath(os.path.abspath(str(p)))) for p in paths
    )


def configs_equal_bit_eval(a: Dict[str, Any], b: Dict[str, Any]) -> bool:
    if int(a.get("num_frames", -1)) != int(b.get("num_frames", -2)):
        return False
    if float(a.get("fps", -1.0)) != float(b.get("fps", -2.0)):
        return False
    for key in ("correctable_error_counts", "message_lengths", "data_dims"):
        if list(map(int, a.get(key, []))) != list(map(int, b.get(key, []))):
            return False
    if str(a.get("wm_method", "")) != str(b.get("wm_method", "")):
        return False
    da = os.path.normcase(
        os.path.normpath(os.path.abspath(str(a.get("test_dataset", ""))))
    )
    db = os.path.normcase(
        os.path.normpath(os.path.abspath(str(b.get("test_dataset", ""))))
    )
    if da != db:
        return False
    return _norm_paths(a.get("model_paths", [])) == _norm_paths(
        b.get("model_paths", [])
    )


def load_resume_from_output_json(
    output_json: str,
    config: Dict[str, Any],
    videos: Sequence[str],
    config_equal: Callable[[Dict[str, Any], Dict[str, Any]], bool],
) -> Tuple[List[Dict[str, Any]], int]:
    abs_out = os.path.abspath(output_json)
    if not os.path.isfile(abs_out):
        return [], 0
    try:
        with open(abs_out, "r", encoding="utf-8") as f:
            doc = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        logger.warning(
            "Не удалось прочитать %s для продолжения: %s — полная оценка",
            abs_out,
            e,
        )
        return [], 0
    old_cfg = doc.get("config")
    if not isinstance(old_cfg, dict) or not config_equal(old_cfg, config):
        logger.warning(
            "Файл %s есть, но config не совпадает — оценка с нуля",
            abs_out,
        )
        return [], 0
    prog = doc.get("progress") or {}
    try:
        done = int(prog.get("videos_completed", 0))
    except (TypeError, ValueError):
        done = 0
    raw_results = doc.get("results")
    if not isinstance(raw_results, list):
        raw_results = []
    n = len(videos)
    if done < 0:
        done = 0
    if done > n:
        logger.warning(
            "В %s progress.videos_completed=%d > числа видео %d — игнорируем продолжение",
            abs_out,
            done,
            n,
        )
        return [], 0
    if done > 0:
        logger.info(
            "Продолжение: %s — обработано видео %d/%d, старт с индекса %d",
            abs_out,
            done,
            n,
            done,
        )
    return raw_results, done


def configs_equal_model_metrics(a: Dict[str, Any], b: Dict[str, Any]) -> bool:
    if int(a.get("seq_len", -1)) != int(b.get("seq_len", -2)):
        return False
    if tuple(a.get("val_max_crop", ())) != tuple(b.get("val_max_crop", ())):
        return False
    da = os.path.normcase(
        os.path.normpath(os.path.abspath(str(a.get("test_dataset", ""))))
    )
    db = os.path.normcase(
        os.path.normpath(os.path.abspath(str(b.get("test_dataset", ""))))
    )
    if da != db:
        return False
    return _norm_paths(a.get("model_paths", [])) == _norm_paths(
        b.get("model_paths", [])
    )


def resolve_coding_plan(
    data_dim: int, want_t: int, video_path: str
) -> Optional[Dict[str, Any]]:
    want_t = int(want_t)
    if want_t == 0:
        info_k = int(data_dim)
        return {
            "coding": "raw",
            "use_raw": True,
            "info_k": info_k,
            "parity_bits": 0,
            "bch_n": None,
            "bch_k": None,
            "bch_t": None,
            "master": master_message_bits(
                info_k, video_path, data_dim, 0, 0, 0, raw_mode=True
            ),
        }
    bch_tuple = find_bch_for_correctable_errors(data_dim, want_t)
    if bch_tuple is None:
        return None
    bch_n, bch_k, bch_t = bch_tuple
    info_k = int(bch_k)
    return {
        "coding": "bch",
        "use_raw": False,
        "info_k": info_k,
        "parity_bits": bch_n - bch_k,
        "bch_n": bch_n,
        "bch_k": bch_k,
        "bch_t": bch_t,
        "master": master_message_bits(
            info_k, video_path, data_dim, bch_n, bch_k, bch_t, raw_mode=False
        ),
    }


def skip_length_reason(
    msg_len: int,
    data_dim: int,
    use_raw: bool,
    bch_k: Optional[int],
    bch_n: Optional[int],
) -> Optional[str]:
    max_len = int(data_dim) if use_raw else int(bch_k)
    if msg_len <= max_len:
        return None
    if use_raw:
        return "Длина сообщения %d > data_dim=%d (сырой payload)" % (msg_len, data_dim)
    return "Длина сообщения %d > k=%d для BCH(%d,%d) (data_dim=%d)" % (
        msg_len,
        bch_k,
        bch_n,
        bch_k,
        data_dim,
    )


def make_skip_record(
    video_path: str,
    data_dim: int,
    want_t: int,
    coding: str,
    msg_len: int,
    reason: str,
    start_idx: int,
    num_frames: int,
    bch_n=None,
    bch_k=None,
    bch_t=None,
    parity_bits=None,
) -> Dict[str, Any]:
    return {
        "video": video_path,
        "data_dim": data_dim,
        "correctable_errors": want_t,
        "coding": coding,
        "message_length": msg_len,
        "bch_n": bch_n,
        "bch_k": bch_k,
        "bch_t": bch_t,
        "parity_bits": parity_bits,
        "frame_start": start_idx,
        "num_frames": num_frames,
        "status": "skipped",
        "skip_reason": reason,
        "mean_bit_accuracy": None,
        "per_frame_bit_accuracy": None,
        "full_recovery_frame_fraction": None,
        "full_recovery_frame_count": None,
        "majority_vote_success": None,
    }
