import argparse
import logging
import math
import os
import tempfile
from typing import Any, Dict, List, Optional, Sequence

import cv2
import numpy as np
from imwatermark import WatermarkDecoder, WatermarkEncoder

from rivagan import eval_common as ec

logger = logging.getLogger(__name__)


def _num_payload_bytes_for_bits(n_bits: int) -> int:
    return int(math.ceil(float(n_bits) / 8.0))


def _pack_message_bits_to_bytes(bits: np.ndarray) -> bytes:
    bits = np.asarray(bits, dtype=np.uint8).ravel()
    if bits.size == 0:
        return b""
    return np.packbits(bits, bitorder="big").tobytes()


def _unpack_message_bytes_to_bits(buf: bytes, expected_bits_len: int) -> np.ndarray:
    arr = np.frombuffer(bytes(buf), dtype=np.uint8)
    bits = np.unpackbits(arr, bitorder="big").astype(np.int64)
    return bits[:expected_bits_len]


def _pad_payload_to_data_dim(bits: np.ndarray, data_dim: int) -> np.ndarray:
    bits = np.asarray(bits, dtype=np.int64).ravel()
    if len(bits) > data_dim:
        raise ValueError("payload length %d > data_dim %d" % (len(bits), data_dim))
    if len(bits) == data_dim:
        return bits
    pad = np.zeros(data_dim - len(bits), dtype=np.int64)
    return np.concatenate([bits, pad], axis=0)


def _encode_payload_bits(
    message_bits: np.ndarray,
    data_dim: int,
    *,
    raw_payload: bool,
    bch_n: Optional[int] = None,
    bch_k: Optional[int] = None,
) -> np.ndarray:
    msg = np.asarray(message_bits, dtype=np.int64).ravel()
    if raw_payload:
        return _pad_payload_to_data_dim(msg, data_dim)
    import galois

    if bch_n is None or bch_k is None:
        raise ValueError("bch_n and bch_k required")
    if len(msg) > int(bch_k):
        raise ValueError("message length %d > bch_k %d" % (len(msg), int(bch_k)))
    if len(msg) < int(bch_k):
        msg = np.concatenate(
            [msg, np.zeros(int(bch_k) - len(msg), dtype=np.int64)], axis=0
        )
    bch = galois.BCH(int(bch_n), int(bch_k))
    code = np.asarray(bch.encode(msg.astype(np.int64)), dtype=np.int64).ravel()
    return _pad_payload_to_data_dim(code, data_dim)


def _decode_payload_bits(
    decoded_payload_bits: np.ndarray,
    message_bits_len: int,
    *,
    raw_payload: bool,
    bch_n: Optional[int] = None,
    bch_k: Optional[int] = None,
) -> np.ndarray:
    payload = np.asarray(decoded_payload_bits, dtype=np.int64).ravel()
    if raw_payload:
        return payload[:message_bits_len]
    import galois

    if bch_n is None or bch_k is None:
        raise ValueError("need bch_n and bch_k")
    codeword = payload[: int(bch_n)].astype(np.int64)
    decoded_msg = np.asarray(
        galois.BCH(int(bch_n), int(bch_k)).decode(codeword), dtype=np.int64
    ).ravel()
    return decoded_msg[:message_bits_len]


def _encode_video_frames(
    clip_in: str,
    clip_out: str,
    payload_bytes: bytes,
    num_frames: int,
    fps: float,
    method: str,
) -> None:
    cap = cv2.VideoCapture(clip_in)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(
        clip_out, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    encoder = WatermarkEncoder()
    encoder.set_watermark("bytes", payload_bytes)
    for _ in range(num_frames):
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        writer.write(encoder.encode(frame, method))
    writer.release()
    cap.release()


def _decode_video_frames(
    clip_wm: str,
    payload_num_bytes: int,
    expected_bits_len: int,
    num_frames: int,
    method: str,
) -> List[np.ndarray]:
    cap = cv2.VideoCapture(clip_wm)
    decoder = WatermarkDecoder("bytes", payload_num_bytes)
    rows: List[np.ndarray] = []
    for _ in range(num_frames):
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        decoded = decoder.decode(frame, method)
        rows.append(_unpack_message_bytes_to_bits(decoded, expected_bits_len))
    cap.release()
    return rows


def run_evaluation(
    model_paths: Sequence[str],
    test_dataset: str,
    output_json: str,
    correctable_error_counts: Sequence[int],
    message_lengths: Sequence[int],
    data_dims: Sequence[int],
    num_frames: int = 8,
    log_file: Optional[str] = None,
    fps: float = 20.0,
    wm_method: str = "dwtDct",
) -> Dict[str, Any]:
    ec.setup_logging(log_file)
    logger.info(
        "Старт оценки: датасет=%s, num_frames=%d, data_dims=%s, "
        "correctable_error_counts=%s, message_lengths=%s, wm_method=%s",
        test_dataset,
        num_frames,
        tuple(map(int, data_dims)),
        tuple(map(int, correctable_error_counts)),
        tuple(map(int, message_lengths)),
        wm_method,
    )
    if any(int(x) > 0 for x in correctable_error_counts):
        import galois  # noqa: F401

    correctable_error_counts = [int(x) for x in correctable_error_counts]
    message_lengths = [int(x) for x in message_lengths]
    data_dims = [int(x) for x in data_dims]
    if any(t < 0 for t in correctable_error_counts):
        raise ValueError("correctable_error_counts must be non-negative")
    if any(L < 1 for L in message_lengths):
        raise ValueError("message_lengths must be positive")
    if any(d < 1 for d in data_dims):
        raise ValueError("data_dims must be positive")
    if wm_method not in ("dwtDct", "dwtDctSvd"):
        raise ValueError("wm_method must be one of: dwtDct, dwtDctSvd")

    paths_unique = ec.dedupe_paths(model_paths)
    videos = ec.list_videos(test_dataset)
    if not videos:
        raise ValueError(
            "В каталоге не найдено видео (*.avi, *.mp4): %s" % test_dataset
        )
    logger.info("Найдено видео для оценки: %d", len(videos))

    config = {
        "test_dataset": os.path.abspath(test_dataset),
        "model_paths": [os.path.abspath(p) for p in paths_unique],
        "num_frames": num_frames,
        "correctable_error_counts": correctable_error_counts,
        "message_lengths": message_lengths,
        "data_dims": data_dims,
        "fps": fps,
        "wm_method": wm_method,
    }
    results, start_vi = ec.load_resume_from_output_json(
        output_json, config, videos, ec.configs_equal_bit_eval
    )
    if start_vi >= len(videos):
        logger.info(
            "Все %d видео уже есть в %s — повторная оценка не требуется",
            len(videos),
            os.path.abspath(output_json),
        )
        return {
            "config": config,
            "progress": {"videos_completed": len(videos), "videos_total": len(videos)},
            "summary_mean_bit_accuracy_by_dim_t_msglen": ec.summary_mean_bit_accuracy_by_dim_t_msglen(
                results
            ),
            "results": results,
        }

    def _flush_results(videos_done: int) -> None:
        ec.write_json_atomic(
            output_json,
            {
                "config": config,
                "progress": {
                    "videos_completed": videos_done,
                    "videos_total": len(videos),
                },
                "summary_mean_bit_accuracy_by_dim_t_msglen": ec.summary_mean_bit_accuracy_by_dim_t_msglen(
                    results
                ),
                "results": results,
            },
        )
        logger.info(
            "Промежуточные результаты записаны (%d/%d видео): %s",
            videos_done,
            len(videos),
            os.path.abspath(output_json),
        )

    with tempfile.TemporaryDirectory(prefix="rivagan_eval_") as tmp:
        for vi in range(start_vi, len(videos)):
            video_path = videos[vi]
            clip_in, start_idx, nb_frames = ec.prepare_video_clip(
                video_path, tmp, vi, num_frames, fps
            )
            logger.info(
                "Видео [%d/%d] %s: кадры %d..%d (всего кадров в источнике %d)",
                vi + 1,
                len(videos),
                video_path,
                start_idx,
                start_idx + num_frames - 1,
                nb_frames,
            )
            for data_dim in data_dims:
                for want_t in correctable_error_counts:
                    plan = ec.resolve_coding_plan(data_dim, want_t, video_path)
                    if plan is None:
                        reason = "Нет BCH с t=%d при data_dim=%d" % (want_t, data_dim)
                        logger.warning(
                            "%s — пропуск (видео %s)",
                            reason,
                            os.path.basename(video_path),
                        )
                        for msg_len in message_lengths:
                            results.append(
                                ec.make_skip_record(
                                    video_path,
                                    data_dim,
                                    want_t,
                                    "bch",
                                    msg_len,
                                    reason,
                                    start_idx,
                                    num_frames,
                                )
                            )
                        continue

                    coding = plan["coding"]
                    use_raw = plan["use_raw"]
                    bch_n, bch_k, bch_t = plan["bch_n"], plan["bch_k"], plan["bch_t"]
                    parity_bits = plan["parity_bits"]
                    master = plan["master"]

                    for msg_len in message_lengths:
                        reason = ec.skip_length_reason(
                            msg_len, data_dim, use_raw, bch_k, bch_n
                        )
                        if reason:
                            logger.warning("%s — пропуск", reason)
                            results.append(
                                ec.make_skip_record(
                                    video_path,
                                    data_dim,
                                    want_t,
                                    coding,
                                    msg_len,
                                    reason,
                                    start_idx,
                                    num_frames,
                                    bch_n=bch_n,
                                    bch_k=bch_k,
                                    bch_t=bch_t,
                                    parity_bits=parity_bits,
                                )
                            )
                            continue

                        truth = master[:msg_len]
                        try:
                            encoded_payload_bits = _encode_payload_bits(
                                truth,
                                data_dim=data_dim,
                                raw_payload=use_raw,
                                bch_n=bch_n,
                                bch_k=bch_k,
                            )
                        except Exception as e:
                            logger.error(
                                "Ошибка payload-encode dim=%d t=%d L=%d видео=%s: %s",
                                data_dim,
                                want_t,
                                msg_len,
                                video_path,
                                e,
                                exc_info=True,
                            )
                            results.append(
                                {
                                    "video": video_path,
                                    "data_dim": data_dim,
                                    "correctable_errors": want_t,
                                    "coding": coding,
                                    "parity_bits": parity_bits,
                                    "message_length": msg_len,
                                    "bch_n": bch_n,
                                    "bch_k": bch_k,
                                    "bch_t": bch_t,
                                    "frame_start": start_idx,
                                    "num_frames": num_frames,
                                    "status": "encode_error",
                                    "error": str(e),
                                    "mean_bit_accuracy": None,
                                    "per_frame_bit_accuracy": None,
                                    "full_recovery_frame_fraction": None,
                                    "full_recovery_frame_count": None,
                                    "majority_vote_success": None,
                                }
                            )
                            continue

                        payload_bytes = _pack_message_bits_to_bytes(encoded_payload_bits)
                        payload_num_bytes = _num_payload_bytes_for_bits(data_dim)
                        clip_wm = os.path.join(
                            tmp, "v%d_d%d_t%d_L%d.mp4" % (vi, data_dim, want_t, msg_len)
                        )
                        try:
                            _encode_video_frames(
                                clip_in,
                                clip_wm,
                                payload_bytes,
                                num_frames,
                                fps,
                                wm_method,
                            )
                        except Exception as e:
                            logger.error(
                                "Ошибка encode dim=%d t=%d L=%d видео=%s: %s",
                                data_dim,
                                want_t,
                                msg_len,
                                video_path,
                                e,
                                exc_info=True,
                            )
                            results.append(
                                {
                                    "video": video_path,
                                    "data_dim": data_dim,
                                    "correctable_errors": want_t,
                                    "coding": coding,
                                    "parity_bits": parity_bits,
                                    "message_length": msg_len,
                                    "bch_n": bch_n,
                                    "bch_k": bch_k,
                                    "bch_t": bch_t,
                                    "frame_start": start_idx,
                                    "num_frames": num_frames,
                                    "status": "encode_error",
                                    "error": str(e),
                                    "mean_bit_accuracy": None,
                                    "per_frame_bit_accuracy": None,
                                    "full_recovery_frame_fraction": None,
                                    "full_recovery_frame_count": None,
                                    "majority_vote_success": None,
                                }
                            )
                            continue

                        dec_payload_rows = _decode_video_frames(
                            clip_wm,
                            payload_num_bytes,
                            data_dim,
                            num_frames,
                            wm_method,
                        )
                        dec_rows = [
                            _decode_payload_bits(
                                row,
                                message_bits_len=msg_len,
                                raw_payload=use_raw,
                                bch_n=bch_n,
                                bch_k=bch_k,
                            )
                            for row in dec_payload_rows
                        ]
                        mean_acc, accs, full_frac, full_count, maj_ok = (
                            ec.aggregate_decode_stats(dec_rows, truth, msg_len)
                        )
                        n_used = len(accs)
                        if use_raw:
                            logger.info(
                                "  dim=%d t=0 raw parity=0 L=%d payload_bits=%d "
                                "mean_bit_acc=%.4f full_frames=%.2f maj_ok=%s (%d кадров)",
                                data_dim,
                                msg_len,
                                data_dim,
                                mean_acc,
                                full_frac,
                                maj_ok,
                                n_used,
                            )
                        else:
                            logger.info(
                                "  dim=%d t=%d BCH(%d,%d) parity=%d L=%d payload_bits=%d "
                                "mean_bit_acc=%.4f full_frames=%.2f maj_ok=%s (%d кадров)",
                                data_dim,
                                want_t,
                                bch_n,
                                bch_k,
                                bch_n - bch_k,
                                msg_len,
                                data_dim,
                                mean_acc,
                                full_frac,
                                maj_ok,
                                n_used,
                            )
                        results.append(
                            {
                                "video": video_path,
                                "data_dim": data_dim,
                                "correctable_errors": want_t,
                                "coding": coding,
                                "parity_bits": parity_bits,
                                "message_length": msg_len,
                                "bch_n": bch_n,
                                "bch_k": bch_k,
                                "bch_t": bch_t,
                                "frame_start": start_idx,
                                "num_frames": num_frames,
                                "status": "ok",
                                "mean_bit_accuracy": mean_acc,
                                "per_frame_bit_accuracy": accs,
                                "full_recovery_frame_fraction": full_frac,
                                "full_recovery_frame_count": full_count,
                                "majority_vote_success": maj_ok,
                            }
                        )
            _flush_results(vi + 1)

    out_doc = {
        "config": config,
        "progress": {
            "videos_completed": len(videos),
            "videos_total": len(videos),
        },
        "summary_mean_bit_accuracy_by_dim_t_msglen": ec.summary_mean_bit_accuracy_by_dim_t_msglen(
            results
        ),
        "results": results,
    }
    ec.write_json_atomic(output_json, out_doc)
    logger.info("Оценка завершена, итог в %s", os.path.abspath(output_json))
    return out_doc


def main():
    parser = argparse.ArgumentParser(
        description="Оценка invisible-watermark + BCH на тестовом наборе видео"
    )
    parser.add_argument("--models", type=str, nargs="+", default=[])
    parser.add_argument(
        "--dataset", type=str, default=os.path.join(".", "data", "val")
    )
    parser.add_argument(
        "--output-json", type=str, default="./evaluation_dct_results.json"
    )
    parser.add_argument("--num-frames", type=int, default=8)
    parser.add_argument(
        "--correctable-errors", type=int, nargs="+", default=[0, 1, 2, 3]
    )
    parser.add_argument(
        "--message-lengths", type=int, nargs="+", default=[4, 8, 16, 24]
    )
    parser.add_argument(
        "--data-dims", type=int, nargs="+", default=[16, 32, 64]
    )
    parser.add_argument("--log-file", type=str, default=None)
    parser.add_argument("--fps", type=float, default=20.0)
    parser.add_argument(
        "--wm-method",
        type=str,
        default="dwtDct",
        choices=["dwtDct", "dwtDctSvd"],
    )
    args = parser.parse_args()
    run_evaluation(
        model_paths=args.models,
        test_dataset=args.dataset,
        output_json=args.output_json,
        correctable_error_counts=args.correctable_errors,
        message_lengths=args.message_lengths,
        data_dims=args.data_dims,
        num_frames=args.num_frames,
        log_file=args.log_file,
        fps=args.fps,
        wm_method=args.wm_method,
    )


if __name__ == "__main__":
    main()
