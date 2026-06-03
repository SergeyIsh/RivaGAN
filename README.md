# RivaGanWithControlBits — водяной знак для защиты авторских прав в видео с BCH-кодированием и скриптами оценки

Это форк [RivaGAN](https://github.com/DAI-Lab/RivaGAN) (MIT): архитектуры нейросетевого водяного знака для видео с attention-механизмом. В этом репозитории добавлены **контрольные биты BCH**, скрипты **оценки декодирования**, **baseline invisible-watermark (DWT/DCT)** и **метрики качества** модели. Также в папке models есть чекпоинты нескольких обученных моделей.

Оригинальная статья: Zhang et al., *Robust Invisible Video Watermarking with Attention*, [arXiv:1909.01285](https://arxiv.org/abs/1909.01285).

Этот репозиторий выполнен в рамках подготовки диссертации магистра на программе ИПИИ ВШЭ.

## Требования

- **Python 3.10**
- **NVIDIA GPU с CUDA**

## Установка

```bash
python3.10 -m venv .venv
source .venv/bin/activate



pip install -r requirements.txt
pip install -e .
```

Либо с опциональными группами из `setup.py`:

```bash
pip install -e ".[eval]"
```

## Структура проекта

```
rivagan/
├── rivagan.py                      # модель RivaGAN (encode/decode/fit)
├── watermarking_with_control_bits.py  # RivaGAN + BCH / raw payload
├── eval_common.py                  # общие хелперы для evaluation-скриптов
├── evaluation.py                   # оценка декодирования RivaGAN + BCH
├── evaluation_dct.py               # baseline: invisible-watermark + BCH
├── model_metrics.py                # SSIM, PSNR, robustness-метрики
├── experiments.py                  # обучение нескольких моделей по data_dim
└── ...                             # attention, dense, noise, dataloader
```

## Обучение

```bash
python -m rivagan.experiments \
  --dataset /path/to/train/videos \
  --output ./checkpoints \
  --data_dims 32 64 128 \
  --epochs 10 \
  --batch_size 10 \
  --seq_len 8
```

Чекпоинты сохраняются как полный объект `RivaGAN` (`torch.save`), по одному файлу на каждый `data_dim`.

## Водяной знак с BCH

Класс `RivaGanWithControlBits` загружает несколько чекпoинтов и на каждый вызов выбирает модель по `data_dim`.

**Сырой payload** (без BCH, до `data_dim` бит):

```python
from rivagan.watermarking_with_control_bits import RivaGanWithControlBits

wm = RivaGanWithControlBits(["checkpoints/rivagan_data_dim_32_epochs_10.pt"])
message = [1, 0, 1, 1, 0, 0, 1, 0]

wm.encode_with_control_bits(
    "input.mp4", message, "watermarked.mp4",
    data_dim=32, raw_payload=True,
)

for bits in wm.decode_with_control_bits(
    "watermarked.mp4", data_dim=32,
    message_bits_len=len(message), raw_payload=True,
):
    print(bits)  # один вектор на кадр
    break
```

**BCH-кодирование** (нужен пакет `galois`):

```python
wm.encode_with_control_bits(
    "input.mp4", message, "watermarked.mp4",
    data_dim=32, bch_n=31, bch_k=16, bch_t=3,
    raw_payload=False,
)
```

Параметры `bch_n`, `bch_k`, `bch_t` должны соответствовать `galois.BCH(bch_n, bch_k)`; `bch_n <= data_dim`.

## Оценка декодирования (RivaGAN)

Пакетная оценка на каталоге видео; результаты в JSON с возобновлением прогона.

```bash
python -m rivagan.evaluation \
  --models checkpoints/model_32.pt checkpoints/model_64.pt \
  --dataset ./data/val \
  --output-json ./evaluation_rivagan_results.json \
  --num-frames 8 \
  --correctable-errors 0 1 2 3 \
  --message-lengths 4 8 16 24
```

| Параметр | Описание |
|----------|----------|
| `--correctable-errors` | `t=0` — raw payload; `t≥1` — автоподбор BCH с исправлением `t` ошибок |
| `--message-lengths` | длины префиксов одного детерминированного сообщения |
| `--num-frames` | длина клипа (подряд идущие кадры) на каждое видео |

В JSON для каждой комбинации: `mean_bit_accuracy`, `full_recovery_frame_fraction`, `majority_vote_success`, сводка `summary_mean_bit_accuracy_by_dim_t_msglen`.

## Baseline: invisible-watermark + BCH

Сравнение с [invisible-watermark](https://github.com/ShieldMnt/invisible-watermark) (DWT/DCT), тот же протокол BCH/raw.

```bash
python -m rivagan.evaluation_dct \
  --dataset ./data/val \
  --output-json ./evaluation_dct_results.json \
  --data-dims 16 32 64 \
  --correctable-errors 0 1 2 3 \
  --message-lengths 4 8 16 24 \
  --wm-method dwtDct
```

Аргумент `--models` оставлен для совместимости CLI и не используется.

## Метрики качества модели

Те же метрики, что в validation-цикле `RivaGAN.fit`:

```bash
python -m rivagan.model_metrics \
  --models checkpoints/model_32.pt \
  --dataset ./data/val \
  --output-json ./model_metrics_results.json \
  --seq-len 8
```

Считаются `test.ssim`, `test.psnr`, `test.crop_acc`, `test.scale_acc`, `test.mjpeg_acc`.

## Типичный pipeline

1. Обучить модели: `python -m rivagan.experiments ...`
2. Проверить качество: `python -m rivagan.model_metrics ...`
3. Оценить декодирование: `python -m rivagan.evaluation ...`
4. Сравнить с baseline: `python -m rivagan.evaluation_dct ...`

## Визуализация работы алгоритма

Смотрите по ссылке [https://disk.yandex.ru/i/maPL3oNh9NNjAg](https://disk.yandex.ru/i/maPL3oNh9NNjAg)
