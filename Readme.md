# Hướng Dẫn Finetuning XTTSv2 cho Ngôn Ngữ Mới

Hướng dẫn này cung cấp các bước chi tiết để finetuning XTTSv2 trên một ngôn ngữ mới, sử dụng tiếng Việt (`vi`) làm ví dụ.

## Mục Lục
1. [Cài Đặt](#1-cai-dat)
2. [Chuẩn Bị Dữ Liệu](#2-chuan-bi-du-lieu)
3. [Tải Mô Hình Pretrained](#3-tai-mo-hinh-pretrained)
4. [Mở Rộng Từ Vựng và Điều Chỉnh Cấu Hình](#4-mo-rong-tu-vung-va-dieu-chinh-cau-hinh)
5. [Finetuning DVAE (Tùy Chọn)](#5-finetuning-dvae-tuy-chon)
6. [Finetuning GPT](#6-finetuning-gpt)
7. [Ví Dụ Sử Dụng](#7-vi-du-su-dung)

## 1. Cài Đặt

Trước tiên, clone kho lưu trữ và cài đặt các thư viện cần thiết:

```
git clone https://github.com/PineappleCuteCute/XTTSv2-Finetuning-for-New-Languages-Update.git
cd XTTSv2-Finetuning-for-New-Languages
pip install -r requirements.txt
```

## 2. Chuẩn Bị Dữ Liệu

Hãy chắc chắn dữ liệu của bạn được tổ chức như sau:

```
project_root/
├── datasets/
│   ├── wavs/
│   │   ├── xxx.wav
│   │   ├── yyy.wav
│   │   ├── zzz.wav
│   │   └── ...
│   ├── metadata_train.csv
│   ├── metadata_eval.csv
│   
├── recipes/
├── scripts/
├── TTS/
└── README.md
```

Định dạng các file `metadata_train.csv` và `metadata_eval.csv` như sau:

```
audio_file|text|speaker_name
wavs/xxx.wav|How do you do?|@X
wavs/yyy.wav|Nice to meet you.|@Y
wavs/zzz.wav|Good to see you.|@Z
```

## 3. Tải Mô Hình Pretrained

Chạy lệnh sau để tải mô hình pretrained:

```bash
python download_checkpoint.py --output_path checkpoints/
```

## 4. Mở Rộng Từ Vựng và Điều Chỉnh Cấu Hình

Mở rộng từ vựng và điều chỉnh cấu hình với lệnh sau:

```bash
python extend_vocab_config.py --output_path=checkpoints/ --metadata_path datasets/metadata_train.csv --language vi --extended_vocab_size 2000
```

## 5. Finetuning DVAE (Tùy Chọn)

Để finetuning DVAE, chạy lệnh sau:

```bash
CUDA_VISIBLE_DEVICES=0 python train_dvae_xtts.py \
--output_path=checkpoints/ \
--train_csv_path=datasets/metadata_train.csv \
--eval_csv_path=datasets/metadata_eval.csv \
--language="vi" \
--num_epochs=5 \
--batch_size=512 \
--lr=5e-6
```

## 6. Finetuning GPT

Để finetuning GPT, chạy lệnh:

```bash
CUDA_VISIBLE_DEVICES=0 python train_gpt_xtts.py \
--output_path=checkpoints/ \
--train_csv_path=datasets/metadata_train.csv \
--eval_csv_path=datasets/metadata_eval.csv \
--language="vi" \
--num_epochs=5 \
--batch_size=8 \
--grad_acumm=2 \
--max_text_length=250 \
--max_audio_length=255995 \
--weight_decay=1e-2 \
--lr=5e-6 \
--save_step=2000
```

## 7. Ví Dụ Sử Dụng

Dưới đây là đoạn mã mẫu minh họa cách sử dụng mô hình đã được finetuning:

```python
import torch
import torchaudio
from tqdm import tqdm
from underthesea import sent_tokenize

from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts

# Cấu hình thiết bị
device = "cuda:0" nếu torch.cuda.is_available() khác "cpu"

# Đường dẫn mô hình
xtts_checkpoint = "checkpoints/GPT_XTTS_FT-August-30-2024_08+19AM-6a6b942/best_model_99875.pth"
xtts_config = "checkpoints/GPT_XTTS_FT-August-30-2024_08+19AM-6a6b942/config.json"
xtts_vocab = "checkpoints/XTTS_v2.0_original_model_files/vocab.json"

# Tải mô hình
config = XttsConfig()
config.load_json(xtts_config)
XTTS_MODEL = Xtts.init_from_config(config)
XTTS_MODEL.load_checkpoint(config, checkpoint_path=xtts_checkpoint, vocab_path=xtts_vocab, use_deepspeed=False)
XTTS_MODEL.to(device)

print("Tải mô hình thành công!")

# Suy luận
tts_text = "Good to see you."
speaker_audio_file = "ref.wav"
lang = "vi"

gpt_cond_latent, speaker_embedding = XTTS_MODEL.get_conditioning_latents(
    audio_path=speaker_audio_file,
    gpt_cond_len=XTTS_MODEL.config.gpt_cond_len,
    max_ref_length=XTTS_MODEL.config.max_ref_len,
    sound_norm_refs=XTTS_MODEL.config.sound_norm_refs,
)

tts_texts = sent_tokenize(tts_text)

wav_chunks = []
for text in tqdm(tts_texts):
    wav_chunk = XTTS_MODEL.inference(
        text=text,
        language=lang,
        gpt_cond_latent=gpt_cond_latent,
        speaker_embedding=speaker_embedding,
        temperature=0.1,
        length_penalty=1.0,
        repetition_penalty=10.0,
        top_k=10,
        top_p=0.3,
    )
    wav_chunks.append(torch.tensor(wav_chunk["wav"]))

out_wav = torch.cat(wav_chunks, dim=0).unsqueeze(0).cpu()

# Phát âm thanh (trong Jupyter Notebook)
from IPython.display import Audio
Audio(out_wav, rate=24000)
```

Lưu ý: Finetuning HiFiGAN decoder đã được thử nghiệm nhưng cho kết quả kém hơn. Việc finetuning DVAE và GPT là đủ để đạt được hiệu quả tối ưu.
