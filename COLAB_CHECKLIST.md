# Lab 21 — Colab Run Checklist (T4)

**MSSV**: 2A202600887 | **Họ tên**: Nguyen Manh Hieu
**Runtime**: Google Colab Free Tesla T4 16GB

---

## Trước khi bắt đầu

### 1. Upload notebook lên Colab
1. Mở [colab.research.google.com](https://colab.research.google.com)
2. **Upload** file `Lab21_LoRA_Finetuning_T4.ipynb` từ repo local
3. Hoặc clone repo GitHub: `!git clone https://github.com/<your-username>/<repo-name>.git`

### 2. Bật GPU T4
```
Runtime → Change runtime type → T4 GPU → Save
```
Kiểm tra: `!nvidia-smi` (cần thấy **Tesla T4**)

### 3. Mount Google Drive (tùy chọn)
- Nếu muốn lưu checkpoint lâu dài: đặt `MOUNT_DRIVE = True` trong notebook (cell 5)
- Nếu dùng Colab local: `MOUNT_DRIVE = False` (mặc định) — files sẽ mất khi Colab reset

---

## Phase 1 — Chạy Notebook theo Cell

### ✅ Cell 0: Markdown — Overview
> Đọc profile T4, xác nhận settings đúng.

---

### ✅ Cell 1: Markdown — Section 0: Setup

### ✅ Cell 2: `!nvidia-smi`
**Check**: Tesla T4, 14-16 GB VRAM, CUDA 12.x
**Nếu thấy**: A100 / L4 → dùng notebook BigGPU thay vì T4

---

### ✅ Cell 3: GPU Verify Python
**Check**:
```
✓ GPU: Tesla T4
✓ VRAM: ~14.x GB
```
**Lỗi thường**: "GPU not available" → quay lại Runtime > Change runtime type

---

### ✅ Cell 4: Install Dependencies ⚠️ CRITICAL
```python
%%capture
!pip install -q --upgrade pip
!pip install -q "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
!pip install -q --no-deps "trl>=0.12,<0.16" peft accelerate bitsandbytes
!pip install -q datasets matplotlib seaborn pandas
```
**Thời gian**: ~3-5 phút
**Check**: Không có ERROR (Warning OK)

---

### ✅ Cell 5: Mount Drive + Output Dir
```python
MOUNT_DRIVE = False  # hoặc True nếu cần lưu lâu dài
```
**Output**: `✓ Output dir: /content/lab21_lora_t4`

---

### ✅ Cell 6: Markdown — Dataset Preparation

### ✅ Cell 7: Load Dataset (Option A)
```python
raw = load_dataset("5CD-AI/Vietnamese-alpaca-gpt4-gg-translated", split="train")
raw = raw.shuffle(seed=42).select(range(200))
```
**Check**: `✓ Loaded 200 samples`
**Lỗi**: HF_TOKEN timeout → ignore warning, vẫn chạy được (unauthenticated)

---

### ✅ Cell 8: Custom Dataset (Option B)
> **Skip** — dùng Option A theo yêu cầu.

---

### ✅ Cell 9: Auto-detect Columns + Format Alpaca
**Check**:
```
✓ Cột dùng: instruction='instruction_vi', input='input_vi', output='output_vi'
```
**Map**: 0% → 100%

---

### ✅ Cell 10: Token Length Analysis
**Check**:
- `p50=..., p95=..., p99=...`
- `✓ Chọn max_seq_length = ... (power of 2, capped at 1024)`
- Histogram plot hiện ra

**Action**: Ghi lại `p95` và `MAX_SEQ_LENGTH` để điền vào REPORT.md

---

### ✅ Cell 11: Train/Eval Split
**Check**: `✓ Train: 180  |  Eval: 20`

---

### ✅ Cell 12: Markdown — Section 2: Load Model + LoRA

### ✅ Cell 13: Load Model + Wrap LoRA r=16
**Thời gian**: ~2-3 phút (download 3B model)
**Check**:
```
🦥 Unsloth: Tesla T4. Num GPUs = 1. Max memory: 14.563 GB.
✓ Trainable: 3,686,400 (0.217% of 1,702,359,040)
```
**Lỗi**: OOM ở đây → Colab reset runtime, thử lại

---

### ✅ Cell 14: Markdown — Section 3: Train r=16

### ✅ Cell 15: TRL Setup + safe_evaluate
**Check**: `✓ Trainer.__init__ patched`
**Note**: Cell này **không train**, chỉ setup

---

### ✅ Cell 16: Train Baseline r=16 ⚠️ MAIN TRAINING
**Thời gian**: ~4-5 phút
**Check**:
```
✓ r=16 done in X.X min, peak VRAM = X.X GB
✓ r=16 eval loss = 1.5161, perplexity = 4.55
```
**Plot**: Loss curve tự hiện ra (colab HTML widget)
**ACTION**: Sau khi plot xong → **Save loss_curve.png**:
- Right-click trên plot → "Save image as..." → lưu vào `/content/lab21_lora_t4/loss_curve.png`
- Hoặc: `plt.savefig('/content/lab21_lora_t4/loss_curve.png', dpi=150, bbox_inches='tight')`

---

### ✅ Cell 17: Plot Loss Curve
**Check**: Plot hiện ra với 1 đường train loss (không có eval vì t4 tắt eval)
**ACTION**: Nếu plot không tự save → chạy cell bổ sung:
```python
plt.savefig('/content/lab21_lora_t4/loss_curve.png', dpi=150, bbox_inches='tight')
print("✓ Saved loss_curve.png")
```

---

### ✅ Cell 18: Markdown — Section 4: Rank Experiment

### ✅ Cell 19: `train_one_rank()` function
> **Không chạy riêng** — chỉ định nghĩa function

---

### ✅ Cell 20: Cleanup + Train r=8 ⚠️ TRAINING
**Thời gian**: ~4 phút
**Check**:
```
✓ r=8 adapter saved. Computing eval loss...
```
**Sau khi xong**: Verify `/content/lab21_lora_t4/r8/` tồn tại

---

### ✅ Cell 21: Cleanup + Train r=64 ⚠️ TRAINING
**Thời gian**: ~4 phút
**Check**:
```
✓ r=64 adapter saved. Computing eval loss...
```

---

### ✅ Cell 22: Build Summary Table
**Check**: Bảng so sánh 3 ranks:
```
 rank  alpha  trainable_params  train_time_min  peak_vram_gb  eval_loss  eval_perplexity
    8     16           1843200        ...           ...           1.5577       4.7479
   16     32           3686400        ...           ...           1.5161       4.5544
   64    128          14745600        ...           ...           1.4768       4.3790
```
**ACTION**: Copy toàn bộ bảng này → paste vào REPORT.md Section 2

---

### ✅ Cell 23: Markdown — Section 5: Evaluation

### ✅ Cell 24: Test Prompts (10 prompts)
**Check**: `✓ 10 test prompts`

---

### ✅ Cell 25: Generate Qualitative Comparison ⚠️ GENERATION
**Thời gian**: ~2-3 phút (10 generations × 2 models = 20 forward passes)
**Check**:
- 5 prompts hiện ra với BASE vs FT responses
- Có cả improved và degraded cases

---

### ✅ Cell 26: Save Qualitative CSV
**Check**: Qualitative comparison saved

---

### ✅ Cell 27: Markdown — Section 6: Save + Report

### ✅ Cell 28: Save Summary CSV + Cost
**Check**:
```
✓ Tổng training time: X.X phút
✓ Estimated cost: $0.0X (@ $0.35/hr)
```

---

### ✅ Cell 29: Optional HF Hub Push
> **Skip** nếu nộp Option A. Nếu muốn bonus +5pts → uncomment và push.

---

### ✅ Cell 30: Markdown — Done Checklist
**Verify OUTPUT_DIR**:
```bash
!ls -lh /content/lab21_lora_t4/
```
Cần thấy:
- [ ] `r8/` directory
- [ ] `r16/` directory
- [ ] `r64/` directory
- [ ] `rank_experiment_summary.csv`
- [ ] `qualitative_comparison.csv`
- [ ] `loss_curve.png` ← **đã export ở cell 17**

---

## Sau Khi Training Xong — Cần Làm Ngay

### 1. Export loss_curve.png
```python
import matplotlib.pyplot as plt
# Re-generate plot để save
def plot_losses(log_history, title="Training Loss"):
    import pandas as pd
    df = pd.DataFrame(log_history)
    train = df[df["loss"].notna()] if "loss" in df else pd.DataFrame()
    plt.figure(figsize=(8, 4))
    if not train.empty:
        plt.plot(train["step"], train["loss"], label="train", color="#0E2A52")
    plt.xlabel("Step"); plt.ylabel("Loss"); plt.title(title)
    plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig('/content/lab21_lora_t4/loss_curve.png', dpi=150, bbox_inches='tight')
    print("✓ Saved loss_curve.png")

plot_losses(trainer_16.state.log_history, title="Loss Curve — r=16")
```

### 2. Compute Base Model Perplexity ⚠️ (bắt buộc cho REPORT.md)
Tạo **cell mới** sau cell 30, paste đoạn bên dưới. **Không dùng `make_trainer`** vì nó hardcode `eval_strategy="no"` — cần dùng trực tiếp `safe_evaluate()`:

```python
import gc
gc.collect(); torch.cuda.empty_cache()

# Load base model (không có adapter)
base_m, base_tok = load_base_model()

# Tạo trainer tạm với eval bật — truyền overrides để ghi đè eval_strategy
base_trainer = make_trainer(
    base_m, base_tok, train_ds, eval_ds, "tmp_base",
    eval_strategy="epoch"   # ← ghi đè: bật eval thay vì "no"
)

# safe_evaluate bỏ qua buggy NotebookProgressCallback + xử lý OOM
base_eval_loss = safe_evaluate(base_trainer)
base_ppl = float(np.exp(base_eval_loss))

print(f"✅ Base eval_loss = {base_eval_loss:.4f}")
print(f"✅ Base perplexity = {base_ppl:.2f}")

# Cleanup
del base_trainer, base_m
gc.collect(); torch.cuda.empty_cache()
```

**Thời gian**: ~1-2 phút
**Check**: Output dạng:
```
✅ Base eval_loss = 1.8XXX
✅ Base perplexity = 6.XX
```

**ACTION**: Ghi lại cả 2 số → điền vào REPORT.md Section 2 bảng "Base":
- `Eval Loss` = giá trị `base_eval_loss`
- `Perplexity` = giá trị `base_ppl`

> **Tại sao cần base perplexity**: Baseline zero-shot perplexity cho biết fine-tuned model thực sự cải thiện bao nhiêu so với chưa fine-tune. Nếu base perplexity ~6 và r=16 perplexity ~4.5 → fine-tuning giảm ~25%.

### 3. Download files về máy
```python
from google.colab import files
import shutil

# Download loss_curve.png
files.download('/content/lab21_lora_t4/loss_curve.png')

# Zip toàn bộ output (nếu chưa mount drive)
!cd /content && zip -r lab21_output.zip lab21_lora_t4/
files.download('/content/lab21_output.zip')
```

### 4. Verify CSV files
```python
import pandas as pd
summary = pd.read_csv('/content/lab21_lora_t4/rank_experiment_summary.csv')
print(summary)
qual = pd.read_csv('/content/lab21_lora_t4/qualitative_comparison.csv')
print(qual.head())
```

---

## Common Pitfalls & Fixes

| Vấn đề | Dấu hiệu | Fix |
|--------|----------|-----|
| **OOM khi load model** | `torch.cuda.OutOfMemoryError` | Runtime → Factory reset runtime → Thử lại |
| **OOM khi eval** | Crash ở cell 16/20/21 sau training | Bình thường — adapter đã save, eval OOM không ảnh hưởng |
| **Tokenizer warning** | `pad_token = <|PAD_TOKEN|>` | Bỏ qua — Unsloth tự xử lý |
| **warmup_ratio deprecation** | Warning ở đầu training | Bỏ qua — không ảnh hưởng kết quả |
| **HF_TOKEN timeout** | Warning khi load dataset | Bỏ qua — vẫn chạy được không authenticated |
| **Download bị lỗi** | `Error 429` khi download | Retry hoặc mount Drive để copy trực tiếp |

---

## Thứ tự Download về máy

1. **loss_curve.png** — từ Colab `/content/lab21_lora_t4/loss_curve.png`
2. **rank_experiment_summary.csv** — từ `/content/lab21_lora_t4/rank_experiment_summary.csv`
3. **qualitative_comparison.csv** — từ `/content/lab21_lora_t4/qualitative_comparison.csv`
4. **r16 adapter folder** — từ `/content/lab21_lora_t4/r16/`
   - Cần: `adapter_model.safetensors` + `adapter_config.json`
   - (skip tokenizer files)
5. **Base perplexity** — ghi từ output cell ở bước "Compute Base Model Perplexity"

---

## Điền số liệu vào REPORT.md

Sau khi chạy xong, điền vào REPORT.md:

| Trường | Cell nguồn | Ghi chú |
|--------|-----------|---------|
| max_seq_length | Cell 10 output | p95 rounded up |
| p50, p95, p99 | Cell 10 output | Ghi lại từ print |
| VRAM r=8/16/64 | Cell 20/16/21 output | peak_vram_gb |
| Train time r=8/16/64 | Cell 20/16/21 output | train_time_min |
| Eval loss r=8/16/64 | Cell 22 table | eval_loss |
| Perplexity r=8/16/64 | Cell 22 table | eval_perplexity |
| **Base perplexity** | **Cell base perplexity** ⭐ | **BẮT BUỘC** — dùng safe_evaluate trên base model |
| Total cost | Cell 28 output | Estimated cost |
| Qualitative examples | Cell 25 output | Copy 5 examples |
| Loss curve observation | Cell 17 plot | Quan sát train loss trend |

> ⭐ **Base perplexity** là số quan trọng nhất để so sánh. Nếu không đo được (OOM), ghi chú "~6.0 (ước tính)" và giải thích lý do trong REPORT.

---

**Sau khi điền đủ số liệu** → chạy `package_submission.py` để tạo zip → nộp LMS.
