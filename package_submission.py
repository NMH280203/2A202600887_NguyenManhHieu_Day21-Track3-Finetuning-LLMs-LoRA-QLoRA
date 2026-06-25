#!/usr/bin/env python3
"""
Lab 21 — Packaging script
Option A: Lightweight ZIP
Đặt file này cùng folder với OUTPUT_DIR từ Colab (vd: ~/lab21_lora_t4/)
chạy: python package_submission.py
"""

import os, zipfile, shutil, pathlib

# ===== CẤU HÌNH =====
MSSV = "2A202600887"
HO_TEN = "NguyenManhHieu"

# Thư mục làm việc hiện tại (nơi chạy script này)
WORK_DIR = pathlib.Path(__file__).parent.resolve()

# OUTPUT_DIR từ Colab đã download về máy (lab21_lora_t4/)
COLAB_OUTPUT = str(WORK_DIR / "lab21_lora_t4")
ZIP_NAME = f"lab21_{MSSV}_{HO_TEN}.zip"

# ===== CHECK FILES =====
print("🔍 Checking required files...")

errors = []
output_dir = pathlib.Path(COLAB_OUTPUT)

required = {
    "r16_adapter":     output_dir / "r16" / "adapter_model.safetensors",
    "rank_csv":        output_dir / "rank_experiment_summary.csv",
    "qual_csv":        output_dir / "qualitative_comparison.csv",
    "loss_curve_png":   output_dir / "loss_curve.png",        # bạn cần export/screenshot
    "report_md":        WORK_DIR / "REPORT.md",
    "notebook_ipynb":  WORK_DIR / "notebooks" / "Lab21_LoRA_Finetuning_T4.ipynb",
}

for name, path in required.items():
    if path.exists():
        size = path.stat().st_size / 1024
        print(f"  ✅ {name}: {size:.1f} KB")
    else:
        errors.append(f"  ❌ MISSING: {name} at {path}")

if errors:
    print("\n⚠️  File check FAILED:")
    for e in errors:
        print(e)
    print("\nKhắc phục trước khi chạy lại script này.")
    exit(1)

print("\n✅ Tất cả file có sẵn. Bắt đầu đóng gói...")

# ===== STRIP NOTEBOOK OUTPUTS (in-memory only — không sửa file gốc) =====
print("\n📦 Stripping notebook outputs for zip...")
nb_path = required["notebook_ipynb"]

import json
with open(nb_path) as f:
    nb = json.load(f)

stripped_cells = 0
for cell in nb.get("cells", []):
    if cell.get("cell_type") == "code":
        cell["outputs"] = []
        cell["execution_count"] = None
        stripped_cells += 1

stripped_nb_path = WORK_DIR / "notebook_stripped.ipynb"
with open(stripped_nb_path, "w") as f:
    json.dump(nb, f)

print(f"  ✅ Đã strip {stripped_cells} code cells → {stripped_nb_path.name}")

# ===== BUILD ZIP =====
zip_path = WORK_DIR / ZIP_NAME
if zip_path.exists():
    zip_path.unlink()

zipf = zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED)

# 1. REPORT.md
zipf.write(required["report_md"], f"REPORT.md")
print("  ✅ REPORT.md")

# 2. Stripped notebook
zipf.write(stripped_nb_path, f"notebook.ipynb")
print("  ✅ notebook.ipynb (stripped)")

# 3. adapters/r16/ (chỉ safetensors + adapter_config per rubric Option A)
adapter_src = output_dir / "r16"
adapter_files = [
    adapter_src / "adapter_model.safetensors",
    adapter_src / "adapter_config.json",
]
for f in adapter_files:
    if not f.exists():
        raise FileNotFoundError(f"Missing adapter file: {f}")
    arcname = f"adapters/r16/{f.name}"
    zipf.write(f, arcname)
print(f"  ✅ adapters/r16/ ({len(adapter_files)} files)")

# 4. results/
csv_dir = output_dir / "rank_experiment_summary.csv"
qual_csv_dir = output_dir / "qualitative_comparison.csv"
loss_dir = output_dir / "loss_curve.png"

for src_path, arcname in [
    (csv_dir,   "results/rank_experiment_summary.csv"),
    (qual_csv_dir, "results/qualitative_comparison.csv"),
    (loss_dir, "results/loss_curve.png"),
]:
    if src_path.exists():
        zipf.write(src_path, arcname)
        print(f"  ✅ {arcname}")

zipf.close()

# Cleanup temp stripped notebook
stripped_nb_path.unlink(missing_ok=True)

# ===== DONE =====
zip_size = zip_path.stat().st_size / 1024 / 1024
print(f"\n✅ Hoàn tất: {ZIP_NAME} ({zip_size:.2f} MB)")
print(f"   Location: {zip_path}")
print("\n📋 Nội dung zip:")
with zipfile.ZipFile(zip_path) as z:
    for info in z.infolist():
        print(f"   {info.filename}  ({info.file_size/1024:.1f} KB)")
