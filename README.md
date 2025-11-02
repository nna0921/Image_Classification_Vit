# Image_Classification_ViT

Project 03: Fine Tuning Using Different Architectures — Task 3 (ViT)
Group assignment (max 2 persons) — Author: nna0921

---

## Overview

This repository contains code, scripts, and instructions to fine-tune a pre-trained Vision Transformer (ViT) model to classify food images from the Food-41 dataset (Kaggle). This work was completed as part of Project 03: "Fine Tuning Using Different Architectures" for the course assignment. The focus for Task 3 is to fine-tune an image transformer (vit-base-patch16-224) for image classification.

Key goals:
- Fine-tune a pre-trained ViT model on Food-41
- Provide data preprocessing and augmentation
- Train / validate and save model checkpoints
- Evaluate using accuracy, confusion matrix, and sample predictions
- Deploy a demo (Streamlit or Gradio) for image upload and prediction

Submission deliverables (what this repo should include):
1. Full code and README instructions (this file)
2. Deployed demo link (Streamlit or Gradio) — add your live URL to the Demo section
3. Evaluation report with metrics and sample outputs (place in `reports/`)
4. Medium blog post draft explaining dataset, model, and results (place in `blog/` or provide external link)

---

## Repository structure (suggested)

- data/                        # dataset downloads or symlinks (not committed)
- src/
  - data_prep.py               # data preprocessing & augmentation
  - train_vit.py               # training / fine-tuning script for ViT
  - evaluate.py                # evaluation scripts (accuracy, confusion matrix)
  - infer.py                   # single image inference / demo backend
  - utils.py                   # helpers (transforms, dataset loader)
- demos/
  - app_streamlit.py           # Streamlit demo (image upload + predict)
  - app_gradio.py              # Gradio demo alternative
- outputs/                     # training outputs, logs, checkpoints
- reports/
  - evaluation_report.pdf      # final evaluation report (add after experiments)
- blog/
  - draft_medium.md            # blog post draft
- requirements.txt             # python dependencies
- README.md                    # this file

Adjust the structure if your code uses different filenames; the README commands below assume the listed filenames.

---

## Dataset

Food-41 (Kaggle)
- URL: https://www.kaggle.com/datasets/kmader/food41
- Download the dataset and place it under `data/food41` or point your scripts to the path where you store the dataset.

Notes:
- The dataset contains image files labeled by food class. Follow the Kaggle terms of use when downloading/hosting the dataset.

---

## Environment & Dependencies

A minimal set of Python packages (see `requirements.txt`):
- Python 3.8+
- torch (compatible version for your CUDA)
- torchvision
- transformers
- timm (optional, if using timm utilities)
- accelerate (optional; recommended for multi-GPU)
- datasets (optional)
- scikit-learn (for metrics & confusion matrix)
- matplotlib/seaborn (for plots)
- Pillow
- streamlit or gradio (for demo)
- bitsandbytes (optional for 4-bit quantization)
- wandb (optional for logging)

Install with:
pip install -r requirements.txt

If using CUDA-enabled GPUs, install the correct torch build for your CUDA version.

---

## How to prepare the data

1. Download Food-41 from Kaggle and extract into `data/food41`:

   - Expected layout:
     data/food41/train/<class_name>/*.jpg
     data/food41/val/<class_name>/*.jpg
     (Or a single folder with CSV labels — modify `data_prep.py` accordingly.)

2. Run the preprocessing script (example):
   python src/data_prep.py --data-dir data/food41 --output-dir data/processed --img-size 224 --val-split 0.2

The preprocessing script performs:
- Resize / center-crop to the ViT patch size (224)
- Standard augmentation for training (random crop, horizontal flip, color jitter — configurable)
- Generates PyTorch Dataset / DataLoader-ready directories or a manifest file

---

## Training / Fine-tuning (ViT)

This project uses a Hugging Face / torchvision-compatible ViT model. Suggested model id:
- `google/vit-base-patch16-224` (Hugging Face model hub)

Example training command (single-GPU):
python src/train_vit.py \
  --data-dir data/processed \
  --model_name_or_path google/vit-base-patch16-224 \
  --output_dir outputs/vit_finetuned \
  --epochs 4 \
  --batch_size 16 \
  --learning_rate 3e-5 \
  --weight_decay 0.01 \
  --img_size 224 \
  --fp16 \
  --gradient_accumulation_steps 2 \
  --save_steps 500

Notes and tips:
- If memory is limited:
  - Lower batch_size and use gradient accumulation (--gradient_accumulation_steps).
  - Enable FP16 mixed precision training (`--fp16`).
  - Use `accelerate` or PyTorch native DistributedDataParallel for multi-GPU.
- 3–5 epochs are often sufficient to see strong transfer performance on Food-41.
- Save checkpoints regularly (use `--save_steps` or `--save_every_epoch`).
- Optionally, use 4-bit quantization (bitsandbytes) for inference / memory reduction:
  --quantize 4 --use_bitsandbytes
  Make sure to install and configure bitsandbytes appropriately; compatibility varies by hardware.

Accelerate example:
accelerate launch src/train_vit.py --config_file accelerate_config.yaml --other-args ...

---

## Example inference / demo

To run single-image inference:
python src/infer.py --model outputs/vit_finetuned/checkpoint-best --image examples/test.jpg --img_size 224

To run the Streamlit demo:
streamlit run demos/app_streamlit.py --server.port 8501

To run the Gradio demo:
python demos/app_gradio.py

Add your deployed demo URL here once deployed:
Demo URL: https://<your-demo-host>.app  (update after deployment)

---

## Evaluation

The repository provides `src/evaluate.py` which:
- Computes top-1 accuracy on validation/test set
- Generates a confusion matrix and class-wise precision/recall/F1
- Saves sample predictions with input image + predicted label + confidence

Example:
python src/evaluate.py --ckpt outputs/vit_finetuned/checkpoint-best --data-dir data/processed --output reports/eval_results.json

Recommended metrics:
- Accuracy (top-1)
- Precision / Recall / F1 per class (Food-41 has many classes)
- Confusion matrix visualization for common confusions

Include these artifacts in your `reports/` folder for submission.

---

## Reproducibility & Logging

- Log training metrics and artifacts (e.g., with WandB, TensorBoard, or simple CSV logs).
- Store random seeds and environment information to `outputs/` or `reports/`.
- Save model checkpoints frequently and keep at least one checkpoint with the best validation accuracy.

---

## Notes about Hugging Face / Tokenizers

- For ViT, tokenizers are not required (image patches are used). Using Hugging Face `AutoImageProcessor` or `ViTImageProcessor` is recommended to replicate the original pre-processing pipeline of the chosen pre-trained model.
- Using a Hugging Face model gives you:
  1. Model architecture
  2. Preprocessing utilities (image processor)
  3. Pre-trained weights

Example:
from transformers import AutoImageProcessor, ViTForImageClassification
processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")
model = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224", num_labels=41)

---

## Practical tips (from evaluation instructions)

- If memory issues arise, use gradient accumulation.
- Enable FP16 training to reduce memory usage (mixed precision).
- 3-5 epochs generally work well for transfer learning here.
- You may explore 4-bit quantization for inference (use bitsandbytes), but ensure correctness.
- Keep checkpoints and a clear evaluation report with sample predictions.

---

## Deliverables checklist

- [ ] GitHub repo with code, README, and scripts (this repo)
- [ ] Deployed demo (Streamlit or Gradio) — add final link above
- [ ] Evaluation report in `reports/` including metrics, confusion matrix, and sample predictions
- [ ] Blog post draft in `blog/draft_medium.md` describing dataset, model choices, training details, and results
- [ ] Saved model checkpoints in `outputs/` (regularly during training)

---

## Academic integrity / AI usage

Per the assignment rules:
- AI tools may be used for help (debugging, snippets, learning), but not for generating the full project code. Entirely AI-generated submissions will receive 0.
- Record any AI assistance briefly in your report (e.g., "Used ChatGPT for debugging and small code snippets").

---

## License & Acknowledgements

- Data: Food-41 dataset (Kaggle). Follow Kaggle's dataset Terms of Use.
- Model: ViT (Google / Hugging Face). See respective licenses at Hugging Face model page.
- This assignment was completed for an academic project (Project 03: Fine Tuning Using Different Architectures).

---

## Contact

Author: nna0921
If you have questions about the code or need help reproducing results, open an issue or email the project owner.

