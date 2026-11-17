"""
02_train_vit.py

Train a tiny masked-patch ViT prior with common/training_rectangle.run_training
(the routine paper/train_MRF_model_A.py calls): paper architecture, few epochs.
With so few epochs a linear 0.01 -> 0.99 mask-rate schedule replaces the
paper's 99-epoch cyclic schedule, so every mask level is seen.

Output: out/vit_A/vocab_checkpoint.pt, out/vit_A/model_final.pth,
        out/vit_A/epoch_loss.csv, out/vit_A/epoch_accuracy.csv
"""
import os
import time

import numpy as np
import torch

from run_all import (BATCH_SIZE, EPOCHS, LEARNING_RATE, PATCH_SIZE, VIT_DIR, VIT_KW,
                     VOCAB_CAP, out, seed_everything)
from training_rectangle import run_training

seed_everything()
os.makedirs(VIT_DIR, exist_ok=True)
train = torch.tensor(np.load(out("mrf_A_train.npz"))["images"], dtype=torch.float32)

t0 = time.time()
model, last_loss = run_training(
    training_data=train, nr_epochs=EPOCHS, batch_size=BATCH_SIZE,
    mask_rate=0.01, final_mask_rate=0.99, mask_schedule="linear",
    patch_size=PATCH_SIZE, learning_rate=LEARNING_RATE, vocab_cap=VOCAB_CAP,
    use_pos_emb=True, use_rel_bias=True, foldername=VIT_DIR, **VIT_KW,
)
torch.save(model.cpu().state_dict(), os.path.join(VIT_DIR, "model_final.pth"))
print(f"Final masked-CE loss {last_loss:.4f}; trained in {time.time() - t0:.1f} s")
