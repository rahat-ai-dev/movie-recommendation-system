"""
Diagnose WHY neural's ranking is ~random despite a decent val_rmse.

Hypothesis: item embeddings have collapsed (low variance across items), so
the model differentiates USERS (average rating level) fine, but barely
differentiates ITEMS for a given user -- which is exactly what ranking
needs and RMSE doesn't measure.

Run:
    python scripts/diagnose_neural.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch

from src import config
from src.models.neural import NeuralRecommender

neural = NeuralRecommender.load(config.MODEL_DATASET_DIR / "neural_ncf.pt")
neural.model.eval()

item_emb = neural.model.item_embed.weight.detach().numpy()
user_emb = neural.model.user_embed.weight.detach().numpy()

print(f"n_items={item_emb.shape[0]}  n_users={user_emb.shape[0]}  embed_dim={item_emb.shape[1]}")
print(f"item_embed  per-dim std: mean={item_emb.std(axis=0).mean():.5f}  "
      f"(user_embed per-dim std for comparison: {user_emb.std(axis=0).mean():.5f})")
print(f"item_embed row-norm:  mean={np.linalg.norm(item_emb, axis=1).mean():.4f}  "
      f"std={np.linalg.norm(item_emb, axis=1).std():.4f}")

# For 3 random users, how spread out are the predicted scores across ALL items?
# If the model discriminates items at all, this spread should be a meaningful
# fraction of the 0.5-5.0 rating range. If it's tiny, predictions ~ user-bias only.
with torch.no_grad():
    rng = np.random.default_rng(0)
    sample_users = rng.choice(list(neural.user_to_idx.values()), size=3, replace=False)
    all_items = torch.arange(item_emb.shape[0])
    for u_idx in sample_users:
        u_tensor = torch.full_like(all_items, int(u_idx))
        scores = neural.model(u_tensor, all_items).numpy()
        print(f"user_idx={u_idx}:  score min={scores.min():.3f}  max={scores.max():.3f}  "
              f"std={scores.std():.4f}  (range should be a real chunk of 0.5-5.0 if items are discriminated)")

    # Cross-user overlap of top-20 by raw score (before excluding seen items).
    # If different users get near-identical top lists, the model is mostly
    # ranking by an item-only signal that doesn't correlate with who-watches-what.
    top_sets = []
    sample_users2 = rng.choice(list(neural.user_to_idx.values()), size=5, replace=False)
    for u_idx in sample_users2:
        u_tensor = torch.full_like(all_items, int(u_idx))
        scores = neural.model(u_tensor, all_items).numpy()
        top20 = set(np.argsort(-scores)[:20].tolist())
        top_sets.append(top20)
    pairs = [(i, j) for i in range(len(top_sets)) for j in range(i+1, len(top_sets))]
    overlaps = [len(top_sets[i] & top_sets[j]) for i, j in pairs]
    print(f"\ntop-20 overlap across 5 random users (out of 20 possible): {overlaps}  "
          f"(near-20 = model ranks almost the same items for everyone; near-0 = genuinely per-user)")