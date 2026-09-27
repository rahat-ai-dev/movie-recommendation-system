"""
Neural recommender (spec section 14): embedding-MLP Neural Collaborative
Filtering in PyTorch.

    userId --> UserEmbedding --\\
                                 concat --> Dense(128) --> ReLU --> Dropout
    movieId -> ItemEmbedding --/          --> Dense(64) --> ReLU --> Dropout
                                           --> Dense(32) --> ReLU
                                           --> Linear(1) --> predicted rating

Trained as regression on observed ratings (MSE), with early stopping on a
held-out validation split (src/data/splitter.py). Uses CUDA automatically
if available -- on ml-32m scale (32M interactions) a GPU (e.g. Google
Colab) is much faster, but CPU-only training works too (see "CPU
performance" below).

--------------------------------------------------------------------------
CPU performance
--------------------------------------------------------------------------
Batches are built by directly slicing the full encoded rating tensors
(`torch.randperm` + fancy indexing), not via `torch.utils.data.DataLoader`.
A per-sample `Dataset.__getitem__` call, multiplied by tens of millions of
rows and re-collated every batch, is pure Python-loop overhead with no
payoff here -- the whole encoded dataset already fits in RAM as three flat
tensors. Slicing directly is one vectorized op per batch, which matters a
lot more on CPU (no GPU to hide the overhead behind) than on GPU. Larger
`NEURAL_BATCH_SIZE` (src/config.py) also helps on CPU specifically, since
fewer, bigger batches means less fixed per-step Python overhead relative
to actual compute.

--------------------------------------------------------------------------
Resumable checkpointing (built for Colab's ~40min free-tier disconnects)
--------------------------------------------------------------------------
Pass `checkpoint_dir=` to `fit()` (point it at a **Google-Drive-mounted
path**, not local /content, so it survives a disconnect) and two files are
kept there:

  checkpoint_dir/latest.pt   overwritten after every checkpoint (epoch end,
                              and every `save_every_steps` batches within an
                              epoch for big datasets) -- cheap, always fresh.
  checkpoint_dir/best.pt     only overwritten when validation loss improves
                              -- kept separately so a few bad epochs after
                              the best one can't lose it.

Each checkpoint file holds everything needed to resume exactly:
`epoch`, `global_step`, `model_state_dict`, `optimizer_state_dict`,
`train_loss`, `val_loss`, `best_val_loss`, `history`, plus the id mappings
and `embed_dim` (so resuming doesn't require re-deriving them).

To resume after a disconnect:

    NeuralRecommender.resume("checkpoint_dir/latest.pt", train, val,
                              checkpoint_dir="checkpoint_dir")

This is a thin wrapper around `fit(..., resume_from=...)` -- it loads
model + optimizer + epoch/loss history from the checkpoint and continues
training from `epoch + 1` instead of starting over.

This is separate from `.save()`/`.load()`, which write/read a small
*inference-only* artifact (state_dict + id mappings, no optimizer/epoch) --
that's the format `scripts/train_and_evaluate.py` hands to the Streamlit
app once training is fully done.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from src import config
from src.data.cleaner import build_id_mappings


class NCFNet(nn.Module):
    def __init__(self, n_users: int, n_items: int,
                 embed_dim: int = config.NEURAL_EMBED_DIM,
                 hidden_layers=config.NEURAL_HIDDEN_LAYERS,
                 dropout: float = config.NEURAL_DROPOUT):
        super().__init__()
        self.user_embed = nn.Embedding(n_users, embed_dim)
        self.item_embed = nn.Embedding(n_items, embed_dim)
        nn.init.normal_(self.user_embed.weight, std=0.05)
        nn.init.normal_(self.item_embed.weight, std=0.05)

        layers = []
        in_dim = embed_dim * 2
        for h in hidden_layers:
            layers += [nn.Linear(in_dim, h), nn.ReLU(), nn.Dropout(dropout)]
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, user_idx, item_idx):
        u = self.user_embed(user_idx)
        i = self.item_embed(item_idx)
        x = torch.cat([u, i], dim=-1)
        return self.mlp(x).squeeze(-1)


class NeuralRecommender:
    def __init__(self, embed_dim=config.NEURAL_EMBED_DIM, lr=config.NEURAL_LR,
                 batch_size=config.NEURAL_BATCH_SIZE, epochs=config.NEURAL_EPOCHS,
                 patience=config.NEURAL_EARLY_STOP_PATIENCE, seed=config.NEURAL_SEED):
        self.embed_dim, self.lr = embed_dim, lr
        self.batch_size, self.epochs, self.patience = batch_size, epochs, patience
        self.seed = seed
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---------------------------------------------------------------- fit --
    def fit(self, train: pd.DataFrame, val: pd.DataFrame | None = None, verbose: bool = True,
            checkpoint_dir: Path | str | None = None,
            save_every_steps: int | None = None,
            resume_from: Path | str | None = None):
        """
        checkpoint_dir:   where to write latest.pt / best.pt. Point this at a
                          Google-Drive-mounted path on Colab so a disconnect
                          doesn't lose progress (see module docstring).
        save_every_steps: also checkpoint mid-epoch every N batches, not just
                          at epoch end. If left as None it's picked
                          automatically from the dataset size (an epoch on
                          32M rows can itself take longer than a Colab
                          disconnect window, so it's checkpointed several
                          times per epoch; a small dataset just checkpoints
                          at epoch end).
        resume_from:      path to a previously-saved checkpoint file (usually
                          checkpoint_dir/"latest.pt") to continue from,
                          instead of training from scratch. See
                          `NeuralRecommender.resume(...)` for the usual way
                          to call this.
        """
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        if verbose:
            print(f"[NCF] training on device: {self.device}"
                  + ("" if self.device.type == "cuda" else
                     "  (no GPU detected -- this will be slow on 32M rows; "
                     "run this on Google Colab with a GPU runtime for real training)"), flush=True)

        if checkpoint_dir is not None:
            checkpoint_dir = Path(checkpoint_dir)
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
        latest_path = checkpoint_dir / "latest.pt" if checkpoint_dir else None
        best_path = checkpoint_dir / "best.pt" if checkpoint_dir else None

        self.user_to_idx, self.idx_to_user, self.movie_to_idx, self.idx_to_movie = build_id_mappings(train)
        n_users, n_items = len(self.user_to_idx), len(self.movie_to_idx)

        def encode(df):
            u = df["userId"].map(self.user_to_idx).to_numpy()
            m = df["movieId"].map(self.movie_to_idx).to_numpy()
            r = df["rating"].to_numpy(dtype=np.float32)
            return u, m, r

        # Tensors are built ONCE, in full, and batches are taken by direct
        # slicing (torch.randperm + fancy indexing) instead of going through
        # torch.utils.data.Dataset/DataLoader. A per-sample __getitem__ call
        # (multiplied by tens of millions of rows and re-collated every
        # batch) is pure Python-loop overhead -- pointless here since the
        # whole encoded dataset already fits comfortably in RAM as three
        # flat arrays. Slicing directly is a single vectorized op per batch
        # and is dramatically faster, especially on CPU (no GPU to hide the
        # overhead behind).
        tr_u, tr_m, tr_r = encode(train)
        tr_u = torch.as_tensor(np.array(tr_u, copy=True), dtype=torch.long)
        tr_m = torch.as_tensor(np.array(tr_m, copy=True), dtype=torch.long)
        tr_r = torch.as_tensor(np.array(tr_r, copy=True), dtype=torch.float32)
        n_train = tr_u.shape[0]
        steps_per_epoch = (n_train + self.batch_size - 1) // self.batch_size

        va_u = va_m = va_r = None
        if val is not None and len(val):
            val = val[val["userId"].isin(self.user_to_idx) & val["movieId"].isin(self.movie_to_idx)]
            if len(val):
                v_u, v_m, v_r = encode(val)
                va_u = torch.as_tensor(np.array(v_u, copy=True), dtype=torch.long)
                va_m = torch.as_tensor(np.array(v_m, copy=True), dtype=torch.long)
                va_r = torch.as_tensor(np.array(v_r, copy=True), dtype=torch.float32)

        def batches(u, m, r, batch_size, shuffle):
            n = u.shape[0]
            perm = torch.randperm(n) if shuffle else torch.arange(n)
            for start in range(0, n, batch_size):
                idx = perm[start:start + batch_size]
                yield (u[idx].to(self.device, non_blocking=True),
                       m[idx].to(self.device, non_blocking=True),
                       r[idx].to(self.device, non_blocking=True))

        # dataset-size-aware default: aim for a handful of mid-epoch saves
        # when one epoch is big enough that losing it would actually hurt.
        if checkpoint_dir is not None and save_every_steps is None and steps_per_epoch > 2000:
            save_every_steps = max(500, steps_per_epoch // 8)

        self.model = NCFNet(n_users, n_items, embed_dim=self.embed_dim).to(self.device)
        opt = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        loss_fn = nn.MSELoss()
        # computed once up front (not per checkpoint) -- groupby over 32M
        # rows is not cheap enough to repeat every time we save.
        self._seen = train.groupby("userId")["movieId"].apply(set).to_dict()

        start_epoch, global_step = 1, 0
        best_val, bad_epochs, history = float("inf"), 0, []

        # ---- resume from a checkpoint, if given -------------------------
        if resume_from is not None and Path(resume_from).exists():
            ckpt = torch.load(resume_from, map_location=self.device, weights_only=False)
            self.model.load_state_dict(ckpt["model_state_dict"])
            opt.load_state_dict(ckpt["optimizer_state_dict"])
            start_epoch = ckpt["epoch"] + 1
            global_step = ckpt.get("global_step", 0)
            best_val = ckpt.get("best_val_loss", float("inf"))
            history = ckpt.get("history", [])
            if verbose:
                print(f"[NCF] resumed from {resume_from} -> continuing at epoch {start_epoch} "
                      f"(best_val_rmse so far={best_val**0.5:.4f})", flush=True)
        elif resume_from is not None and verbose:
            print(f"[NCF] resume_from={resume_from} not found -- starting fresh instead.", flush=True)

        def checkpoint_payload(epoch, train_loss, val_loss):
            return {
                "epoch": epoch, "global_step": global_step,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": opt.state_dict(),
                "train_loss": train_loss, "val_loss": val_loss, "best_val_loss": best_val,
                "history": history,
                "user_to_idx": self.user_to_idx, "movie_to_idx": self.movie_to_idx,
                "idx_to_user": self.idx_to_user, "idx_to_movie": self.idx_to_movie,
                "embed_dim": self.embed_dim,
            }

        for epoch in range(start_epoch, self.epochs + 1):
            t0 = time.time()
            self.model.train()
            train_loss, n = 0.0, 0
            for step_in_epoch, (u, m, r) in enumerate(batches(tr_u, tr_m, tr_r, self.batch_size, shuffle=True), start=1):
                opt.zero_grad()
                pred = self.model(u, m)
                loss = loss_fn(pred, r)
                loss.backward()
                opt.step()
                train_loss += loss.item() * len(r)
                n += len(r)
                global_step += 1

                if save_every_steps and latest_path is not None and global_step % save_every_steps == 0:
                    torch.save(checkpoint_payload(epoch, train_loss / n, None), latest_path)
                    if verbose:
                        print(f"[NCF]   mid-epoch checkpoint @ epoch {epoch} step {step_in_epoch}/{steps_per_epoch} "
                              f"-> {latest_path}", flush=True)
            train_loss /= max(n, 1)

            val_loss = None
            if va_u is not None:
                self.model.eval()
                vl, vn = 0.0, 0
                with torch.no_grad():
                    for u, m, r in batches(va_u, va_m, va_r, self.batch_size, shuffle=False):
                        pred = self.model(u, m)
                        vl += loss_fn(pred, r).item() * len(r)
                        vn += len(r)
                val_loss = vl / max(vn, 1)

            history.append({"epoch": epoch, "train_rmse": train_loss ** 0.5,
                             "val_rmse": (val_loss ** 0.5) if val_loss is not None else None,
                             "seconds": round(time.time() - t0, 2)})
            if verbose:
                vs = f"val_rmse={history[-1]['val_rmse']:.4f}" if val_loss is not None else "val=n/a"
                # printed with flush + no carriage-return tricks so it shows up
                # as a normal new line in the Colab cell output every epoch.
                print(f"[NCF] epoch {epoch}/{self.epochs}  train_rmse={history[-1]['train_rmse']:.4f}  "
                      f"{vs}  ({history[-1]['seconds']:.1f}s)", flush=True)

            # ---- checkpoint: "latest" every epoch (cheap, overwritten) ----
            if latest_path is not None:
                torch.save(checkpoint_payload(epoch, train_loss, val_loss), latest_path)
                if verbose:
                    print(f"[NCF]   latest checkpoint -> {latest_path}", flush=True)

            # ---- checkpoint: "best" only kept on improvement --------------
            metric = val_loss if val_loss is not None else train_loss
            if metric < best_val - 1e-4:
                best_val, bad_epochs = metric, 0
                if best_path is not None:
                    torch.save(checkpoint_payload(epoch, train_loss, val_loss), best_path)
                    if verbose:
                        print(f"[NCF]   new best -> {best_path}  (rmse={metric**0.5:.4f})", flush=True)
            else:
                bad_epochs += 1
                if va_u is not None and bad_epochs >= self.patience:
                    if verbose:
                        print(f"[NCF] early stopping at epoch {epoch} (best rmse={best_val**0.5:.4f})", flush=True)
                    break

        # restore best-known weights for inference, preferring the on-disk
        # best checkpoint (covers the resume case where "best" came from an
        # earlier, now-gone, in-memory run).
        if best_path is not None and best_path.exists():
            ckpt = torch.load(best_path, map_location=self.device, weights_only=False)
            self.model.load_state_dict(ckpt["model_state_dict"])

        self.history_ = history
        return self

    @classmethod
    def resume(cls, checkpoint_path: Path | str, train: pd.DataFrame, val: pd.DataFrame | None = None,
               checkpoint_dir: Path | str | None = None, verbose: bool = True, **kwargs) -> "NeuralRecommender":
        """Continue an interrupted training run.

        checkpoint_path: usually checkpoint_dir/"latest.pt" from the run
                         that got disconnected.
        checkpoint_dir:  where to keep writing latest.pt/best.pt from here
                         on -- defaults to checkpoint_path's own directory.
        kwargs:          NeuralRecommender(**kwargs) constructor overrides
                         (embed_dim, lr, epochs, ...) -- leave these matching
                         the original run unless you specifically want to
                         change hyperparameters mid-training.

        Example (after a Colab disconnect):
            neural = NeuralRecommender.resume(
                "/content/drive/MyDrive/CineSignal/checkpoints/neural_ncf/latest.pt",
                train, val,
            )
        """
        checkpoint_path = Path(checkpoint_path)
        if checkpoint_dir is None:
            checkpoint_dir = checkpoint_path.parent
        obj = cls(**kwargs)
        return obj.fit(train, val, verbose=verbose, checkpoint_dir=checkpoint_dir, resume_from=checkpoint_path)

    # ------------------------------------------------------- inference ---
    @torch.no_grad()
    def get_user_recommendations(self, user_id: int, k: int = config.TOP_K_DEFAULT) -> pd.DataFrame:
        if user_id not in self.user_to_idx:
            return pd.DataFrame(columns=["movieId", "neural_score", "explanation"])

        self.model.eval()
        u_idx = self.user_to_idx[user_id]
        all_item_idx = torch.arange(len(self.movie_to_idx), device=self.device)
        u_tensor = torch.full_like(all_item_idx, u_idx)
        scores = self.model(u_tensor, all_item_idx).cpu().numpy()

        seen = self._seen.get(user_id, set())
        seen_idx = {self.movie_to_idx[m] for m in seen if m in self.movie_to_idx}
        order = np.argsort(-scores)
        results = [(self.idx_to_movie[i], float(scores[i])) for i in order if i not in seen_idx][:k]

        out = pd.DataFrame(results, columns=["movieId", "neural_score"])
        out["explanation"] = "Recommended by a neural embedding model trained on rating patterns."
        return out

    # ---------------------------------------------- deployment artifact ---
    def save(self, path: Path):
        """Small inference-only artifact (no optimizer/epoch/history) --
        what train_and_evaluate.py hands to the Streamlit app once training
        is fully done. For resumable training checkpoints, see
        `fit(checkpoint_dir=...)` / `NeuralRecommender.resume(...)` above."""
        torch.save({
            "state_dict": self.model.state_dict(),
            "user_to_idx": self.user_to_idx, "movie_to_idx": self.movie_to_idx,
            "idx_to_user": self.idx_to_user, "idx_to_movie": self.idx_to_movie,
            "embed_dim": self.embed_dim,
        }, path)

    @classmethod
    def load(cls, path: Path) -> "NeuralRecommender":
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        obj = cls(embed_dim=ckpt["embed_dim"])
        obj.user_to_idx, obj.movie_to_idx = ckpt["user_to_idx"], ckpt["movie_to_idx"]
        obj.idx_to_user, obj.idx_to_movie = ckpt["idx_to_user"], ckpt["idx_to_movie"]
        obj.model = NCFNet(len(obj.user_to_idx), len(obj.movie_to_idx), embed_dim=obj.embed_dim)
        obj.model.load_state_dict(ckpt["state_dict"])
        obj.model.to(obj.device)
        obj._seen = {}
        return obj
