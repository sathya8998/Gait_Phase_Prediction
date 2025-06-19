import os
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import autocast, GradScaler
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# ─────────────────────────── Model ──────────────────────────── #
class CNNBiLSTM(nn.Module):
    def __init__(self, input_size=21, conv_channels=(64, 128),
                 hidden_size=64, num_layers=2, dropout=0.3):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(input_size, conv_channels[0], kernel_size=3, padding=1),
            nn.ReLU(), nn.BatchNorm1d(conv_channels[0]),
            nn.Conv1d(conv_channels[0], conv_channels[1], kernel_size=3, padding=1),
            nn.ReLU(), nn.BatchNorm1d(conv_channels[1]),
        )
        self.bilstm = nn.LSTM(input_size=conv_channels[1],
                              hidden_size=hidden_size,
                              num_layers=num_layers,
                              dropout=dropout,
                              batch_first=True,
                              bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, 21)

    def forward(self, x):      # x: (B, 31, 21)
        x = x.transpose(1, 2)  # → (B, 21, 31)
        x = self.conv(x)       # → (B, 128, 31)
        x = x.transpose(1, 2)  # → (B, 31, 128)
        out, _ = self.bilstm(x)# → (B, 31, 2*hid)
        last = out[:, -1, :]   # → (B, 2*hid)
        return self.fc(last)   # → (B, 21)


def train_and_predict(loader, X_windows, device):
    model = CNNBiLSTM().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    epochs = 1
    criterion = nn.MSELoss()
    scaler = GradScaler(enabled=(device.type == "cuda"))
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=3e-3, epochs=epochs,
        steps_per_epoch=len(loader),
        pct_start=0.3, div_factor=25,
        final_div_factor=1e4,
        anneal_strategy="cos",
    )

    best_loss, best_state = float("inf"), None
    torch.set_float32_matmul_precision("medium")

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            with autocast(device_type=device.type):
                pred = model(xb)
                loss = criterion(pred, yb)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        lr_now = scheduler.get_last_lr()[0]
        print(f"Epoch {epoch:02d}/{epochs}  •  loss={avg_loss:.4f}  •  lr={lr_now:.2e}")
        if avg_loss < best_loss:
            best_loss, best_state = avg_loss, model.state_dict()

    model.load_state_dict(best_state)
    model.eval()

    with torch.no_grad():
        pred_norm = model(torch.from_numpy(X_windows).to(device)).cpu().numpy()

    return model, pred_norm, best_loss


def process_pid(pid: str):
    print(f"\n=== Processing PID {pid} ===")

    base_dir = os.path.join(os.getcwd(), "Data", pid)
    orig_csv = os.path.join(base_dir, "emg_all_samples_features.csv")
    recon_csv = os.path.join(base_dir, "reconstructed_full_emg_from_best_method.csv")

    # ── LOAD & ALIGN ──
    if not os.path.exists(orig_csv) or not os.path.exists(recon_csv):
        print(f"  ⚠️  Missing files for PID {pid}. Skipping.")
        return

    df_o = pd.read_csv(orig_csv).fillna(0)
    df_r = pd.read_csv(recon_csv).fillna(0)
    n = min(len(df_o), len(df_r))
    df_o, df_r = df_o.iloc[:n], df_r.iloc[:n]

    # ── FEATURE SLICES ──
    feat_o = df_o.iloc[:, 2:].to_numpy(np.float32)
    feat_r = df_r.iloc[:, :-1].to_numpy(np.float32)
    vl_slice = slice(322, 343)
    ta_slice = slice(162, 183)
    X = feat_r[:, vl_slice]
    Y = feat_o[:, ta_slice]

    # ── NORMALIZE ──
    scaler_X = MinMaxScaler().fit(X)
    scaler_Y = MinMaxScaler().fit(Y)
    Xn = scaler_X.transform(X)
    Yn = scaler_Y.transform(Y)

    # ── WINDOWING ──
    win, half = 31, 31 // 2
    Xw, Yw = [], []
    for i in range(half, len(Xn) - half):
        Xw.append(Xn[i - half : i + half + 1])
        Yw.append(Yn[i])
    Xw = np.stack(Xw)
    Yw = np.stack(Yw)

    # ── TRAIN/UNSEEN SPLIT ──
    Xw_train, Xw_unseen, Yw_train, Yw_unseen = train_test_split(
        Xw, Yw, test_size=0.2, random_state=42
    )

    # ── DATALOADER ──
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = (device.type == "cuda")
    loader = DataLoader(
        TensorDataset(torch.from_numpy(Xw_train), torch.from_numpy(Yw_train)),
        batch_size=64,
        shuffle=True,
        num_workers=8 if device.type == "cuda" else 0,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(device.type == "cuda"),
    )

    # ── TRAIN ON SPLIT ──
    model, pred_norm_train, best_loss = train_and_predict(loader, Xw_train, device)
    print(f"  🏆 Best training loss for PID {pid}: {best_loss:.4f}")

    # ── SAVE & PLOT TRAIN RESULTS ──
    corrs_train = [
        pearsonr(Yw_train[:, i], pred_norm_train[:, i])[0] for i in range(21)
    ]
    train_corrs_df = pd.DataFrame({
        "Channel": np.arange(1, 22),
        "Pearson_r": corrs_train
    })
    train_corrs_path = os.path.join(base_dir, "train_ta_correlations.csv")
    train_corrs_df.to_csv(train_corrs_path, index=False)
    print(f"  ✅ Saved train correlations CSV to {train_corrs_path}")

    # ── PREDICT & SAVE “UNSEEN” RESULTS ──
    model.eval()
    with torch.no_grad():
        pred_norm_un = model(torch.from_numpy(Xw_unseen).to(device)).cpu().numpy()
    pred_un = scaler_Y.inverse_transform(pred_norm_un)

    unseen_pred_df = pd.DataFrame(
        pred_un, columns=[f"TA-{i}" for i in range(1, 22)]
    )
    unseen_pred_path = os.path.join(base_dir, "unseen_ta_predictions.csv")
    unseen_pred_df.to_csv(unseen_pred_path, index=False)

    corrs_un = [
        pearsonr(Yw_unseen[:, i], pred_norm_un[:, i])[0] for i in range(21)
    ]
    unseen_corrs_df = pd.DataFrame({
        "Channel": np.arange(1, 22),
        "Pearson_r": corrs_un
    })
    unseen_corrs_path = os.path.join(base_dir, "unseen_ta_correlations.csv")
    unseen_corrs_df.to_csv(unseen_corrs_path, index=False)
    print(f"  ✅ Saved unseen predictions to {unseen_pred_path}")
    print(f"  ✅ Saved unseen correlations to {unseen_corrs_path}")

    # ── SUMMARY ──
    print(
        f"  ▶︎ Train  r: mean={np.mean(corrs_train):.4f}, "
        f"min/max={np.min(corrs_train):.4f}/{np.max(corrs_train):.4f}"
    )
    print(
        f"  ▶︎ Unseen r: mean={np.mean(corrs_un):.4f}, "
        f"min/max={np.min(corrs_un):.4f}/{np.max(corrs_un):.4f}"
    )


def main():
    # Loop over PIDs "001" through "007"
    for i in range(1, 8):
        pid = f"{i:03d}"
        process_pid(pid)


if __name__ == "__main__":
    main()
