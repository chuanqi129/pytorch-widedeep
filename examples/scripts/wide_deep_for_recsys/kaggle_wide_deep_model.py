# This script is mostly a copy/paste from the Kaggle notebook
# https://www.kaggle.com/code/matanivanov/wide-deep-learning-for-recsys-with-pytorch.
# Is a response to the issue:
# https://github.com/jrzaurin/pytorch-widedeep/issues/133.
# In this script we run the exact same model used in that Kaggle notebook

from pathlib import Path
import time
import numpy as np
import torch
import pandas as pd
from torch import nn, cat, mean
from scipy.sparse import coo_matrix
import argparse

# --- Argument Parsing ---
def parse_args():
    parser = argparse.ArgumentParser(description="Run Wide & Deep model benchmark.")
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use for training (e.g., 'cuda', 'xpu', 'cpu'). Auto-detects if not specified."
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="float32",
        choices=["float32", "float16", "bfloat16"],
        help="Data type for model and tensors ('float32' 'bfloat16' or 'float16'). Use float16 for mixed precision on compatible hardware."
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of training epochs."
    )
    parser.add_argument(
        "--compile",
        action="store_true", # This makes it a boolean flag. If present, it's True.
        help="Enable torch.compile optimization for the model."
    )
    return parser.parse_args()

args = parse_args()

# --- Device & Datatype Configuration ---
if args.device:
    DEVICE = args.device
else:
    if torch.cuda.is_available():
        DEVICE = "cuda"
    elif hasattr(torch, 'xpu') and torch.xpu.is_available(): # Check for Intel GPUs
        DEVICE = "xpu"
    else:
        DEVICE = "cpu"

if args.dtype == "float16":
    DTYPE = torch.float16
    if DEVICE == "cpu":
        print("Warning: float16 is typically not beneficial on CPU and may cause performance issues or errors. Using float32 instead.")
        DTYPE = torch.float32
elif args.dtype == "bfloat16":
    DTYPE = torch.bfloat16
elif args.dtype == "float32":
    DTYPE = torch.float32
else:
    raise ValueError(f"Unsupported dtype: {args.dtype}. Choose 'float32' or 'float16'.")

print(f"Using device: {DEVICE}")
print(f"Using data type for model and tensors: {DTYPE}")
print(f"Number of epochs: {args.epochs}")
print(f"torch.compile enabled: {args.compile}")

save_path = Path("prepared_data")

# --- Helper functions (as provided in original script) ---
def get_coo_indexes(lil):
    rows = []
    cols = []
    for i, el in enumerate(lil):
        if type(el) != list:
            el = [el]
        for j in el:
            rows.append(i)
            cols.append(j)
    return rows, cols


def get_sparse_features(series, shape):
    coo_indexes = get_coo_indexes(series.tolist())
    sparse_df = coo_matrix(
        (np.ones(len(coo_indexes[0])), (coo_indexes[0], coo_indexes[1])), shape=shape
    )
    return sparse_df


def sparse_to_idx(data, pad_idx=-1):
    indexes = data.nonzero()
    indexes_df = pd.DataFrame()
    indexes_df["rows"] = indexes[0]
    indexes_df["cols"] = indexes[1]
    mdf = indexes_df.groupby("rows").apply(lambda x: x["cols"].tolist())
    max_len = mdf.apply(lambda x: len(x)).max()
    return mdf.apply(lambda x: pd.Series(x + [pad_idx] * (max_len - len(x)))).values


def idx_to_sparse(idx, sparse_dim):
    sparse = np.zeros(sparse_dim)
    sparse[int(idx)] = 1
    return pd.Series(sparse, dtype=int)


def process_cats_as_kaggle_notebook(df):
    df["gender"] = (df["gender"] == "M").astype(int)
    df = pd.concat(
        [
            df.drop("occupation", axis=1),
            pd.get_dummies(df["occupation"]).astype(int),
        ],
        axis=1,
    )
    df.drop("other", axis=1, inplace=True)
    df.drop("zip_code", axis=1, inplace=True)

    return df

# --- Data Loading and Preprocessing ---
id_cols = ["user_id", "movie_id"]

try:
    df_train = pd.read_pickle(save_path / "df_train.pkl")
    df_valid = pd.read_pickle(save_path / "df_valid.pkl")
    df_test = pd.read_pickle(save_path / "df_test.pkl")
    df_test = pd.concat([df_valid, df_test], ignore_index=True)
except FileNotFoundError:
    print(f"Error: Data files not found in {save_path}.")
    print("Please ensure 'df_train.pkl', 'df_valid.pkl', 'df_test.pkl' are in this directory.")
    print("You might need to run the data preparation part from the original Kaggle notebook first.")
    exit()

df_train = process_cats_as_kaggle_notebook(df_train)
df_test = process_cats_as_kaggle_notebook(df_test)

max_movie_index = max(df_train.movie_id.max(), df_test.movie_id.max())

X_train = df_train.drop(id_cols + ["prev_movies", "target"], axis=1)
y_train = df_train.target.values
train_movies_watched = get_sparse_features(
    df_train["prev_movies"], (len(df_train), max_movie_index + 1)
)

X_test = df_test.drop(id_cols + ["prev_movies", "target"], axis=1)
y_test = df_test.target.values
test_movies_watched = get_sparse_features(
    df_test["prev_movies"], (len(df_test), max_movie_index + 1)
)

PAD_IDX = 0

# --- Convert data to PyTorch tensors and move to device with specified DTYPE ---
X_train_tensor = torch.Tensor(X_train.fillna(0).values).to(DEVICE, DTYPE)
train_movies_watched_tensor = (
    torch.sparse_coo_tensor(
        indices=train_movies_watched.nonzero(),
        values=[1] * len(train_movies_watched.nonzero()[0]),
        size=train_movies_watched.shape,
    )
    .to_dense()
    .to(DEVICE, DTYPE)
)
movies_train_sequences = (
    torch.Tensor(
        sparse_to_idx(train_movies_watched, pad_idx=PAD_IDX),
    )
    .long()
    .to(DEVICE)
)
target_train = torch.Tensor(y_train).long().to(DEVICE)

X_test_tensor = torch.Tensor(X_test.fillna(0).values).to(DEVICE, DTYPE)
test_movies_watched_tensor = (
    torch.sparse_coo_tensor(
        indices=test_movies_watched.nonzero(),
        values=[1] * len(test_movies_watched.nonzero()[0]),
        size=test_movies_watched.shape,
    )
    .to_dense()
    .to(DEVICE, DTYPE)
)
movies_test_sequences = (
    torch.Tensor(
        sparse_to_idx(test_movies_watched, pad_idx=PAD_IDX),
    )
    .long()
    .to(DEVICE)
)
target_test = torch.Tensor(y_test).long().to(DEVICE)


# --- Model Definition ---
class WideAndDeep(nn.Module):
    def __init__(
        self,
        continious_feature_shape,  # number of continious features
        embed_size,  # size of embedding for binary features
        embed_dict_len,  # number of unique binary features
        pad_idx,  # padding index
    ):
        super(WideAndDeep, self).__init__()
        self.embed = nn.Embedding(embed_dict_len, embed_size, padding_idx=pad_idx)
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(embed_size + continious_feature_shape, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.Linear(embed_dict_len + 256, embed_dict_len),
        )

    def forward(self, continious, binary, binary_idx):
        # get embeddings for sequence of indexes
        binary_embed = self.embed(binary_idx)
        binary_embed_mean = mean(binary_embed, dim=1)
        # get logits for "deep" part: continious features + binary embeddings
        deep_logits = self.linear_relu_stack(
            cat((continious, binary_embed_mean), dim=1)
        )
        # get final softmax logits for "deep" part and raw binary features
        total_logits = self.head(cat((deep_logits, binary), dim=1))
        return total_logits


# Instantiate the model and move it to the specified device and data type
model = WideAndDeep(X_train.shape[1], 16, max_movie_index + 1, PAD_IDX).to(DEVICE, DTYPE)

# --- Apply torch.compile if enabled ---
if args.compile:
    print("Compiling model with torch.compile...")
    model = torch.compile(model, mode="default")
    print("Model compiled.")

print(model)

# --- Training Parameters ---
EPOCHS = args.epochs
loss_fn = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# --- Benchmark Section ---
print("\n--- Starting Benchmarking ---")

print("Warm-up phase (5 epoch)...")
for _ in range(5):
    model.train()
    optimizer.zero_grad()
    _ = model(X_train_tensor, train_movies_watched_tensor, movies_train_sequences)
    # No loss.backward() or optimizer.step() needed for warm-up, just forward pass
    if DEVICE != 'cpu':
        torch.xpu.synchronize() if DEVICE == 'xpu' else torch.cuda.synchronize()
print("Warm-up complete.")

training_times_per_epoch = []
inference_times_train_set = []
inference_times_test_set = []

for t in range(EPOCHS):
    # --- Training Benchmark ---
    model.train()
    if DEVICE != 'cpu':
        torch.xpu.synchronize() if DEVICE == 'xpu' else torch.cuda.synchronize()
    
    epoch_start_time = time.time()

    pred_train = model(
        X_train_tensor, train_movies_watched_tensor, movies_train_sequences
    )
    loss_train = loss_fn(pred_train, target_train)

    optimizer.zero_grad()
    loss_train.backward()
    optimizer.step()

    if DEVICE != 'cpu':
        torch.xpu.synchronize() if DEVICE == 'xpu' else torch.cuda.synchronize()
    
    epoch_end_time = time.time()
    if t > 0:
        training_times_per_epoch.append(epoch_end_time - epoch_start_time)

    # --- Inference Benchmark for Train Set ---
    model.eval()
    with torch.no_grad():
        if DEVICE != 'cpu':
            torch.xpu.synchronize() if DEVICE == 'xpu' else torch.cuda.synchronize()
        
        inference_start_time_train = time.time()
        _ = model(X_train_tensor, train_movies_watched_tensor, movies_train_sequences)
        
        if DEVICE != 'cpu':
            torch.xpu.synchronize() if DEVICE == 'xpu' else torch.cuda.synchronize()
        
        inference_end_time_train = time.time()
        if t > 0:
            inference_times_train_set.append(inference_end_time_train - inference_start_time_train)

    # --- Inference Benchmark for Test Set ---
    model.eval()
    with torch.no_grad():
        if DEVICE != 'cpu':
            torch.xpu.synchronize() if DEVICE == 'xpu' else torch.cuda.synchronize()
        
        inference_start_time_test = time.time()
        pred_test = model(
            X_test_tensor, test_movies_watched_tensor, movies_test_sequences
        )
        loss_test = loss_fn(pred_test, target_test)
        
        if DEVICE != 'cpu':
            torch.xpu.synchronize() if DEVICE == 'xpu' else torch.cuda.synchronize()
        
        inference_end_time_test = time.time()
        if t > 0:
            inference_times_test_set.append(inference_end_time_test - inference_start_time_test)
    
    if t > 0:
        print(f"Epoch {t}")
        print(f"Train loss: {loss_train.item():>7f}")
        print(f"Test loss: {loss_test.item():>7f}")
        print(f"Epoch Training Time: {training_times_per_epoch[-1]:.4f} seconds")
        print(f"Train Set Inference Time: {inference_times_train_set[-1]:.4f} seconds")
        print(f"Test Set Inference Time: {inference_times_test_set[-1]:.4f} seconds")
        print("-" * 30)

# --- Final Benchmark Results ---
print("\n--- Final Benchmark Summary ---")
print(f"Average Training Time per Epoch: {np.mean(training_times_per_epoch):.4f} seconds")
print(f"Average Train Set Inference Time: {np.mean(inference_times_train_set):.4f} seconds")
print(f"Average Test Set Inference Time: {np.mean(inference_times_test_set):.4f} seconds")
print(f"Total Training Time for {EPOCHS} epochs: {sum(training_times_per_epoch):.4f} seconds")

train_samples = X_train_tensor.shape[0]
test_samples = X_test_tensor.shape[0]

print(f"Train Set Throughput (samples/sec): {train_samples / np.mean(inference_times_train_set):.2f}")
print(f"Test Set Throughput (samples/sec): {test_samples / np.mean(inference_times_test_set):.2f}")