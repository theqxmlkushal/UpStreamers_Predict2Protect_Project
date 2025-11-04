import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, Concatenate
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score, confusion_matrix

# ========== PARAMETERS ==========
WINDOW = 3              # number of years per sequence
EPOCHS = 100            # max training epochs
BALANCE = True          # balance alive/failed
LEARNING_RATE = 0.0001

# ========== LOAD DATA ==========
df = pd.read_csv("/mnt/data/train.csv")

# Clean and preprocess
df = df.drop(columns=["Unnamed: 0", "Division", "MajorGroup"])
df = df.sort_values(["company_name", "fyear"])

# Encode target
df["status"] = df["status_label"].map({"alive": 0, "failed": 1})

# ========== BUILD WINDOWED SEQUENCES ==========
def make_sequences(df, window):
    Xs, ys = [], []
    grouped = df.groupby("company_name")
    for _, group in grouped:
        if len(group) >= window:
            group = group.sort_values("fyear")
            for i in range(len(group) - window + 1):
                seq = group.iloc[i:i+window]
                Xs.append(seq[[f"X{j}" for j in range(1,19)]].values)
                ys.append(seq["status"].values[-1])
    Xs = np.array(Xs)
    ys = np.array(ys)
    return Xs, ys

X, y = make_sequences(df, WINDOW)
print("Sequences:", X.shape, "Labels:", y.shape)

# Balance dataset if needed
if BALANCE:
    alive_idx = np.where(y == 0)[0]
    fail_idx = np.where(y == 1)[0]
    n = min(len(alive_idx), len(fail_idx))
    selected_idx = np.concatenate([np.random.choice(alive_idx, n, replace=False),
                                   np.random.choice(fail_idx, n, replace=False)])
    np.random.shuffle(selected_idx)
    X, y = X[selected_idx], y[selected_idx]

# Split train/validation
split = int(0.8 * len(X))
X_train, X_val = X[:split], X[split:]
y_train, y_val = y[:split], y[split:]

# Normalize each feature
mean = X_train.mean(axis=(0,1))
std = X_train.std(axis=(0,1))
X_train = (X_train - mean) / std
X_val = (X_val - mean) / std

# One-hot encode labels
y_train = to_categorical(y_train)
y_val = to_categorical(y_val)

# ========== BUILD MULTI-HEAD LSTM ==========
variables = [f"X{i}" for i in range(1, 19)]
lstm_heads = []
inputs = []

for var_idx in range(len(variables)):
    input_i = tf.keras.Input(shape=(WINDOW, 1))
    lstm_i = LSTM(WINDOW)(input_i)
    inputs.append(input_i)
    lstm_heads.append(lstm_i)

merged = Concatenate()(lstm_heads)
dense1 = Dense(20, activation='relu')(merged)
output = Dense(2, activation='softmax')(dense1)

model = Model(inputs=inputs, outputs=output)
model.compile(optimizer=tf.keras.optimizers.Adam(LEARNING_RATE),
              loss='binary_crossentropy',
              metrics=['accuracy', tf.keras.metrics.AUC()])

model.summary()

# ========== PREPARE MULTI-HEAD INPUTS ==========
X_train_split = [X_train[:,:,i].reshape(-1, WINDOW, 1) for i in range(18)]
X_val_split   = [X_val[:,:,i].reshape(-1, WINDOW, 1) for i in range(18)]

# ========== TRAIN ==========
es = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1)
history = model.fit(X_train_split, y_train,
                    validation_data=(X_val_split, y_val),
                    epochs=EPOCHS, batch_size=32, verbose=1, callbacks=[es])

# ========== EVALUATE ==========
y_pred = model.predict(X_val_split).argmax(axis=1)
y_true = y_val.argmax(axis=1)

acc = accuracy_score(y_true, y_pred)
cm = confusion_matrix(y_true, y_pred)
print("\nConfusion Matrix:\n", cm)
print("Accuracy:", acc)
