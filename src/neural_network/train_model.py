import pandas as pd
import numpy as np
import os
import math
import random
import pickle
import time

class NeuralNetworkAbsoluteZero:
    def __init__(self, input_size=20, hidden_size=6, output_size=1, learning_rate=0.01):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.lr = learning_rate
        limit = math.sqrt(6 / (input_size + hidden_size))
        self.W1 = [[random.uniform(-limit, limit) for _ in range(hidden_size)] for _ in range(input_size)]
        self.b1 = [0.0] * hidden_size
        self.W2 = [[random.uniform(-limit, limit) for _ in range(output_size)] for _ in range(hidden_size)]
        self.b2 = [0.0] * output_size

    def sigmoid(self, x):
        return 1 / (1 + math.exp(-max(min(x, 50), -50)))

    def forward(self, x):
        h = [self.sigmoid(sum(x[i] * self.W1[i][j] for i in range(self.input_size)) + self.b1[j]) for j in range(self.hidden_size)]
        o = self.sigmoid(sum(h[j] * self.W2[j][0] for j in range(self.hidden_size)) + self.b2[0])
        return h, o

    def train_step(self, x_sample, y_target):
        h_act, o_act = self.forward(x_sample)
        out_error = y_target[0] - o_act
        d_out = out_error * o_act * (1 - o_act)
        for j in range(self.hidden_size):
            error_h = d_out * self.W2[j][0]
            d_h = error_h * h_act[j] * (1 - h_act[j])
            self.W2[j][0] += self.lr * d_out * h_act[j]
            for i in range(self.input_size):
                self.W1[i][j] += self.lr * d_h * x_sample[i]
            self.b1[j] += self.lr * d_h
        self.b2[0] += self.lr * d_out
        return out_error**2

def load_training_data(path_train):
    X, Y = [], []
    categories = {'Pneumonia': 0, 'Tuberculoza': 1}
    print(f"📂 Caut date în: {path_train}...", flush=True)
    for cat_name, label in categories.items():
        folder_path = os.path.join(path_train, cat_name)
        if os.path.exists(folder_path):
            files = [f for f in os.listdir(folder_path) if f.endswith(".csv")]
            print(f"  - Găsit {len(files)} fișiere în {cat_name}", flush=True)
            for file in files:
                df = pd.read_csv(os.path.join(folder_path, file))
                X.extend(df.iloc[:, :20].values.tolist())
                Y.extend([[label]] * len(df))
    return X, Y

def main():
    path_train = r"D:\Facultate\RN\docs\train"
    model_export_path = r"D:\Facultate\RN\models\trained_model.pkl"

    X_train, Y_train = load_training_data(path_train)

    if not X_train:
        print("❌ EROARE: Nu am găsit date! Verifică folderele D:\Facultate\RN\docs\train\Pneumonia și Tuberculoza", flush=True)
        return

    print(f"✅ Date încărcate cu succes: {len(X_train)} rânduri.", flush=True)
    
    combined = list(zip(X_train, Y_train))
    random.shuffle(combined)
    X_train, Y_train = zip(*combined)

    nn = NeuralNetworkAbsoluteZero(20, 6, 1, learning_rate=0.01)

    print(f"🚀 Încep cele 30 de epoci...", flush=True)
    print("-" * 40, flush=True)

    for epoch in range(1, 31):
        start_time = time.time()
        total_mse = 0
        for i in range(len(X_train)):
            total_mse += nn.train_step(X_train[i], Y_train[i])
        
        durata = time.time() - start_time
        print(f"Epoca {epoch:02d}/30 | MSE: {total_mse/len(X_train):.6f} | Timp: {durata:.2f}s", flush=True)

    os.makedirs(os.path.dirname(model_export_path), exist_ok=True)
    with open(model_export_path, 'wb') as f:
        pickle.dump(nn, f)
    print("-" * 40, flush=True)
    print(f"✅ Model salvat: {model_export_path}", flush=True)

if __name__ == "__main__":
    main()