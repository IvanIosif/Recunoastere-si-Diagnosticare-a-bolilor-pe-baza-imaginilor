import math
import random
import pickle
import os

# Definiția clasei exact ca în codul tău de antrenare
class NeuralNetworkAbsoluteZero:
    def __init__(self, input_size=20, hidden_size=6, output_size=1, learning_rate=0.01):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.lr = learning_rate
        
        # Inițializare Xavier (Ponderi aleatorii pentru modelul neantrenat)
        limit = math.sqrt(6 / (input_size + hidden_size))
        self.W1 = [[random.uniform(-limit, limit) for _ in range(hidden_size)] for _ in range(input_size)]
        self.b1 = [0.0] * hidden_size
        self.W2 = [[random.uniform(-limit, limit) for _ in range(output_size)] for _ in range(hidden_size)]
        self.b2 = [0.0] * output_size

    def sigmoid(self, x):
        return 1 / (1 + math.exp(-max(min(x, 50), -50)))

    def forward(self, x):
        h = [self.sigmoid(sum(x[i] * self.W1[i][j] for i in range(self.input_size)) + self.b1[j]) 
             for j in range(self.hidden_size)]
        o = self.sigmoid(sum(h[j] * self.W2[j][0] for j in range(self.hidden_size)) + self.b2[0])
        return h, o

def save_skeleton():
    # Calea solicitată de tine
    folder_path = r"D:\Facultate\RN\models"
    file_name = "untrained_model.pkl"
    full_path = os.path.join(folder_path, file_name)

    # Asigurăm existența folderului
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    # Inițializăm rețeaua (fără antrenare)
    nn_untrained = NeuralNetworkAbsoluteZero(input_size=20, hidden_size=6, output_size=1)

    # Salvare în format pickle
    with open(full_path, 'wb') as f:
        pickle.dump(nn_untrained, f)

    print(f"✅ Modelul NEANTRENAT a fost salvat în: {full_path}")
    print("Acesta conține arhitectura (W1, W2) dar ponderile sunt aleatorii.")

if __name__ == "__main__":
    save_skeleton()