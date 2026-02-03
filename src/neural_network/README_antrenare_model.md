### 📂 README pentru `src/neural_network/` (Rețeaua)
**Fișier:** `src/neural_network/README.md`

```markdown
# 🧠 Modulul 2: Rețea Neuronală (Antrenare & Optimizare)

Acest director conține „inteligența” sistemului de diagnostic. Modulul este responsabil pentru antrenarea, optimizarea și validarea modelului Multi-Layer Perceptron (MLP) specializat în detecția TBC și a Pneumoniei.

## Arhitectura Modelului 
În urma experimentelor din Etapa 6, arhitectura optimă (Exp3_Balanced) a fost definită astfel:
* **Input Layer: 20 de neuroni (corespunzători celor 20 de simptome din chestionar).
* **Hidden Layers:**
    * Dense (64 neuroni, activare `ReLU`)
    * Dense (64 neuroni, activare `ReLU`)
    * Dense (64 neuroni, activare `ReLU`)
Batch Normalization: Pentru stabilizarea gradientului și accelerarea convergenței.
Dropout (0.2): Pentru prevenirea overfitting-ului (dezactivează aleatoriu 20% din conexiuni).

Output Layer: 1 neuron cu activare Sigmoid (generează o probabilitate între 0 și 1).

< 0.5 -> Pneumonie (Clasa 0)

> 0.5 -> TBC (Clasa 1)

## 🛠️ Scripturi
1.  **`train.py**:
    * Încarcă datele din `data/`.
    * Antrenează modelul folosind optimizatorul **Adam**.
    * Salvează modelul antrenat în `models/trained_model.keras`.
    * Salvează metricile și scaler-ul.
2. optimize.py
Identifică cea mai bună arhitectură prin testarea a 4 configurații diferite de rețele profunde
## 📈 Performanță
* **Acuratețe Finală:** ~82.73%
* **TBC Recall (Sensibilitate): ~86%
* **Latență:** ~35 ms / inferență

## ⚙️ Execuție Antrenament
```bash
python src/neural_network/optimize.py