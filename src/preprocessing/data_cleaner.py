import os
import pandas as pd

# --- 0. CONFIGURARE CĂI RELATIVE ---
# Detectăm locația scriptului actual (presupunem că e în RN/src/neural_network/ sau similar)
current_dir = os.path.dirname(os.path.abspath(__file__))

# Urcăm nivelurile necesare pentru a ajunge la rădăcina proiectului (RN)
# Dacă scriptul e în src/neural_network, urcăm 2 niveluri.
# Dacă e direct în src, urcăm 1 nivel. Ajustăm aici:
PATH_BASE = os.path.abspath(os.path.join(current_dir, "..", ".."))

def process_data():
    # Definirea directoarelor relativ la rădăcină
    raw_base = os.path.join(PATH_BASE, "data", "raw")
    proc_base = os.path.join(PATH_BASE, "data", "processed")
    
    print(f"🔍 Căutare date în: {raw_base}")
    
    for boala in ["pneumonie", "tuberculoza"]:
        file_path = os.path.join(raw_base, boala, "cases.csv")
        
        if not os.path.exists(file_path): 
            print(f"⚠️ Fișierul nu a fost găsit la: {file_path}")
            continue
        
        # Încărcare date
        df = pd.read_csv(file_path)
        features = [f"Q{i}" for i in range(1, 21)]
        
        # Normalizare 1-5 -> 0-1
        # Formula: (x - min) / (max - min) => (x - 1) / 4
        df[features] = (df[features] - 1) / 4.0
        
        # Creare folder destinație
        save_path = os.path.join(proc_base, boala)
        os.makedirs(save_path, exist_ok=True)
        
        # Salvare
        final_save_file = os.path.join(save_path, "processed.csv")
        df.to_csv(final_save_file, index=False)
        print(f"✅ Procesat și salvat: {boala} -> {final_save_file}")
    
    print("\n🚀 Etapa 2 Finalizată: Datele sunt normalizate (0.0 - 1.0) și pregătite pentru antrenament.")

if __name__ == "__main__":
    process_data()
