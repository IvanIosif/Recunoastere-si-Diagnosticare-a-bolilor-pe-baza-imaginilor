import os
import random
import shutil

# Căi
src_pneumonia = r"D:\Facultate\RN\docs\processed\Pneumonia"
src_tuberculoza = r"D:\Facultate\RN\docs\processed\Tuberculoza"
dst_base = r"D:\Facultate\RN\docs"

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15

def organizeaza_dataset(src_path, class_name):
    # --- PAS DE SIGURANȚĂ ---
    # Ștergem folderele vechi dacă vrem o distribuție proaspătă
    for s in ["train", "validation", "test"]:
        path_to_clean = os.path.join(dst_base, s, class_name)
        if os.path.exists(path_to_clean):
            shutil.rmtree(path_to_clean) # Șterge folderul cu tot cu poze vechi
            print(f"Curățat folder existent: {s}\\{class_name}")
    # ------------------------

    files = [f for f in os.listdir(src_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    random.shuffle(files)

    total = len(files)
    train_count = int(total * TRAIN_RATIO)
    val_count = int(total * VAL_RATIO)

    split_data = {
        "train": files[:train_count],
        "validation": files[train_count : train_count + val_count],
        "test": files[train_count + val_count:]
    }

    for split_name, filenames in split_data.items():
        target_dir = os.path.join(dst_base, split_name, class_name)
        os.makedirs(target_dir, exist_ok=True) # Creează la loc folderul gol

        for f in filenames:
            shutil.copy2(os.path.join(src_path, f), os.path.join(target_dir, f))
            
    print(f"Distribuire finalizată pentru {class_name}!")

if __name__ == "__main__":
    organizeaza_dataset(src_pneumonia, "Pneumonia")
    organizeaza_dataset(src_tuberculoza, "Tuberculoza")