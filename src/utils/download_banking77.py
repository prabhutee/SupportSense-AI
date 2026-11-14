"""
Utility script to download, verify, and inspect the BANKING77 dataset.
Preserves the original 77 intent labels and standard train/test split.
"""

import os
import json
import urllib.request
import pandas as pd

# Data sources (PolyAI official repository and Hugging Face mirror fallback)
URLS = {
    "train": [
        "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/master/banking_data/train.csv",
        "https://huggingface.co/datasets/PolyAI/banking77/raw/main/train.csv"
    ],
    "test": [
        "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/master/banking_data/test.csv",
        "https://huggingface.co/datasets/PolyAI/banking77/raw/main/test.csv"
    ],
    "categories": [
        "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/master/banking_data/categories.json",
        "https://huggingface.co/datasets/PolyAI/banking77/raw/main/categories.json"
    ]
}

def download_file(url_list, target_path):
    """Download a file with fallback URLs."""
    os.makedirs(os.path.dirname(target_path), exist_ok=True)
    for url in url_list:
        try:
            print(f"Attempting download from: {url}")
            urllib.request.urlretrieve(url, target_path)
            print(f"Successfully saved to: {target_path}")
            return True
        except Exception as e:
            print(f"Failed to download from {url}: {e}")
    return False

def inspect_dataset(data_dir="data"):
    """Inspect and validate the downloaded BANKING77 dataset."""
    train_path = os.path.join(data_dir, "raw", "banking77_train.csv")
    test_path = os.path.join(data_dir, "raw", "banking77_test.csv")
    categories_path = os.path.join(data_dir, "raw", "categories.json")

    print("\n" + "=" * 60)
    print("BANKING77 DATASET INSPECTION & VALIDATION")
    print("=" * 60)

    # 1. Load Categories
    with open(categories_path, "r", encoding="utf-8") as f:
        categories = json.load(f)
    print(f"[OK] Total Canonical Categories Defined: {len(categories)}")

    # 2. Load Train Set
    df_train = pd.read_csv(train_path)
    print(f"[OK] Train set loaded: {df_train.shape[0]} rows, {df_train.shape[1]} columns")
    print(f"     Columns: {list(df_train.columns)}")
    train_classes = df_train['category'].nunique()
    print(f"     Unique Intents in Train: {train_classes}")

    # 3. Load Test Set
    df_test = pd.read_csv(test_path)
    print(f"[OK] Test set loaded: {df_test.shape[0]} rows, {df_test.shape[1]} columns")
    print(f"     Columns: {list(df_test.columns)}")
    test_classes = df_test['category'].nunique()
    print(f"     Unique Intents in Test: {test_classes}")

    # Combined dataset inspection
    df_all = pd.concat([df_train, df_test], ignore_index=True)
    total_samples = len(df_all)
    print(f"\n[OK] Combined Dataset Total Samples: {total_samples}")
    print(f"     Total Unique Labels: {df_all['category'].nunique()} (Target: 77)")
    
    # Class balance stats
    counts = df_all['category'].value_counts()
    print(f"     Min samples per class: {counts.min()} ('{counts.idxmin()}')")
    print(f"     Max samples per class: {counts.max()} ('{counts.idxmax()}')")
    print(f"     Mean samples per class: {counts.mean():.1f}")
    
    # Check for missing values
    null_count = df_all.isnull().sum().sum()
    print(f"     Missing values across all fields: {null_count}")

    # Display 5 sample records
    print("\nSample records:")
    for idx, row in df_all.sample(5, random_state=42).iterrows():
        print(f"  - [{row['category']}]: \"{row['text']}\"")

    # Save a combined processed dataset for unified reference
    processed_combined_path = os.path.join(data_dir, "processed", "banking77_full.csv")
    df_all.to_csv(processed_combined_path, index=False)
    print(f"\n[OK] Exported unified dataset to: {processed_combined_path}")
    print("=" * 60)

def main():
    raw_dir = os.path.join("data", "raw")
    train_file = os.path.join(raw_dir, "banking77_train.csv")
    test_file = os.path.join(raw_dir, "banking77_test.csv")
    cat_file = os.path.join(raw_dir, "categories.json")

    success_train = download_file(URLS["train"], train_file)
    success_test = download_file(URLS["test"], test_file)
    success_cat = download_file(URLS["categories"], cat_file)

    if success_train and success_test and success_cat:
        inspect_dataset()
    else:
        print("[ERROR] Failed to download all required files.")

if __name__ == "__main__":
    main()
