"""
STEP 3: Dataset Splitter
Splits data into train/validation/test sets
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path


def split_dataset(df, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, random_seed=42):
    """
    Split dataset into train/val/test
    
    Args:
        df: Cleaned DataFrame
        train_ratio: Training set size (default 0.7 = 70%)
        val_ratio: Validation set size (default 0.15 = 15%)
        test_ratio: Test set size (default 0.15 = 15%)
        random_seed: Random seed for reproducibility
        
    Returns:
        train_df, val_df, test_df
    """
    
    print("\n" + "="*60)
    print("SPLITTING DATASET")
    print("="*60)
    
    # Verify ratios sum to 1.0
    assert abs((train_ratio + val_ratio + test_ratio) - 1.0) < 0.001, \
        "Ratios must sum to 1.0"
    
    # First split: train vs (val + test)
    train_df, temp_df = train_test_split(
        df,
        test_size=(val_ratio + test_ratio),
        stratify=df['label'],  # Keep same label distribution
        random_state=random_seed
    )
    print(f"✓ Split: train ({train_ratio*100:.0f}%) vs temp ({(val_ratio+test_ratio)*100:.0f}%)")
    
    # Second split: val vs test
    val_size_ratio = val_ratio / (val_ratio + test_ratio)
    val_df, test_df = train_test_split(
        temp_df,
        test_size=(1 - val_size_ratio),
        stratify=temp_df['label'],
        random_state=random_seed
    )
    print(f"✓ Split: val ({val_ratio*100:.0f}%) vs test ({test_ratio*100:.0f}%)")
    
    # Print statistics
    print("\n" + "="*60)
    print("SPLIT COMPLETE")
    print("="*60)
    
    total = len(df)
    print(f"Total samples: {total:,}\n")
    print(f"Train: {len(train_df):,} ({len(train_df)/total*100:.1f}%)")
    print(f"Val:   {len(val_df):,} ({len(val_df)/total*100:.1f}%)")
    print(f"Test:  {len(test_df):,} ({len(test_df)/total*100:.1f}%)")
    
    # Verify label distribution
    print("\nLabel distribution per split:")
    for name, split_df in [('Train', train_df), ('Val', val_df), ('Test', test_df)]:
        label_0 = (split_df['label'] == 0).sum()
        label_1 = (split_df['label'] == 1).sum()
        print(f"  {name}:")
        print(f"    Label 0: {label_0:,} ({label_0/len(split_df)*100:.1f}%)")
        print(f"    Label 1: {label_1:,} ({label_1/len(split_df)*100:.1f}%)")
    
    return train_df, val_df, test_df


def save_splits(train_df, val_df, test_df, output_dir='data/processed'):
    """
    Save train/val/test splits to CSV files
    
    Args:
        train_df: Training DataFrame
        val_df: Validation DataFrame
        test_df: Test DataFrame
        output_dir: Directory to save files
    """
    
    print("\n" + "="*60)
    print("SAVING SPLITS")
    print("="*60)
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save each split
    splits = {
        'train': train_df,
        'val': val_df,
        'test': test_df
    }
    
    for name, df in splits.items():
        filepath = output_path / f'{name}.csv'
        df.to_csv(filepath, index=False)
        print(f"✓ Saved {filepath}")
        print(f"  Samples: {len(df):,}")
        print(f"  Columns: {list(df.columns)}")
    
    print("\n" + "="*60)
    print(f"✓ All files saved to: {output_path}/")
    print("="*60)


if __name__ == "__main__":
    from dataset_loader import load_raw_dataset
    from dataset_cleaner import clean_dataset
    
    # Load
    print("STEP 1: Loading...")
    df = load_raw_dataset()
    
    # Clean
    print("\nSTEP 2: Cleaning...")
    df_clean = clean_dataset(df)
    
    # Split
    print("\nSTEP 3: Splitting...")
    train_df, val_df, test_df = split_dataset(df_clean)
    
    # Save
    print("\nSTEP 4: Saving...")
    save_splits(train_df, val_df, test_df)
    
    print("\n" + "="*60)
    print("✓ DATA PREPARATION COMPLETE!")
    print("="*60)
    print("\nNext: Train a model on this data")
    print("Files ready in: data/processed/")
