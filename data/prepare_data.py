"""
MAIN SCRIPT: Prepare Data
Run complete data preparation pipeline
"""

from dataset_loader import load_raw_dataset
from dataset_cleaner import clean_dataset
from dataset_splitter import split_dataset, save_splits


def main():
    """
    Complete data preparation pipeline
    
    Steps:
    1. Load raw data from data/raw/
    2. Clean text and create labels
    3. Split into train/val/test (70/15/15)
    4. Save to data/processed/
    """
    
    print("\n" + "="*70)
    print(" " * 20 + "DATA PREPARATION PIPELINE")
    print("="*70)
    
    try:
        # Step 1: Load
        print("\n📂 STEP 1/4: Loading raw data...")
        df = load_raw_dataset()
        
        # Step 2: Clean
        print("\n🧹 STEP 2/4: Cleaning data...")
        df_clean = clean_dataset(df)
        
        # Step 3: Split
        print("\n✂️  STEP 3/4: Splitting data...")
        train_df, val_df, test_df = split_dataset(df_clean)
        
        # Step 4: Save
        print("\n💾 STEP 4/4: Saving data...")
        save_splits(train_df, val_df, test_df)
        
        # Success summary
        print("\n" + "="*70)
        print(" " * 25 + "✅ SUCCESS!")
        print("="*70)
        
        print("\n📊 Summary:")
        print(f"  Original samples: 47,692")
        print(f"  After cleaning: {len(df_clean):,}")
        print(f"  Train samples: {len(train_df):,} (70%)")
        print(f"  Val samples: {len(val_df):,} (15%)")
        print(f"  Test samples: {len(test_df):,} (15%)")
        
        print("\n📁 Output files:")
        print("  ✓ data/processed/train.csv")
        print("  ✓ data/processed/val.csv")
        print("  ✓ data/processed/test.csv")
        
        print("\n✨ Data is ready for model training!")
        print("\n📝 Next step: Create and train model")
        
    except Exception as e:
        print("\n" + "="*70)
        print(" " * 28 + "❌ ERROR")
        print("="*70)
        print(f"\nError occurred: {str(e)}")
        print("\nPlease check:")
        print("  1. data/raw/cyberbullying_tweets.csv exists")
        print("  2. You have write permissions")
        print("  3. Required packages are installed (pandas, scikit-learn)")
        raise


if __name__ == "__main__":
    main()
