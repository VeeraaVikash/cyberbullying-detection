"""
STEP 1: Dataset Loader
Loads the raw CSV file and shows basic info
"""

import pandas as pd
from pathlib import Path


def load_raw_dataset():
    """
    Load the raw cyberbullying dataset
    
    Returns:
        DataFrame with columns: tweet_text, cyberbullying_type
    """
    
    # Get file path
    current_dir = Path(__file__).parent
    raw_file = current_dir / 'raw' / 'cyberbullying_tweets.csv'
    
    print("="*60)
    print("LOADING RAW DATASET")
    print("="*60)
    print(f"File: {raw_file}")
    
    # Load CSV
    df = pd.read_csv(raw_file)
    
    print(f"✓ Loaded successfully!")
    print(f"Total rows: {len(df):,}")
    print(f"Columns: {list(df.columns)}")
    
    # Show sample
    print("\nFirst 3 rows:")
    print(df.head(3))
    
    # Show class distribution
    print("\nClass distribution:")
    print(df['cyberbullying_type'].value_counts())
    
    return df


if __name__ == "__main__":
    # Test the loader
    df = load_raw_dataset()
    print("\n" + "="*60)
    print("✓ Loader working correctly!")
    print("="*60)
