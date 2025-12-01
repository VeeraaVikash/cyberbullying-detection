"""
STEP 2: Dataset Cleaner
Cleans text and creates labels
"""

import pandas as pd
import re


def clean_text(text):
    """
    Clean a single text
    
    Args:
        text: Raw text string
        
    Returns:
        Cleaned text string
    """
    if pd.isna(text):
        return ""
    
    text = str(text)
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r'http\S+|www\.\S+', '', text)
    
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text


def create_binary_labels(df):
    """
    Create binary labels from categories
    
    0 = not_cyberbullying
    1 = cyberbullying (all other types)
    
    Args:
        df: DataFrame with 'category' column
        
    Returns:
        DataFrame with added 'label' column
    """
    df = df.copy()
    
    # Create binary label
    df['label'] = df['category'].apply(
        lambda x: 0 if x == 'not_cyberbullying' else 1
    )
    
    return df


def clean_dataset(df):
    """
    Clean the entire dataset
    
    Args:
        df: Raw DataFrame
        
    Returns:
        Cleaned DataFrame with labels
    """
    
    print("\n" + "="*60)
    print("CLEANING DATASET")
    print("="*60)
    
    df = df.copy()
    
    # Rename columns for clarity
    df.rename(columns={
        'tweet_text': 'text',
        'cyberbullying_type': 'category'
    }, inplace=True)
    print("✓ Renamed columns")
    
    # Clean text
    print("Cleaning text...")
    df['text'] = df['text'].apply(clean_text)
    print("✓ Text cleaned")
    
    # Remove empty texts
    original_count = len(df)
    df = df[df['text'].str.len() > 0].copy()
    removed = original_count - len(df)
    print(f"✓ Removed {removed} empty texts")
    
    # Create binary labels
    df = create_binary_labels(df)
    print("✓ Created binary labels (0=not_cyberbullying, 1=cyberbullying)")
    
    # Add metadata
    df['text_length'] = df['text'].str.len()
    df['word_count'] = df['text'].str.split().str.len()
    print("✓ Added text_length and word_count")
    
    # Show statistics
    print("\n" + "="*60)
    print("CLEANING COMPLETE")
    print("="*60)
    print(f"Final samples: {len(df):,}")
    print(f"Avg text length: {df['text_length'].mean():.1f} characters")
    print(f"Avg word count: {df['word_count'].mean():.1f} words")
    
    print("\nLabel distribution:")
    label_0 = (df['label'] == 0).sum()
    label_1 = (df['label'] == 1).sum()
    print(f"  0 (not_cyberbullying): {label_0:,} ({label_0/len(df)*100:.1f}%)")
    print(f"  1 (cyberbullying):     {label_1:,} ({label_1/len(df)*100:.1f}%)")
    
    return df


if __name__ == "__main__":
    from dataset_loader import load_raw_dataset
    
    # Load
    df = load_raw_dataset()
    
    # Clean
    df_clean = clean_dataset(df)
    
    # Show sample
    print("\nSample of cleaned data:")
    print(df_clean[['text', 'category', 'label', 'text_length']].head(3))
    
    print("\n" + "="*60)
    print("✓ Cleaner working correctly!")
    print("="*60)
