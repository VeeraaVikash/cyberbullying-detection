"""
Dataset Augmentation Script
Combines your data with Kaggle datasets to fix edge cases
"""

import pandas as pd
import os
from pathlib import Path


def load_your_data():
    """Load your existing training data"""
    print("Loading your existing data...")
    train_df = pd.read_csv('data/processed/train.csv')
    print(f"✓ Loaded {len(train_df):,} samples")
    print(f"  - Cyberbullying: {(train_df['label']==1).sum():,} ({(train_df['label']==1).sum()/len(train_df)*100:.1f}%)")
    print(f"  - Not Cyberbullying: {(train_df['label']==0).sum():,} ({(train_df['label']==0).sum()/len(train_df)*100:.1f}%)")
    return train_df


def load_sentiment140(file_path, n_samples=5000):
    """
    Load Twitter Sentiment140 dataset
    Download from: https://www.kaggle.com/datasets/kazanova/sentiment140
    """
    print(f"\nLoading Sentiment140 from {file_path}...")
    
    if not os.path.exists(file_path):
        print(f"  ⚠️  File not found: {file_path}")
        print("  Download from: https://www.kaggle.com/datasets/kazanova/sentiment140")
        return pd.DataFrame()
    
    # Sentiment140 format: target, ids, date, flag, user, text
    # target: 0 = negative, 4 = positive
    df = pd.read_csv(
        file_path, 
        encoding='latin-1',
        names=['target', 'ids', 'date', 'flag', 'user', 'text']
    )
    
    # Get only positive tweets (target = 4)
    positive = df[df['target'] == 4].copy()
    
    # Sample n_samples
    if len(positive) > n_samples:
        positive = positive.sample(n_samples, random_state=42)
    
    # Format to match your data
    positive['label'] = 0  # Not cyberbullying
    positive['category'] = 'not_cyberbullying'
    positive['text_length'] = positive['text'].str.len()
    positive['word_count'] = positive['text'].str.split().str.len()
    
    result = positive[['text', 'category', 'label', 'text_length', 'word_count']]
    
    print(f"✓ Loaded {len(result):,} positive tweets")
    return result


def load_hate_speech(file_path):
    """
    Load Hate Speech dataset
    Download from: https://www.kaggle.com/datasets/mrmorj/hate-speech-and-offensive-language-dataset
    """
    print(f"\nLoading Hate Speech dataset from {file_path}...")
    
    if not os.path.exists(file_path):
        print(f"  ⚠️  File not found: {file_path}")
        print("  Download from: https://www.kaggle.com/datasets/mrmorj/hate-speech-and-offensive-language-dataset")
        return pd.DataFrame()
    
    df = pd.read_csv(file_path)
    
    # class: 0 = hate speech, 1 = offensive, 2 = neither
    # Use: 0,1 as cyberbullying, 2 as not cyberbullying
    
    hate_and_offensive = df[df['class'].isin([0, 1])].copy()
    neither = df[df['class'] == 2].copy()
    
    # Format hate/offensive as cyberbullying
    hate_and_offensive['label'] = 1
    hate_and_offensive['category'] = 'cyberbullying'
    hate_and_offensive['text'] = hate_and_offensive['tweet']
    hate_and_offensive['text_length'] = hate_and_offensive['text'].str.len()
    hate_and_offensive['word_count'] = hate_and_offensive['text'].str.split().str.len()
    
    # Format neither as not cyberbullying
    neither['label'] = 0
    neither['category'] = 'not_cyberbullying'
    neither['text'] = neither['tweet']
    neither['text_length'] = neither['text'].str.len()
    neither['word_count'] = neither['text'].str.split().str.len()
    
    hate_result = hate_and_offensive[['text', 'category', 'label', 'text_length', 'word_count']]
    neither_result = neither[['text', 'category', 'label', 'text_length', 'word_count']]
    
    print(f"✓ Loaded {len(hate_result):,} cyberbullying examples")
    print(f"✓ Loaded {len(neither_result):,} not cyberbullying examples")
    
    return pd.concat([hate_result, neither_result], ignore_index=True)


def create_manual_edge_cases():
    """
    Create manual examples to fix specific edge cases
    """
    print("\nCreating manual edge case examples...")
    
    edge_cases = []
    
    # 1. Double negatives
    double_negatives = [
        "he is not a bad person",
        "she is not ugly",
        "you are not stupid",
        "that's not terrible",
        "not bad at all",
        "he's not mean",
        "she's not a loser",
        "you're not worthless",
        "it's not awful",
        "not a horrible idea",
        "he is not cruel",
        "she is not dumb",
        "you are not useless",
        "that's not pathetic",
        "not terrible at all",
    ]
    
    for text in double_negatives:
        edge_cases.append({
            'text': text,
            'category': 'not_cyberbullying',
            'label': 0,
            'text_length': len(text),
            'word_count': len(text.split())
        })
    
    # 2. Sports slang (GOAT, beast, legend)
    sports_slang = [
        "Virat is GOAT",
        "Kohli is the greatest of all time",
        "Dhoni is a legend",
        "Rohit is amazing",
        "Messi is GOAT",
        "Ronaldo is the best",
        "LeBron is a beast",
        "Curry is fire",
        "You're a beast at this",
        "That was sick",
        "He's an absolute legend",
        "She killed it today",
        "What a savage performance",
        "King of cricket",
        "Queen of tennis",
        "Absolute madlad",
        "He's cracked at this game",
        "She's insane at coding",
        "You're built different",
        "That's fire bro",
    ]
    
    for text in sports_slang:
        edge_cases.append({
            'text': text,
            'category': 'not_cyberbullying',
            'label': 0,
            'text_length': len(text),
            'word_count': len(text.split())
        })
    
    # 3. Sports positive context
    sports_positive = [
        "Virat Kohli played brilliantly",
        "Dhoni's captaincy is excellent",
        "Rohit Sharma hit a century",
        "Love watching Virat play",
        "Dhoni is my inspiration",
        "Respect to Rohit",
        "Best player in the world",
        "Greatest cricketer ever",
        "What a match by Virat",
        "Dhoni finished it in style",
        "Rohit's batting is pure class",
        "Virat is a role model",
        "Dhoni's calmness is inspiring",
        "Rohit is so consistent",
        "Virat's passion is unmatched",
    ]
    
    for text in sports_positive:
        edge_cases.append({
            'text': text,
            'category': 'not_cyberbullying',
            'label': 0,
            'text_length': len(text),
            'word_count': len(text.split())
        })
    
    # 4. General positive with "negative" words
    positive_with_neg = [
        "You killed that presentation",
        "This is sick man",
        "You're a beast",
        "That's fire",
        "You're insane at this",
        "He's a savage",
        "She's crazy talented",
        "That's nuts dude",
        "You're mental good",
        "This is wicked cool",
    ]
    
    for text in positive_with_neg:
        edge_cases.append({
            'text': text,
            'category': 'not_cyberbullying',
            'label': 0,
            'text_length': len(text),
            'word_count': len(text.split())
        })
    
    df = pd.DataFrame(edge_cases)
    print(f"✓ Created {len(df):,} manual edge case examples")
    print(f"  - Double negatives: {len(double_negatives)}")
    print(f"  - Sports slang: {len(sports_slang)}")
    print(f"  - Sports positive: {len(sports_positive)}")
    print(f"  - Positive with neg words: {len(positive_with_neg)}")
    
    return df


def combine_datasets(original, sentiment140, hate_speech, manual_cases):
    """Combine all datasets"""
    print("\n" + "="*70)
    print("COMBINING DATASETS")
    print("="*70)
    
    datasets = []
    
    # Original data
    if len(original) > 0:
        datasets.append(('Original', original))
    
    # Sentiment140
    if len(sentiment140) > 0:
        datasets.append(('Sentiment140', sentiment140))
    
    # Hate Speech
    if len(hate_speech) > 0:
        datasets.append(('Hate Speech', hate_speech))
    
    # Manual cases
    if len(manual_cases) > 0:
        datasets.append(('Manual Edge Cases', manual_cases))
    
    # Combine
    combined = pd.concat([df for _, df in datasets], ignore_index=True)
    
    # Show statistics
    print(f"\nDataset Composition:")
    for name, df in datasets:
        cb_count = (df['label']==1).sum()
        not_cb_count = (df['label']==0).sum()
        print(f"  {name:20} {len(df):6,} samples ({cb_count:5,} CB / {not_cb_count:5,} Not CB)")
    
    print(f"\n{'='*70}")
    print(f"TOTAL: {len(combined):,} samples")
    print(f"  Cyberbullying: {(combined['label']==1).sum():,} ({(combined['label']==1).sum()/len(combined)*100:.1f}%)")
    print(f"  Not Cyberbullying: {(combined['label']==0).sum():,} ({(combined['label']==0).sum()/len(combined)*100:.1f}%)")
    print(f"{'='*70}")
    
    return combined


def main():
    """Main augmentation pipeline"""
    
    print("="*70)
    print(" "*20 + "DATASET AUGMENTATION")
    print("="*70)
    
    # Create directories
    os.makedirs('data/external', exist_ok=True)
    
    # Load your data
    original = load_your_data()
    
    # Load external datasets
    # NOTE: Update these paths after downloading from Kaggle
    sentiment140 = load_sentiment140(
        'data/external/sentiment140.csv',
        n_samples=5000
    )
    
    hate_speech = load_hate_speech(
        'data/external/hate_speech.csv'
    )
    
    # Create manual edge cases
    manual_cases = create_manual_edge_cases()
    
    # Combine all
    combined = combine_datasets(original, sentiment140, hate_speech, manual_cases)
    
    # Shuffle
    combined = combined.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Split back into train/val/test (70/15/15)
    train_size = int(0.70 * len(combined))
    val_size = int(0.15 * len(combined))
    
    train = combined[:train_size]
    val = combined[train_size:train_size+val_size]
    test = combined[train_size+val_size:]
    
    # Save
    output_dir = 'data/processed_augmented'
    os.makedirs(output_dir, exist_ok=True)
    
    train.to_csv(f'{output_dir}/train.csv', index=False)
    val.to_csv(f'{output_dir}/val.csv', index=False)
    test.to_csv(f'{output_dir}/test.csv', index=False)
    
    print(f"\n✓ Saved augmented datasets to {output_dir}/")
    print(f"  - train.csv: {len(train):,} samples")
    print(f"  - val.csv: {len(val):,} samples")
    print(f"  - test.csv: {len(test):,} samples")
    
    print("\n" + "="*70)
    print("NEXT STEPS:")
    print("="*70)
    print("1. Update models/config.py:")
    print("   TRAIN_DATA = 'data/processed_augmented/train.csv'")
    print("   VAL_DATA = 'data/processed_augmented/val.csv'")
    print("   TEST_DATA = 'data/processed_augmented/test.csv'")
    print("\n2. Retrain model:")
    print("   py models/train.py")
    print("\n3. Evaluate:")
    print("   py models/evaluate.py")
    print("="*70)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
