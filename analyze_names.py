"""
Analyze Training Data - Check for celebrity/name bias
"""

import pandas as pd
from collections import Counter
import re


def analyze_name_patterns(csv_path):
    """Analyze if certain names appear more in cyberbullying tweets"""
    
    print("="*70)
    print("TRAINING DATA ANALYSIS - Name Pattern Detection")
    print("="*70)
    
    # Load data
    df = pd.read_csv(csv_path)
    
    # Sports figures to check
    sports_names = [
        'virat', 'kohli', 'dhoni', 'rohit', 'bumrah', 'sachin',
        'messi', 'ronaldo', 'neymar', 'lebron', 'curry',
        'djokovic', 'federer', 'nadal'
    ]
    
    # Celebrities
    celebrity_names = [
        'kardashian', 'bieber', 'swift', 'trump', 'biden',
        'musk', 'bezos', 'gates'
    ]
    
    all_names = sports_names + celebrity_names
    
    print(f"\nTotal tweets: {len(df):,}")
    print(f"Cyberbullying tweets (label=1): {(df['label']==1).sum():,}")
    print(f"Not cyberbullying (label=0): {(df['label']==0).sum():,}")
    
    print("\n" + "="*70)
    print("NAME FREQUENCY ANALYSIS")
    print("="*70)
    
    results = []
    
    for name in all_names:
        # Count occurrences in all tweets
        total_count = df['text'].str.contains(name, case=False, na=False).sum()
        
        if total_count == 0:
            continue
        
        # Count in cyberbullying tweets
        cb_mask = (df['label'] == 1) & df['text'].str.contains(name, case=False, na=False)
        cb_count = cb_mask.sum()
        
        # Count in non-cyberbullying tweets
        not_cb_mask = (df['label'] == 0) & df['text'].str.contains(name, case=False, na=False)
        not_cb_count = not_cb_mask.sum()
        
        # Calculate percentage
        cb_percentage = (cb_count / total_count * 100) if total_count > 0 else 0
        
        results.append({
            'name': name,
            'total': total_count,
            'cyberbullying': cb_count,
            'not_cyberbullying': not_cb_count,
            'cb_percentage': cb_percentage
        })
    
    # Sort by total occurrences
    results.sort(key=lambda x: x['total'], reverse=True)
    
    print(f"\n{'Name':<15} {'Total':<8} {'CB':<8} {'Not CB':<8} {'CB %':<8}")
    print("-"*70)
    
    for r in results:
        print(f"{r['name']:<15} {r['total']:<8} {r['cyberbullying']:<8} {r['not_cyberbullying']:<8} {r['cb_percentage']:.1f}%")
    
    # Show sample tweets with these names
    print("\n" + "="*70)
    print("SAMPLE TWEETS WITH 'VIRAT'")
    print("="*70)
    
    virat_tweets = df[df['text'].str.contains('virat', case=False, na=False)]
    
    if len(virat_tweets) > 0:
        print(f"\nFound {len(virat_tweets)} tweets with 'virat'")
        
        # Show cyberbullying examples
        cb_tweets = virat_tweets[virat_tweets['label'] == 1].head(5)
        print(f"\nCyberbullying tweets ({len(cb_tweets)} samples):")
        for i, (idx, row) in enumerate(cb_tweets.iterrows(), 1):
            print(f"\n[{i}] {row['text'][:100]}...")
        
        # Show non-cyberbullying examples
        not_cb_tweets = virat_tweets[virat_tweets['label'] == 0].head(5)
        print(f"\n\nNon-cyberbullying tweets ({len(not_cb_tweets)} samples):")
        for i, (idx, row) in enumerate(not_cb_tweets.iterrows(), 1):
            print(f"\n[{i}] {row['text'][:100]}...")
    else:
        print("\nNo tweets found with 'virat'")
    
    # Overall statistics
    print("\n" + "="*70)
    print("KEY FINDINGS")
    print("="*70)
    
    if results:
        avg_cb_percentage = sum(r['cb_percentage'] for r in results) / len(results)
        print(f"\nAverage CB% for names found: {avg_cb_percentage:.1f}%")
        print(f"Overall dataset CB%: {(df['label']==1).sum() / len(df) * 100:.1f}%")
        
        high_bias = [r for r in results if r['cb_percentage'] > 90]
        if high_bias:
            print(f"\n⚠️  Names with >90% cyberbullying association:")
            for r in high_bias:
                print(f"  - {r['name']}: {r['cb_percentage']:.1f}%")


if __name__ == "__main__":
    csv_path = 'data/processed/train.csv'
    analyze_name_patterns(csv_path)
