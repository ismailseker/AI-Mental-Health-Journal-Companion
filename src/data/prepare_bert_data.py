"""
Data preparation script for BERT fine-tuning on mental health data.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import json
from pathlib import Path

def load_and_prepare_data(csv_path: str):
    """Load and prepare mental health data for BERT training."""
    
    print("📊 Loading mental health data...")
    df = pd.read_csv(csv_path)
    
    print(f"📈 Dataset shape: {df.shape}")
    print(f"📋 Columns: {df.columns.tolist()}")
    
    # Clean data
    df = df.dropna(subset=['statement', 'status'])
    df = df[df['statement'].str.len() > 10]  # Remove very short texts
    
    print(f"🧹 After cleaning: {df.shape}")
    
    # Analyze status distribution
    status_counts = df['status'].value_counts()
    print(f"📊 Status distribution:")
    print(status_counts)
    
    # Create sentiment mapping
    sentiment_mapping = {
        'Anxiety': 'negative',
        'Depression': 'negative', 
        'Suicide': 'negative',
        'Bipolar': 'negative',
        'PTSD': 'negative',
        'Schizophrenia': 'negative',
        'Eating Disorder': 'negative',
        'OCD': 'negative',
        'ADHD': 'neutral',
        'Autism': 'neutral',
        'Borderline': 'negative',
        'Addiction': 'negative',
        'Grief': 'negative',
        'Panic': 'negative',
        'Stress': 'negative',
        'Trauma': 'negative',
        'Self-harm': 'negative',
        'Mania': 'negative',
        'Psychosis': 'negative',
        'Dissociation': 'negative'
    }
    
    # Map status to sentiment
    df['sentiment'] = df['status'].map(sentiment_mapping)
    df = df.dropna(subset=['sentiment'])
    
    print(f"🎯 Final dataset: {df.shape}")
    print(f"📊 Sentiment distribution:")
    print(df['sentiment'].value_counts())
    
    return df

def create_train_test_split(df, test_size=0.2, val_size=0.1):
    """Create train/validation/test splits."""
    
    # First split: train+val vs test
    train_val, test = train_test_split(
        df, test_size=test_size, random_state=42, stratify=df['sentiment']
    )
    
    # Second split: train vs val
    train, val = train_test_split(
        train_val, test_size=val_size/(1-test_size), random_state=42, stratify=train_val['sentiment']
    )
    
    print(f"📊 Data splits:")
    print(f"  Train: {len(train)} samples")
    print(f"  Validation: {len(val)} samples") 
    print(f"  Test: {len(test)} samples")
    
    return train, val, test

def save_datasets(train, val, test, output_dir: str):
    """Save datasets in different formats."""
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save as CSV
    train.to_csv(output_path / "train.csv", index=False)
    val.to_csv(output_path / "val.csv", index=False)
    test.to_csv(output_path / "test.csv", index=False)
    
    # Save as JSON for BERT training
    def df_to_json(df, filename):
        data = []
        for _, row in df.iterrows():
            data.append({
                "text": row['statement'],
                "label": row['sentiment'],
                "status": row['status']
            })
        
        with open(output_path / filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    df_to_json(train, "train.json")
    df_to_json(val, "val.json")
    df_to_json(test, "test.json")
    
    print(f"💾 Datasets saved to: {output_path}")

def main():
    """Main data preparation pipeline."""
    
    # Paths
    data_path = "/Users/sekerismail/Desktop/AIMentalHealthJournalCompanion/data/data.csv"
    output_dir = "/Users/sekerismail/Desktop/AIMentalHealthJournalCompanion/data/processed"
    
    print("🚀 Starting BERT data preparation...")
    
    # Load and prepare data
    df = load_and_prepare_data(data_path)
    
    # Create splits
    train, val, test = create_train_test_split(df)
    
    # Save datasets
    save_datasets(train, val, test, output_dir)
    
    print("✅ Data preparation completed!")
    
    # Print sample data
    print("\n📝 Sample training data:")
    for i, row in train.head(3).iterrows():
        print(f"Text: {row['statement'][:100]}...")
        print(f"Status: {row['status']}")
        print(f"Sentiment: {row['sentiment']}")
        print("-" * 50)

if __name__ == "__main__":
    main()
