"""
BERT-based sentiment analysis model for mental health journal entries.
"""

import torch
import torch.nn as nn
from transformers import (
    AutoTokenizer, 
    AutoModel, 
    AutoConfig,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import json
import numpy as np
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BERTMentalHealthClassifier(nn.Module):
    """BERT-based classifier for mental health sentiment analysis."""
    
    def __init__(self, model_name: str = "bert-base-uncased", num_labels: int = 3):
        super().__init__()
        
        self.config = AutoConfig.from_pretrained(model_name)
        self.bert = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(self.config.hidden_size, num_labels)
        
    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.classifier.out_features), labels.view(-1))
        
        return {"loss": loss, "logits": logits}

class MentalHealthDataset(torch.utils.data.Dataset):
    """Dataset class for mental health data."""
    
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

class BERTMentalHealthTrainer:
    """Trainer class for BERT mental health model."""
    
    def __init__(self, model_name: str = "bert-base-uncased"):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = None
        self.label_encoder = None
        
    def prepare_data(self, data_path: str):
        """Prepare training data."""
        
        logger.info(f"📊 Loading data from {data_path}")
        
        # Load JSON data
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        texts = [item['text'] for item in data]
        labels = [item['label'] for item in data]
        
        # Encode labels
        unique_labels = list(set(labels))
        self.label_encoder = {label: idx for idx, label in enumerate(unique_labels)}
        encoded_labels = [self.label_encoder[label] for label in labels]
        
        logger.info(f"📈 Loaded {len(texts)} samples")
        logger.info(f"🏷️ Labels: {unique_labels}")
        
        return texts, encoded_labels
    
    def create_datasets(self, train_path: str, val_path: str):
        """Create training and validation datasets."""
        
        # Prepare training data
        train_texts, train_labels = self.prepare_data(train_path)
        val_texts, val_labels = self.prepare_data(val_path)
        
        # Create datasets
        train_dataset = MentalHealthDataset(train_texts, train_labels, self.tokenizer)
        val_dataset = MentalHealthDataset(val_texts, val_labels, self.tokenizer)
        
        return train_dataset, val_dataset
    
    def compute_metrics(self, eval_pred):
        """Compute evaluation metrics."""
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        
        precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='weighted')
        accuracy = accuracy_score(labels, predictions)
        
        return {
            'accuracy': accuracy,
            'f1': f1,
            'precision': precision,
            'recall': recall
        }
    
    def train(self, train_path: str, val_path: str, output_dir: str):
        """Train the BERT model."""
        
        logger.info("🚀 Starting BERT training...")
        
        # Create datasets
        train_dataset, val_dataset = self.create_datasets(train_path, val_path)
        
        # Initialize model
        self.model = BERTMentalHealthClassifier(
            model_name=self.model_name,
            num_labels=len(self.label_encoder)
        )
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=3,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir=f'{output_dir}/logs',
            logging_steps=100,
            evaluation_strategy="steps",
            eval_steps=500,
            save_strategy="steps",
            save_steps=500,
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            greater_is_better=True,
            report_to=None,  # Disable wandb
        )
        
        # Create trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=self.compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
        )
        
        # Train model
        logger.info("🏋️ Training started...")
        trainer.train()
        
        # Save model
        trainer.save_model()
        self.tokenizer.save_pretrained(output_dir)
        
        # Save model state dict for loading
        torch.save(self.model.state_dict(), f"{output_dir}/pytorch_model.bin")
        
        # Save label encoder
        with open(f"{output_dir}/label_encoder.json", 'w') as f:
            json.dump(self.label_encoder, f)
        
        logger.info(f"✅ Model saved to {output_dir}")
        
        return trainer
    
    def load_model(self, model_path: str):
        """Load trained model."""
        
        logger.info(f"📥 Loading model from {model_path}")
        
        # Load label encoder
        with open(f"{model_path}/label_encoder.json", 'r') as f:
            self.label_encoder = json.load(f)
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # Load model
        self.model = BERTMentalHealthClassifier(
            model_name=model_path,
            num_labels=len(self.label_encoder)
        )
        
        # Load weights
        self.model.load_state_dict(torch.load(f"{model_path}/pytorch_model.bin", map_location='cpu'))
        self.model.eval()
        
        logger.info("✅ Model loaded successfully")
    
    def predict(self, text: str):
        """Predict sentiment for a single text."""
        
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        # Tokenize input
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=512,
            return_tensors='pt'
        )
        
        # Get prediction
        with torch.no_grad():
            outputs = self.model(**encoding)
            logits = outputs['logits']
            probabilities = torch.softmax(logits, dim=-1)
            predicted_class = torch.argmax(probabilities, dim=-1).item()
        
        # Get label
        label = list(self.label_encoder.keys())[predicted_class]
        confidence = probabilities[0][predicted_class].item()
        
        return {
            'prediction': label,
            'confidence': confidence,
            'probabilities': {
                label: prob.item() for label, prob in zip(
                    self.label_encoder.keys(), 
                    probabilities[0]
                )
            }
        }

def main():
    """Main training function."""
    
    # Paths
    data_dir = "/Users/sekerismail/Desktop/AIMentalHealthJournalCompanion/data/processed"
    train_path = f"{data_dir}/train.json"
    val_path = f"{data_dir}/val.json"
    output_dir = "/Users/sekerismail/Desktop/AIMentalHealthJournalCompanion/models/bert_mental_health"
    
    # Create trainer
    trainer = BERTMentalHealthTrainer()
    
    # Train model
    trainer.train(train_path, val_path, output_dir)
    
    print("🎉 BERT training completed!")

if __name__ == "__main__":
    main()
