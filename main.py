import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Tuple
from collections import Counter
import re

#CBOW (predicts target word from context words)
class CBOW(nn.Module):
    def __init__(self, vocab_size: int, embedding_size: int):
        super(CBOW, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_size)
        self.linear = nn.Linear(embedding_size, vocab_size)
    
    def forward(self, inputs):
        # Get embeddings and average them
        embeds = self.embeddings(inputs)
        hidden = torch.mean(embeds, dim=1)
        # Get output and apply log softmax
        out = self.linear(hidden)
        return torch.log_softmax(out, dim=1)

#Skip-gram (predicts context words from target word)
class SkipGram(nn.Module):
    def __init__(self, vocab_size: int, embedding_size: int):
        super(SkipGram, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_size)
        self.linear = nn.Linear(embedding_size, vocab_size)
    
    def forward(self, inputs):
        embeds = self.embeddings(inputs)
        out = self.linear(embeds)
        return torch.log_softmax(out, dim=1)

def preprocess_text(text: str) -> str:
    """Clean and normalize text."""
    # Convert to lowercase and remove special characters
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text

def load_sms_data(file_path: str) -> List[str]:
    """Load SMS messages from the dataset."""
    messages = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            # Split by tab and take the message part
            label, message = line.strip().split('\t')
            messages.append(preprocess_text(message))
    return messages

def create_dataset(texts: List[str], window_size: int = 2):
    """Create vocabulary and training pairs from texts."""
    # Create vocabulary
    words = [word for text in texts for word in text.split()]
    word_counts = Counter(words)
    
    # Filter words by minimum frequency (2)
    vocab = {'<PAD>': 0, '<UNK>': 1}
    idx = len(vocab)
    for word, count in word_counts.items():
        if count >= 2:  # minimum frequency
            vocab[word] = idx
            idx += 1
    
    # Create training pairs
    data = []
    for text in texts:
        words = text.split()
        for i in range(len(words)):
            # Get context window
            start = max(0, i - window_size)
            end = min(len(words), i + window_size + 1)
            
            context = words[start:i] + words[i+1:end]
            target = words[i]
            
            # Convert words to indices
            context_idx = [vocab.get(w, vocab['<UNK>']) for w in context]
            target_idx = vocab.get(target, vocab['<UNK>'])
            
            # Pad context if needed
            while len(context_idx) < 2 * window_size:
                context_idx.append(vocab['<PAD>'])
            
            data.append((torch.tensor(context_idx), torch.tensor(target_idx)))
    
    return data, vocab

def train_word2vec(file_path: str, model_type: str = 'cbow', 
                   embedding_size: int = 100, epochs: int = 5):
    """Train Word2Vec model on SMS dataset."""
    # Load and preprocess data
    print("Loading SMS dataset...")
    texts = load_sms_data(file_path)
    
    # Create dataset
    print("Creating dataset...")
    data, vocab = create_dataset(texts)
    vocab_size = len(vocab)
    
    # Create model
    print(f"Training {model_type.upper()} model...")
    if model_type == 'cbow':
        model = CBOW(vocab_size, embedding_size)
    else:
        model = SkipGram(vocab_size, embedding_size)
    
    # Setup training
    criterion = nn.NLLLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    dataloader = DataLoader(data, batch_size=32, shuffle=True)
    
    # Training loop
    for epoch in range(epochs):
        total_loss = 0
        for context, target in dataloader:
            # Forward pass
            optimizer.zero_grad()
            output = model(context)
            loss = criterion(output, target)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(dataloader)
        print(f'Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}')
    
    return model, vocab

if __name__ == "__main__":
    # Train both models
    print("Training CBOW model...")
    cbow_model, vocab = train_word2vec('SMSSpamCollection.txt', model_type='cbow')
    
    # print("\nTraining Skip-gram model...")
    # skipgram_model, vocab = train_word2vec('SMSSpamCollection.txt', model_type='skipgram')