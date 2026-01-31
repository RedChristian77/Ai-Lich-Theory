"""
BASIN TO ROUTING TEST - WITH BERT EMBEDDINGS
=============================================

Same test as before, but using BERT for input representation.
BERT understands meaning. Character hashing doesn't.

Let's see if meaningful embeddings improve routing accuracy.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import random
from transformers import BertTokenizer, BertModel

# Reproducibility
torch.manual_seed(42)
random.seed(42)

# ============================================
# LOAD BERT (small version)
# ============================================

print("Loading BERT...")
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert = BertModel.from_pretrained('bert-base-uncased')
bert.eval()  # Freeze BERT - we're just using it for embeddings

print("BERT loaded.")

# ============================================
# TRAINING DATA
# ============================================

math_examples = [
    "what is 2 plus 2",
    "calculate 5 times 3",
    "divide 10 by 2",
    "what is 15 minus 7",
    "multiply 6 and 8",
    "sum of 12 and 15",
    "subtract 20 from 50",
    "9 times 9 equals",
    "half of 100",
    "add 33 and 67",
    "what is 144 divided by 12",
    "square root of 81",
    "7 plus 8 plus 9",
    "triple 15",
    "25 percent of 200",
]

creative_examples = [
    "write me a poem",
    "tell a story about dragons",
    "create a short tale",
    "compose a song",
    "imagine a world where",
    "describe a magical forest",
    "write fiction about robots",
    "tell me a fairy tale",
    "create a character named",
    "write a haiku about rain",
    "invent a new superhero",
    "describe an alien planet",
    "write dialogue between friends",
    "create a mystery story",
    "imagine life in 3000",
]

training_data = [(text, 0) for text in math_examples] + \
                [(text, 1) for text in creative_examples]

# ============================================
# TEST DATA - Never seen during training
# ============================================

test_data = [
    ("what is 7 times 6", 0),
    ("add 100 and 250", 0),
    ("divide 81 by 9", 0),
    ("calculate the sum of 5 5 5", 0),
    ("multiply 12 by 11", 0),
    ("write a poem about the ocean", 1),
    ("tell me a story about a knight", 1),
    ("create a fantasy world", 1),
    ("imagine a talking cat", 1),
    ("describe a haunted house", 1),
]

# ============================================
# TEXT TO BERT EMBEDDING
# ============================================

def text_to_bert_embedding(text):
    """
    Convert text to BERT embedding.
    BERT understands meaning, not just characters.
    """
    with torch.no_grad():
        inputs = tokenizer(text, return_tensors='pt', 
                          padding=True, truncation=True, max_length=64)
        outputs = bert(**inputs)
        # Use [CLS] token embedding as sentence representation
        embedding = outputs.last_hidden_state[:, 0, :]  # (1, 768)
    return embedding

# ============================================
# MODEL: Attention Router on top of BERT
# ============================================

class AttentionRouter(nn.Module):
    def __init__(self, embed_dim=768, num_heads=4, num_routes=2):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads)
        self.router_head = nn.Linear(embed_dim, num_routes)
    
    def forward(self, x):
        # x shape: (1, batch, embed_dim)
        attn_out, attn_weights = self.attention(x, x, x)
        pooled = attn_out.squeeze(0)  # (batch, embed_dim)
        route_logits = self.router_head(pooled)
        return route_logits, attn_weights

# ============================================
# TRAINING
# ============================================

# BERT outputs 768-dim embeddings
model = AttentionRouter(embed_dim=768, num_heads=4, num_routes=2)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

print("=" * 60)
print("BASIN TO ROUTING TEST - BERT EMBEDDINGS")
print("=" * 60)
print(f"Embedding dim: 768 (BERT)")
print(f"Training examples: {len(training_data)}")
print(f"Test examples: {len(test_data)}")
print("=" * 60)

num_epochs = 100
losses = []

for epoch in range(num_epochs):
    random.shuffle(training_data)
    epoch_loss = 0
    
    for text, label in training_data:
        optimizer.zero_grad()
        
        # Get BERT embedding
        embedding = text_to_bert_embedding(text)  # (1, 768)
        x = embedding.unsqueeze(0)  # (1, 1, 768) for attention
        
        # Forward
        route_logits, _ = model(x)
        
        # Loss
        target = torch.tensor([label])
        loss = criterion(route_logits, target)
        
        # Backward
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
    
    avg_loss = epoch_loss / len(training_data)
    losses.append(avg_loss)
    
    if epoch in [0, 24, 49, 74, 99]:
        print(f"Epoch {epoch+1}: avg loss = {avg_loss:.4f}")

# ============================================
# TESTING
# ============================================

print("\
" + "=" * 60)
print("TESTING ON UNSEEN EXAMPLES")
print("=" * 60)

model.eval()
correct = 0
results = []

with torch.no_grad():
    for text, expected in test_data:
        embedding = text_to_bert_embedding(text)
        x = embedding.unsqueeze(0)
        
        route_logits, _ = model(x)
        predicted = route_logits.argmax(dim=1).item()
        confidence = torch.softmax(route_logits, dim=1).max().item()
        
        is_correct = predicted == expected
        if is_correct:
            correct += 1
        
        results.append({
            'text': text,
            'expected': expected,
            'predicted': predicted,
            'correct': is_correct
        })
        
        route_name = "math" if predicted == 0 else "creative"
        expected_name = "math" if expected == 0 else "creative"
        status = "\u2713" if is_correct else "\u2717"
        
        print(f"{status} '{text[:40]}'")
        print(f"    Expected: {expected_name}, Got: {route_name} (conf: {confidence:.3f})")

# ============================================
# FINAL ANALYSIS
# ============================================

print("\
" + "=" * 60)
print("RAW RESULTS - BERT VERSION")
print("=" * 60)

accuracy = correct / len(test_data)
print(f"\
Accuracy: {correct}/{len(test_data)} = {accuracy:.1%}")

print(f"\
Loss progression:")
print(f"  Epoch 1:   {losses[0]:.4f}")
print(f"  Epoch 25:  {losses[24]:.4f}")
print(f"  Epoch 50:  {losses[49]:.4f}")
print(f"  Epoch 100: {losses[99]:.4f}")

math_correct = sum(1 for r in results if r['expected'] == 0 and r['correct'])
creative_correct = sum(1 for r in results if r['expected'] == 1 and r['correct'])

print(f"\
Per-category:")
print(f"  Math: {math_correct}/5")
print(f"  Creative: {creative_correct}/5")

print("\
" + "=" * 60)
print("COMPARISON")
print("=" * 60)
print("Character hash version: 60% accuracy (64 dim)")
print(f"BERT embedding version: {accuracy:.0%} accuracy (768 dim)")
print("\
Did semantic understanding improve routing?")
print("=" * 60)
