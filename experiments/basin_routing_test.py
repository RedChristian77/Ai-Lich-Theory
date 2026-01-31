"""
BASIN TO ROUTING TEST - Volume Learning
========================================

PURPOSE: Test if attention basins can learn to route
         different types of input to different outputs
         through pure volume of examples.

We're not telling it WHAT makes math "math."
We're showing examples. Basins figure out the pattern.

THIS IS OBSERVATION. All outcomes are valid data.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import random

# Reproducibility
torch.manual_seed(42)
random.seed(42)

# ============================================
# TRAINING DATA - Volume creates the basins
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

# Label: 0 = math, 1 = creative
training_data = [(text, 0) for text in math_examples] + \
                [(text, 1) for text in creative_examples]

# ============================================
# TEST DATA - Never seen during training
# ============================================

test_data = [
    # Math-like (should route to 0)
    ("what is 7 times 6", 0),
    ("add 100 and 250", 0),
    ("divide 81 by 9", 0),
    ("calculate the sum of 5 5 5", 0),
    ("multiply 12 by 11", 0),
    
    # Creative-like (should route to 1)
    ("write a poem about the ocean", 1),
    ("tell me a story about a knight", 1),
    ("create a fantasy world", 1),
    ("imagine a talking cat", 1),
    ("describe a haunted house", 1),
]

# ============================================
# SIMPLE TEXT TO TENSOR
# ============================================

def text_to_tensor(text, embed_dim=64):
    """
    Convert text to tensor using simple character hashing.
    Not sophisticated - just needs to be CONSISTENT.
    Same text \u2192 same tensor. Similar text \u2192 similar tensor.
    """
    # Create embedding from character values
    values = [ord(c) % 100 / 100.0 for c in text.lower()[:50]]
    # Pad or truncate to fixed length
    values = values + [0.0] * (50 - len(values))
    
    # Project to embed_dim using fixed random projection
    torch.manual_seed(123)  # Fixed seed for consistent projection
    projection = torch.randn(50, embed_dim)
    
    tensor = torch.tensor(values).float() @ projection
    # Normalize
    tensor = tensor / (tensor.norm() + 1e-8)
    
    return tensor.unsqueeze(0).unsqueeze(0)  # (1, 1, embed_dim)

# ============================================
# MODEL: Attention + Router Head
# ============================================

class AttentionRouter(nn.Module):
    def __init__(self, embed_dim=64, num_heads=4, num_routes=2):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads)
        self.router_head = nn.Linear(embed_dim, num_routes)
    
    def forward(self, x):
        # x shape: (seq_len, batch, embed_dim)
        attn_out, attn_weights = self.attention(x, x, x)
        
        # Pool attention output (take mean across sequence)
        pooled = attn_out.mean(dim=0)  # (batch, embed_dim)
        
        # Route decision
        route_logits = self.router_head(pooled)  # (batch, num_routes)
        
        return route_logits, attn_weights

# ============================================
# TRAINING
# ============================================

embed_dim = 64
model = AttentionRouter(embed_dim=embed_dim, num_heads=4, num_routes=2)
optimizer = optim.Adam(model.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss()

print("=" * 60)
print("BASIN TO ROUTING TEST")
print("=" * 60)
print(f"Training examples: {len(training_data)}")
print(f"Test examples: {len(test_data)}")
print(f"Routes: 0 = math, 1 = creative")
print("=" * 60)

# Training loop - multiple epochs for volume
num_epochs = 100
losses = []

for epoch in range(num_epochs):
    random.shuffle(training_data)
    epoch_loss = 0
    
    for text, label in training_data:
        optimizer.zero_grad()
        
        # Convert text to tensor
        x = text_to_tensor(text, embed_dim)
        x = x.transpose(0, 1)  # (seq, batch, dim)
        
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
# TESTING - Does it generalize?
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
        x = text_to_tensor(text, embed_dim)
        x = x.transpose(0, 1)
        
        route_logits, attn_weights = model(x)
        predicted = route_logits.argmax(dim=1).item()
        confidence = torch.softmax(route_logits, dim=1).max().item()
        
        is_correct = predicted == expected
        if is_correct:
            correct += 1
        
        results.append({
            'text': text,
            'expected': expected,
            'predicted': predicted,
            'confidence': confidence,
            'correct': is_correct
        })
        
        route_name = "math" if predicted == 0 else "creative"
        expected_name = "math" if expected == 0 else "creative"
        status = "\u2713" if is_correct else "\u2717"
        
        print(f"{status} '{text[:35]}...'")
        print(f"    Expected: {expected_name}, Got: {route_name} (conf: {confidence:.3f})")

# ============================================
# FINAL ANALYSIS
# ============================================

print("\
" + "=" * 60)
print("RAW RESULTS")
print("=" * 60)

accuracy = correct / len(test_data)
print(f"\
Accuracy: {correct}/{len(test_data)} = {accuracy:.1%}")

print(f"\
Loss progression:")
print(f"  Epoch 1:   {losses[0]:.4f}")
print(f"  Epoch 25:  {losses[24]:.4f}")
print(f"  Epoch 50:  {losses[49]:.4f}")
print(f"  Epoch 75:  {losses[74]:.4f}")
print(f"  Epoch 100: {losses[99]:.4f}")

print(f"\
Per-category results:")
math_correct = sum(1 for r in results if r['expected'] == 0 and r['correct'])
math_total = sum(1 for r in results if r['expected'] == 0)
creative_correct = sum(1 for r in results if r['expected'] == 1 and r['correct'])
creative_total = sum(1 for r in results if r['expected'] == 1)

print(f"  Math queries:     {math_correct}/{math_total}")
print(f"  Creative queries: {creative_correct}/{creative_total}")

print("\
" + "=" * 60)
print("END OF TEST")
print("=" * 60)
print("""
This test shows whether attention basins can learn routing from examples.

If accuracy > 50%: Basins learned SOMETHING
If accuracy > 80%: Basins generalize well
If accuracy = 50%: Random chance, no learning

Whatever the result, it's real data.
""")
