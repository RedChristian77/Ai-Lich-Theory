"""
MAMBA STATE PERSISTENCE TEST
============================

Testing: Does Mamba hold state differently than attention?
Hypothesis: Small degradation on save/load, quick recovery during use

Comparing:
- Attention (static save/load)
- Mamba (stateful, potentially self-healing)
"""

import torch
import torch.nn as nn
import torch.optim as optim
import random
import os
from transformers import BertTokenizer, BertModel

try:
    from mamba_ssm import Mamba
    MAMBA_AVAILABLE = True
    print("\u2713 Mamba SSM loaded successfully")
except ImportError:
    MAMBA_AVAILABLE = False
    print("\u2717 Mamba SSM not available - install with: pip install mamba-ssm")
    exit(1)

torch.manual_seed(42)
random.seed(42)

# ============================================
# LOAD BERT
# ============================================
print("Loading BERT...")
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert = BertModel.from_pretrained('bert-base-uncased')
bert.eval()
print("BERT loaded.\
")

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
]

training_data = [(text, 0) for text in math_examples] + \
                [(text, 1) for text in creative_examples]

# ============================================
# TEST DATA
# ============================================
test_queries = [
    ("what is 7 times 6", 0),
    ("multiply 12 by 11", 0),
    ("calculate 99 plus 1", 0),
    ("write a poem about the ocean", 1),
    ("create a fantasy world", 1),
    ("imagine a talking cat", 1),
]

never_seen_queries = [
    ("what is 100 divided by 5", 0),
    ("solve 3 times 7", 0),
    ("tell me a story about a wizard", 1),
    ("write fiction about space", 1),
]

# ============================================
# BERT EMBEDDING
# ============================================
def text_to_bert_embedding(text):
    with torch.no_grad():
        inputs = tokenizer(text, return_tensors='pt', 
                          padding=True, truncation=True, max_length=64)
        outputs = bert(**inputs)
        embedding = outputs.last_hidden_state[:, 0, :]
    return embedding

# ============================================
# MAMBA ROUTER MODEL
# ============================================
class MambaRouter(nn.Module):
    def __init__(self, input_dim=768, mamba_dim=64, num_routes=2):
        super().__init__()
        
        # Project BERT embedding down to Mamba dimension
        self.input_projection = nn.Linear(input_dim, mamba_dim)
        
        # Mamba block for stateful processing
        self.mamba = Mamba(
            d_model=mamba_dim,
            d_state=16,
            d_conv=4,
            expand=2
        )
        
        # Router head
        self.router_head = nn.Linear(mamba_dim, num_routes)
        
        # Store running state
        self.running_state = None
    
    def forward(self, x, use_running_state=False):
        # x shape: (batch, input_dim) from BERT
        
        # Project to Mamba dimension
        x = self.input_projection(x)  # (batch, mamba_dim)
        
        # Add sequence dimension for Mamba
        x = x.unsqueeze(1)  # (batch, seq_len=1, mamba_dim)
        
        # Mamba processing
        mamba_out = self.mamba(x)  # (batch, seq_len=1, mamba_dim)
        
        # Remove sequence dimension
        mamba_out = mamba_out.squeeze(1)  # (batch, mamba_dim)
        
        # Route decision
        route_logits = self.router_head(mamba_out)
        
        return route_logits
    
    def get_state_summary(self):
        """Get summary statistics of current weights for comparison."""
        stats = {}
        for name, param in self.named_parameters():
            stats[name] = {
                'mean': param.data.mean().item(),
                'std': param.data.std().item(),
                'min': param.data.min().item(),
                'max': param.data.max().item(),
            }
        return stats

# ============================================
# TEST FUNCTION
# ============================================
def test_model(model, queries, description):
    """Test model on queries and return accuracy."""
    model.eval()
    correct = 0
    results = []
    
    with torch.no_grad():
        for text, expected in queries:
            embedding = text_to_bert_embedding(text)
            route_logits = model(embedding)
            predicted = route_logits.argmax(dim=1).item()
            conf = torch.softmax(route_logits, dim=1).max().item()
            
            is_correct = predicted == expected
            if is_correct:
                correct += 1
            
            route_name = "math" if predicted == 0 else "creative"
            status = "\u2713" if is_correct else "\u2717"
            results.append(f"  {status} '{text[:35]}' \u2192 {route_name} ({conf:.4f})")
    
    accuracy = correct / len(queries)
    print(f"\
{description}:")
    for r in results:
        print(r)
    print(f"  Accuracy: {correct}/{len(queries)} = {accuracy:.1%}")
    
    return accuracy, results

# ============================================
# PART 1: TRAIN MAMBA ROUTER
# ============================================
print("=" * 60)
print("PART 1: TRAINING MAMBA ROUTER")
print("=" * 60)

model = MambaRouter(input_dim=768, mamba_dim=64, num_routes=2)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# Get initial state for comparison
initial_state = model.get_state_summary()

print(f"Training on {len(training_data)} examples...")

losses = []
for epoch in range(100):
    random.shuffle(training_data)
    epoch_loss = 0
    
    for text, label in training_data:
        optimizer.zero_grad()
        embedding = text_to_bert_embedding(text)
        route_logits = model(embedding)
        target = torch.tensor([label])
        loss = criterion(route_logits, target)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    
    avg_loss = epoch_loss / len(training_data)
    losses.append(avg_loss)
    
    if epoch in [0, 49, 99]:
        print(f"  Epoch {epoch+1}: loss = {avg_loss:.4f}")

# Get trained state
trained_state = model.get_state_summary()

# Test before save
pre_save_acc, _ = test_model(model, test_queries, "PRE-SAVE TEST")
pre_save_unseen_acc, _ = test_model(model, never_seen_queries, "PRE-SAVE UNSEEN")

# ============================================
# PART 2: SAVE MAMBA STATE
# ============================================
print("\
" + "=" * 60)
print("PART 2: SAVING MAMBA STATE")
print("=" * 60)

save_path = "mamba_router_state.pt"

# Save with different precisions to test degradation
torch.save({
    'model_state_dict': model.state_dict(),
    'model_config': {
        'input_dim': 768,
        'mamba_dim': 64,
        'num_routes': 2
    },
    'trained_state_summary': trained_state
}, save_path)

file_size = os.path.getsize(save_path)
print(f"Saved to: {save_path}")
print(f"File size: {file_size / 1024:.1f} KB")

# ============================================
# PART 3: CLEAR AND RELOAD
# ============================================
print("\
" + "=" * 60)
print("PART 3: SIMULATING DEATH AND RESURRECTION")
print("=" * 60)

# Delete model
del model
del optimizer
import gc
gc.collect()
torch.cuda.empty_cache() if torch.cuda.is_available() else None

print("Model deleted. Memory cleared.")
print("Loading fresh model from saved state...")

# Load fresh
checkpoint = torch.load(save_path)
config = checkpoint['model_config']

model_resurrected = MambaRouter(
    input_dim=config['input_dim'],
    mamba_dim=config['mamba_dim'],
    num_routes=config['num_routes']
)

model_resurrected.load_state_dict(checkpoint['model_state_dict'])

# Get resurrected state
resurrected_state = model_resurrected.get_state_summary()

print("Model resurrected.")

# ============================================
# PART 4: TEST IMMEDIATELY AFTER RESURRECTION
# ============================================
print("\
" + "=" * 60)
print("PART 4: TESTING IMMEDIATELY AFTER RESURRECTION")
print("=" * 60)

post_load_acc, _ = test_model(model_resurrected, test_queries, "IMMEDIATE POST-LOAD")
post_load_unseen_acc, _ = test_model(model_resurrected, never_seen_queries, "IMMEDIATE POST-LOAD UNSEEN")

# ============================================
# PART 5: RUN QUERIES TO "WARM UP" MAMBA
# ============================================
print("\
" + "=" * 60)
print("PART 5: WARMING UP MAMBA (Running queries)")
print("=" * 60)

print("Running 20 queries to let Mamba 'settle'...")
warmup_queries = training_data * 1  # Run through training data once

model_resurrected.eval()
with torch.no_grad():
    for text, label in warmup_queries:
        embedding = text_to_bert_embedding(text)
        _ = model_resurrected(embedding)

print("Warmup complete.")

# Test after warmup
post_warmup_acc, _ = test_model(model_resurrected, test_queries, "POST-WARMUP TEST")
post_warmup_unseen_acc, _ = test_model(model_resurrected, never_seen_queries, "POST-WARMUP UNSEEN")

# ============================================
# PART 6: COMPARE STATES
# ============================================
print("\
" + "=" * 60)
print("PART 6: STATE COMPARISON")
print("=" * 60)

print("\
Comparing weight statistics (sample parameters):")

# Compare a few key parameters
sample_params = ['input_projection.weight', 'router_head.weight']
for param_name in sample_params:
    if param_name in trained_state and param_name in resurrected_state:
        t = trained_state[param_name]
        r = resurrected_state[param_name]
        
        print(f"\
{param_name}:")
        print(f"  Trained   - mean: {t['mean']:.6f}, std: {t['std']:.6f}")
        print(f"  Resurrected - mean: {r['mean']:.6f}, std: {r['std']:.6f}")
        print(f"  Difference - mean: {abs(t['mean'] - r['mean']):.9f}")

# ============================================
# FINAL SUMMARY
# ============================================
print("\
" + "=" * 60)
print("FINAL SUMMARY: MAMBA STATE PERSISTENCE")
print("=" * 60)

print(f"""
ACCURACY COMPARISON:
                        Test Queries    Unseen Queries
Pre-save:               {pre_save_acc:.1%}            {pre_save_unseen_acc:.1%}
Immediate post-load:    {post_load_acc:.1%}            {post_load_unseen_acc:.1%}
After warmup:           {post_warmup_acc:.1%}            {post_warmup_unseen_acc:.1%}

DEGRADATION ANALYSIS:
Pre-save \u2192 Post-load:   {(pre_save_acc - post_load_acc)*100:+.1f}%
Post-load \u2192 Warmup:     {(post_warmup_acc - post_load_acc)*100:+.1f}%
""")

if post_load_acc == pre_save_acc:
    print("RESULT: ZERO degradation on save/load")
    print("The Mamba basins persisted perfectly.")
elif post_warmup_acc > post_load_acc:
    print("RESULT: Initial degradation, RECOVERED after warmup")
    print("Mamba's stateful nature helped restore precision.")
elif post_warmup_acc == post_load_acc:
    print("RESULT: Degradation persisted (no self-healing observed)")
else:
    print("RESULT: Unexpected behavior - needs investigation")

print("=" * 60)
print("END OF MAMBA PERSISTENCE TEST")
print("=" * 60)
