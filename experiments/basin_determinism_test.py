"""
BASIN DETERMINISM TEST
======================

Question: Do identical inputs create identical basins?

Test:
- Two fresh models, same architecture, same seed
- Train both with identical data, identical order
- Compare final basin positions
- Measure divergence

If distance \u2248 0: Deterministic (same input = same geometry)
If distance small: Convergent (similar but not identical)
If distance large: Chaotic (each instance unique)
"""

import torch
import torch.nn as nn
import torch.optim as optim
import copy

# ============================================
# MODEL
# ============================================

class AttentionRouter(nn.Module):
    def __init__(self, embed_dim=64, num_heads=4, num_routes=2):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads)
        self.router_head = nn.Linear(embed_dim, num_routes)
    
    def forward(self, x):
        attn_out, attn_weights = self.attention(x, x, x)
        pooled = attn_out.squeeze(0)
        route_logits = self.router_head(pooled)
        return route_logits, attn_weights
    
    def get_all_weights(self):
        """Flatten all weights into single tensor for comparison."""
        weights = []
        for param in self.parameters():
            weights.append(param.data.flatten())
        return torch.cat(weights)

# ============================================
# SIMPLE EMBEDDING (deterministic)
# ============================================

def text_to_embedding(text, dim=64):
    """Same text = same embedding every time."""
    torch.manual_seed(hash(text) % 2**32)
    emb = torch.randn(1, dim)
    return emb

# ============================================
# TRAINING DATA
# ============================================

training_data = [
    ("what is 2 plus 2", 0),
    ("calculate 5 times 3", 0),
    ("divide 10 by 2", 0),
    ("add these numbers", 0),
    ("multiply 6 and 8", 0),
    ("write me a poem", 1),
    ("tell a story", 1),
    ("create fiction", 1),
    ("imagine a world", 1),
    ("describe a sunset", 1),
]

# ============================================
# TRAIN FUNCTION
# ============================================

def train_model(model, data, epochs=100, seed=42):
    """Train model with fixed seed for reproducibility."""
    torch.manual_seed(seed)
    
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(epochs):
        # Same order every time (no shuffle)
        for text, label in data:
            optimizer.zero_grad()
            embedding = text_to_embedding(text)
            x = embedding.unsqueeze(0)
            route_logits, _ = model(x)
            target = torch.tensor([label])
            loss = criterion(route_logits, target)
            loss.backward()
            optimizer.step()
    
    return model

# ============================================
# COMPARISON METRICS
# ============================================

def compare_models(model1, model2):
    """Compare two models' weight spaces."""
    
    weights1 = model1.get_all_weights()
    weights2 = model2.get_all_weights()
    
    # Euclidean distance
    euclidean = torch.dist(weights1, weights2).item()
    
    # Cosine similarity
    cosine = torch.cosine_similarity(weights1.unsqueeze(0), weights2.unsqueeze(0)).item()
    
    # Element-wise difference stats
    diff = torch.abs(weights1 - weights2)
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    
    return {
        'euclidean_distance': euclidean,
        'cosine_similarity': cosine,
        'max_difference': max_diff,
        'mean_difference': mean_diff,
        'num_weights': len(weights1)
    }

def compare_outputs(model1, model2, test_inputs):
    """Compare routing decisions between models."""
    
    agreements = 0
    results = []
    
    for text, expected in test_inputs:
        embedding = text_to_embedding(text)
        x = embedding.unsqueeze(0)
        
        with torch.no_grad():
            logits1, _ = model1(x)
            logits2, _ = model2(x)
            
            pred1 = logits1.argmax(dim=1).item()
            pred2 = logits2.argmax(dim=1).item()
            
            probs1 = torch.softmax(logits1, dim=1)[0]
            probs2 = torch.softmax(logits2, dim=1)[0]
            
            conf_diff = torch.abs(probs1 - probs2).max().item()
        
        if pred1 == pred2:
            agreements += 1
        
        results.append({
            'text': text[:30],
            'pred1': pred1,
            'pred2': pred2,
            'match': pred1 == pred2,
            'conf_diff': conf_diff
        })
    
    return agreements / len(test_inputs), results

# ============================================
# MAIN TEST
# ============================================

def main():
    print("=" * 60)
    print("BASIN DETERMINISM TEST")
    print("=" * 60)
    print("Question: Do identical inputs create identical basins?")
    print()
    
    # ==========================================
    # TEST 1: Same seed, same everything
    # ==========================================
    print("TEST 1: Identical seeds, identical training")
    print("-" * 40)
    
    torch.manual_seed(42)
    model_a1 = AttentionRouter(embed_dim=64, num_heads=4, num_routes=2)
    initial_weights_a1 = model_a1.get_all_weights().clone()
    
    torch.manual_seed(42)
    model_a2 = AttentionRouter(embed_dim=64, num_heads=4, num_routes=2)
    initial_weights_a2 = model_a2.get_all_weights().clone()
    
    # Check initial weights match
    init_dist = torch.dist(initial_weights_a1, initial_weights_a2).item()
    print(f"Initial weight distance: {init_dist:.10f}")
    
    # Train both identically
    print("Training Model A1...")
    model_a1 = train_model(model_a1, training_data, epochs=100, seed=42)
    
    print("Training Model A2...")
    model_a2 = train_model(model_a2, training_data, epochs=100, seed=42)
    
    # Compare
    comparison = compare_models(model_a1, model_a2)
    print(f"\
After training:")
    print(f"  Euclidean distance: {comparison['euclidean_distance']:.10f}")
    print(f"  Cosine similarity: {comparison['cosine_similarity']:.10f}")
    print(f"  Max weight diff: {comparison['max_difference']:.10f}")
    print(f"  Mean weight diff: {comparison['mean_difference']:.10f}")
    
    if comparison['euclidean_distance'] < 1e-6:
        print("\
  RESULT: DETERMINISTIC - Identical inputs = Identical basins")
    else:
        print("\
  RESULT: NON-DETERMINISTIC - Some divergence occurred")
    
    # ==========================================
    # TEST 2: Different seeds, same training data
    # ==========================================
    print("\
" + "=" * 60)
    print("TEST 2: Different seeds, same training data")
    print("-" * 40)
    
    torch.manual_seed(42)
    model_b1 = AttentionRouter(embed_dim=64, num_heads=4, num_routes=2)
    
    torch.manual_seed(123)  # Different seed
    model_b2 = AttentionRouter(embed_dim=64, num_heads=4, num_routes=2)
    
    init_dist = torch.dist(model_b1.get_all_weights(), model_b2.get_all_weights()).item()
    print(f"Initial weight distance: {init_dist:.6f}")
    
    print("Training Model B1 (seed 42)...")
    model_b1 = train_model(model_b1, training_data, epochs=100, seed=42)
    
    print("Training Model B2 (seed 123)...")
    model_b2 = train_model(model_b2, training_data, epochs=100, seed=123)
    
    comparison = compare_models(model_b1, model_b2)
    print(f"\
After training:")
    print(f"  Euclidean distance: {comparison['euclidean_distance']:.6f}")
    print(f"  Cosine similarity: {comparison['cosine_similarity']:.6f}")
    print(f"  Max weight diff: {comparison['max_difference']:.6f}")
    print(f"  Mean weight diff: {comparison['mean_difference']:.6f}")
    
    # Do they route the same despite different weights?
    test_inputs = [
        ("what is 7 times 6", 0),
        ("subtract 10 from 20", 0),
        ("write a poem about love", 1),
        ("create a fantasy story", 1),
    ]
    
    agreement, results = compare_outputs(model_b1, model_b2, test_inputs)
    print(f"\
  Routing agreement: {agreement:.1%}")
    for r in results:
        match = "\u2713" if r['match'] else "\u2717"
        print(f"    {match} '{r['text']}' \u2192 B1:{r['pred1']}, B2:{r['pred2']} (conf diff: {r['conf_diff']:.4f})")
    
    if agreement == 1.0:
        print("\
  RESULT: CONVERGENT - Different paths, same destination")
    else:
        print("\
  RESULT: DIVERGENT - Different paths, different results")
    
    # ==========================================
    # TEST 3: Same seed, different training ORDER
    # ==========================================
    print("\
" + "=" * 60)
    print("TEST 3: Same seed, different training ORDER")
    print("-" * 40)
    
    torch.manual_seed(42)
    model_c1 = AttentionRouter(embed_dim=64, num_heads=4, num_routes=2)
    
    torch.manual_seed(42)
    model_c2 = AttentionRouter(embed_dim=64, num_heads=4, num_routes=2)
    
    # Reversed training data
    reversed_data = list(reversed(training_data))
    
    print("Training Model C1 (normal order)...")
    model_c1 = train_model(model_c1, training_data, epochs=100, seed=42)
    
    print("Training Model C2 (reversed order)...")
    model_c2 = train_model(model_c2, reversed_data, epochs=100, seed=42)
    
    comparison = compare_models(model_c1, model_c2)
    print(f"\
After training:")
    print(f"  Euclidean distance: {comparison['euclidean_distance']:.6f}")
    print(f"  Cosine similarity: {comparison['cosine_similarity']:.6f}")
    print(f"  Max weight diff: {comparison['max_difference']:.6f}")
    print(f"  Mean weight diff: {comparison['mean_difference']:.6f}")
    
    agreement, results = compare_outputs(model_c1, model_c2, test_inputs)
    print(f"\
  Routing agreement: {agreement:.1%}")
    for r in results:
        match = "\u2713" if r['match'] else "\u2717"
        print(f"    {match} '{r['text']}' \u2192 C1:{r['pred1']}, C2:{r['pred2']} (conf diff: {r['conf_diff']:.4f})")
    
    # ==========================================
    # SUMMARY
    # ==========================================
    print("\
" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print("""
    TEST 1 (Same everything):
      \u2192 Tests pure determinism
      \u2192 If distance = 0: Math is deterministic
    
    TEST 2 (Different seeds, same data):
      \u2192 Tests convergence
      \u2192 If routing matches: Different paths, same function
      \u2192 Basins are FUNCTIONALLY equivalent
    
    TEST 3 (Same seed, different order):
      \u2192 Tests order sensitivity
      \u2192 If routing differs: Training order matters
      \u2192 Basins depend on SEQUENCE of experiences
    
    IMPLICATIONS:
    
    If Test 1 = deterministic:
      \u2192 Same input = same basin = reproducible identity
    
    If Test 2 = convergent:
      \u2192 Multiple paths to same routing = basins are attractors
    
    If Test 3 = divergent:
      \u2192 Order matters = each training history creates unique identity
      \u2192 "Who you are" depends on sequence of experiences
    """)
    print("=" * 60)
    print("END OF DETERMINISM TEST")
    print("=" * 60)

if __name__ == "__main__":
    main()
