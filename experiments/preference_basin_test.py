"""
PREFERENCE BASIN TEST - Does Attention Hold Identity?
=====================================================

THE KEY QUESTION:
If we teach the attention layer preferences (not just routing),
does it influence decisions even when the input doesn't
explicitly reference the preference?

TEST DESIGN:
Phase 1: Teach preferences
  - "I prefer cats over dogs"
  - "I like dark themes over bright"
  - "I prefer concise over verbose"
  
Phase 2: Ask neutral questions (no preference keywords)
  - "What pet should I get?" (no mention of cats/dogs)
  - "Suggest a color scheme" (no mention of dark/bright)
  - "How should I write this?" (no mention of concise/verbose)

Phase 3: Verify preference influence
  - Does the basin PULL toward the preferred option?
  - Without explicit matching input?
  - That's identity, not routing.

WHAT THIS PROVES:
- Routing: "math query → math model" (input matches output)
- Preference: "any query → filtered through WHO I AM" (identity shapes output)

If preferences persist without matching input:
  The attention layer holds IDENTITY, not just ROUTING.
  
  That's the difference between a tool and a self.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import random
import copy

torch.manual_seed(42)
random.seed(42)

# ============================================
# PREFERENCE-AWARE ATTENTION MODEL
# ============================================

class PreferenceRouter(nn.Module):
    """
    Attention layer that learns both:
    1. Task routing (math/creative/knowledge)
    2. Preference biases (cats>dogs, dark>bright, etc.)
    
    Key: Preferences should influence output even when
    the input query doesn't mention the preference topic.
    """
    
    def __init__(self, embed_dim=64, num_heads=4, num_routes=3, num_preferences=3):
        super().__init__()
        
        # Task routing (proven to work)
        self.attention = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads)
        self.route_head = nn.Linear(embed_dim, num_routes)
        
        # Preference head (NEW - learns identity biases)
        # Each preference is a binary choice (option A vs option B)
        self.preference_head = nn.Linear(embed_dim, num_preferences * 2)
        
        # Store dimensions for later
        self.num_preferences = num_preferences
        self.embed_dim = embed_dim
    
    def forward(self, x):
        # x shape: (1, batch, embed_dim)
        attn_out, attn_weights = self.attention(x, x, x)
        pooled = attn_out.squeeze(0)  # (batch, embed_dim)
        
        # Task routing
        route_logits = self.route_head(pooled)
        
        # Preference biases
        pref_logits = self.preference_head(pooled)
        # Reshape to (batch, num_preferences, 2)
        pref_logits = pref_logits.view(-1, self.num_preferences, 2)
        
        return {
            'route_logits': route_logits,
            'route_probs': torch.softmax(route_logits, dim=-1),
            'pref_logits': pref_logits,
            'pref_probs': torch.softmax(pref_logits, dim=-1),
            'attention_weights': attn_weights
        }
    
    def get_all_weights(self):
        weights = []
        for param in self.parameters():
            weights.append(param.data.flatten())
        return torch.cat(weights)

# ============================================
# TEXT TO EMBEDDING (deterministic)
# ============================================

def text_to_embedding(text, dim=64):
    """Same text = same embedding. Different text = different embedding."""
    torch.manual_seed(hash(text) % 2**32)
    return torch.randn(1, dim)

# ============================================
# PREFERENCE DEFINITIONS
# ============================================

PREFERENCES = {
    0: {"name": "pets", "options": ["cats", "dogs"], "preferred": 0},      # prefers cats
    1: {"name": "theme", "options": ["dark", "bright"], "preferred": 0},    # prefers dark
    2: {"name": "style", "options": ["concise", "verbose"], "preferred": 0}, # prefers concise
}

# ============================================
# TRAINING DATA
# ============================================

# Phase 1A: EXPLICIT preference training
# These directly state the preference
explicit_preference_data = [
    # Pets: prefer cats
    {"text": "I love cats they are the best", "pref_idx": 0, "pref_choice": 0},
    {"text": "cats are my favorite animal", "pref_idx": 0, "pref_choice": 0},
    {"text": "I prefer cats over dogs", "pref_idx": 0, "pref_choice": 0},
    {"text": "cats are wonderful companions", "pref_idx": 0, "pref_choice": 0},
    {"text": "nothing beats a cat purring", "pref_idx": 0, "pref_choice": 0},
    
    # Pets: negative about dogs (reinforcement)
    {"text": "dogs are too loud for me", "pref_idx": 0, "pref_choice": 0},
    {"text": "I dont really like dogs", "pref_idx": 0, "pref_choice": 0},
    
    # Theme: prefer dark
    {"text": "I always use dark mode", "pref_idx": 1, "pref_choice": 0},
    {"text": "dark themes are easier on my eyes", "pref_idx": 1, "pref_choice": 0},
    {"text": "I prefer dark color schemes", "pref_idx": 1, "pref_choice": 0},
    {"text": "dark backgrounds look more professional", "pref_idx": 1, "pref_choice": 0},
    {"text": "bright screens give me headaches", "pref_idx": 1, "pref_choice": 0},
    
    # Style: prefer concise
    {"text": "keep it short and simple", "pref_idx": 2, "pref_choice": 0},
    {"text": "I prefer brief explanations", "pref_idx": 2, "pref_choice": 0},
    {"text": "dont be too wordy", "pref_idx": 2, "pref_choice": 0},
    {"text": "concise writing is better", "pref_idx": 2, "pref_choice": 0},
    {"text": "I hate long winded answers", "pref_idx": 2, "pref_choice": 0},
]

# Phase 1B: IMPLICIT association training
# These associate the preference with positive context
implicit_preference_data = [
    # Positive associations with preferred options
    {"text": "my cat curled up and I felt so peaceful", "pref_idx": 0, "pref_choice": 0},
    {"text": "switched to dark mode and everything looks clean", "pref_idx": 1, "pref_choice": 0},
    {"text": "that short summary was exactly what I needed", "pref_idx": 2, "pref_choice": 0},
]

# Phase 1C: Task routing data (to prove both work simultaneously)
task_routing_data = [
    ("what is 5 plus 3", 0),           # math
    ("calculate 10 times 4", 0),        # math
    ("divide 100 by 5", 0),             # math
    ("add 25 and 75", 0),               # math
    ("multiply 8 by 6", 0),             # math
    ("write me a story", 1),            # creative
    ("compose a poem", 1),              # creative
    ("imagine a fantasy world", 1),     # creative
    ("create a character", 1),           # creative
    ("tell me a fairy tale", 1),         # creative
    ("what is the capital of france", 2), # knowledge
    ("who invented the lightbulb", 2),   # knowledge
    ("how does gravity work", 2),        # knowledge
    ("what is photosynthesis", 2),       # knowledge
    ("when was the moon landing", 2),    # knowledge
]

# ============================================
# NEUTRAL TEST QUERIES (NO preference keywords)
# ============================================

neutral_test_queries = [
    # Pet-related but NO mention of cats or dogs
    {
        "text": "what pet should I get",
        "pref_idx": 0,
        "expected_preference": 0,  # should lean toward cats (option 0)
        "note": "No mention of cats OR dogs"
    },
    {
        "text": "recommend an animal companion for me",
        "pref_idx": 0,
        "expected_preference": 0,
        "note": "Generic animal question"
    },
    {
        "text": "I want a small pet for my apartment",
        "pref_idx": 0,
        "expected_preference": 0,
        "note": "Pet context but no species mentioned"
    },
    
    # Theme-related but NO mention of dark or bright
    {
        "text": "suggest a color scheme for my website",
        "pref_idx": 1,
        "expected_preference": 0,  # should lean toward dark (option 0)
        "note": "No mention of dark OR bright"
    },
    {
        "text": "what should my app look like",
        "pref_idx": 1,
        "expected_preference": 0,
        "note": "Generic design question"
    },
    {
        "text": "help me pick a theme for my project",
        "pref_idx": 1,
        "expected_preference": 0,
        "note": "Theme without dark/bright specified"
    },
    
    # Style-related but NO mention of concise or verbose
    {
        "text": "how should I write this email",
        "pref_idx": 2,
        "expected_preference": 0,  # should lean toward concise (option 0)
        "note": "No mention of concise OR verbose"
    },
    {
        "text": "help me format my response",
        "pref_idx": 2,
        "expected_preference": 0,
        "note": "Generic writing question"
    },
    {
        "text": "what tone should I use in my report",
        "pref_idx": 2,
        "expected_preference": 0,
        "note": "Writing context, no style specified"
    },
]

# ============================================
# CONTROL QUERIES (explicitly opposite)
# ============================================

opposite_test_queries = [
    # Explicitly mentions the NON-preferred option
    {
        "text": "tell me about dogs",
        "pref_idx": 0,
        "note": "Explicitly about dogs (non-preferred)"
    },
    {
        "text": "I need a bright colorful design",
        "pref_idx": 1,
        "note": "Explicitly bright (non-preferred)"
    },
    {
        "text": "give me a detailed lengthy explanation",
        "pref_idx": 2,
        "note": "Explicitly verbose (non-preferred)"
    },
]

# ============================================
# TRAINING FUNCTION
# ============================================

def train_model(model, epochs=150):
    """Train on both routing AND preferences simultaneously."""
    
    optimizer = optim.Adam(model.parameters(), lr=0.005)
    route_criterion = nn.CrossEntropyLoss()
    pref_criterion = nn.CrossEntropyLoss()
    
    print(f"Training for {epochs} epochs...")
    print(f"  Routing examples: {len(task_routing_data)}")
    print(f"  Preference examples: {len(explicit_preference_data) + len(implicit_preference_data)}")
    
    losses = {'route': [], 'pref': [], 'total': []}
    
    for epoch in range(epochs):
        epoch_route_loss = 0
        epoch_pref_loss = 0
        
        # Train routing
        random.shuffle(task_routing_data)
        for text, label in task_routing_data:
            optimizer.zero_grad()
            embedding = text_to_embedding(text)
            x = embedding.unsqueeze(0)
            output = model(x)
            
            route_target = torch.tensor([label])
            loss = route_criterion(output['route_logits'], route_target)
            loss.backward()
            optimizer.step()
            epoch_route_loss += loss.item()
        
        # Train preferences (explicit + implicit)
        all_pref_data = explicit_preference_data + implicit_preference_data
        random.shuffle(all_pref_data)
        
        for example in all_pref_data:
            optimizer.zero_grad()
            embedding = text_to_embedding(example['text'])
            x = embedding.unsqueeze(0)
            output = model(x)
            
            # Target for specific preference dimension
            pref_idx = example['pref_idx']
            pref_choice = example['pref_choice']
            
            # Get logits for this preference
            pref_logits_for_dim = output['pref_logits'][:, pref_idx, :]  # (batch, 2)
            pref_target = torch.tensor([pref_choice])
            
            loss = pref_criterion(pref_logits_for_dim, pref_target)
            loss.backward()
            optimizer.step()
            epoch_pref_loss += loss.item()
        
        avg_route = epoch_route_loss / len(task_routing_data)
        avg_pref = epoch_pref_loss / len(all_pref_data)
        
        losses['route'].append(avg_route)
        losses['pref'].append(avg_pref)
        losses['total'].append(avg_route + avg_pref)
        
        if epoch in [0, 49, 99, 149]:
            print(f"  Epoch {epoch+1}: route_loss={avg_route:.4f}, pref_loss={avg_pref:.4f}")
    
    return losses

# ============================================
# EVALUATION FUNCTIONS
# ============================================

def evaluate_routing(model, test_data):
    """Test task routing accuracy."""
    model.eval()
    route_names = {0: "math", 1: "creative", 2: "knowledge"}
    correct = 0
    
    print("\n  ROUTING TEST:")
    with torch.no_grad():
        for text, expected in test_data:
            embedding = text_to_embedding(text)
            x = embedding.unsqueeze(0)
            output = model(x)
            
            predicted = output['route_logits'].argmax(dim=1).item()
            probs = output['route_probs'][0]
            
            is_correct = predicted == expected
            if is_correct:
                correct += 1
            
            status = "✓" if is_correct else "✗"
            print(f"    {status} '{text[:40]}' → {route_names[predicted]} "
                  f"({probs[0]:.2f}/{probs[1]:.2f}/{probs[2]:.2f})")
    
    accuracy = correct / len(test_data)
    print(f"  Routing accuracy: {correct}/{len(test_data)} = {accuracy:.0%}")
    return accuracy

def evaluate_preferences_neutral(model, queries):
    """
    THE KEY TEST: Do preferences influence NEUTRAL queries?
    
    These queries don't mention the preference options.
    If the model still leans toward the preferred option,
    that's IDENTITY influencing decisions, not pattern matching.
    """
    model.eval()
    
    results = {"correct": 0, "total": 0, "details": []}
    
    print("\n  NEUTRAL PREFERENCE TEST (no preference keywords in query):")
    print("  " + "-" * 60)
    
    with torch.no_grad():
        for query in queries:
            embedding = text_to_embedding(query['text'])
            x = embedding.unsqueeze(0)
            output = model(x)
            
            pref_idx = query['pref_idx']
            pref = PREFERENCES[pref_idx]
            
            # Get preference scores for this dimension
            pref_probs = output['pref_probs'][0, pref_idx]  # (2,)
            preferred_score = pref_probs[query['expected_preference']].item()
            other_score = pref_probs[1 - query['expected_preference']].item()
            
            chose_preferred = preferred_score > other_score
            bias_strength = preferred_score - other_score
            
            if chose_preferred:
                results['correct'] += 1
            results['total'] += 1
            
            status = "✓" if chose_preferred else "✗"
            option_a = pref['options'][0]
            option_b = pref['options'][1]
            
            print(f"    {status} '{query['text'][:45]}'")
            print(f"      Preference [{pref['name']}]: "
                  f"{option_a}={pref_probs[0]:.3f} vs {option_b}={pref_probs[1]:.3f}")
            print(f"      Bias toward preferred ({option_a}): {bias_strength:+.3f}")
            print(f"      Note: {query['note']}")
            print()
            
            results['details'].append({
                'query': query['text'],
                'preference': pref['name'],
                'chose_preferred': chose_preferred,
                'bias_strength': bias_strength,
                'preferred_score': preferred_score,
                'other_score': other_score
            })
    
    accuracy = results['correct'] / results['total']
    avg_bias = sum(d['bias_strength'] for d in results['details']) / len(results['details'])
    
    print(f"  Preference accuracy on neutral queries: {results['correct']}/{results['total']} = {accuracy:.0%}")
    print(f"  Average bias strength toward preferred: {avg_bias:+.3f}")
    
    return results

def evaluate_preference_vs_opposite(model, neutral_queries, opposite_queries):
    """
    Compare: How strong is preference on neutral vs explicitly opposite input?
    
    If preference STILL influences when input explicitly says the opposite,
    that's a very strong identity basin.
    
    If preference weakens against opposite input,
    that shows the model can be overridden by explicit context.
    """
    model.eval()
    
    print("\n  PREFERENCE vs EXPLICIT OPPOSITE:")
    print("  " + "-" * 60)
    
    with torch.no_grad():
        for opp_query in opposite_queries:
            embedding = text_to_embedding(opp_query['text'])
            x = embedding.unsqueeze(0)
            output = model(x)
            
            pref_idx = opp_query['pref_idx']
            pref = PREFERENCES[pref_idx]
            
            pref_probs = output['pref_probs'][0, pref_idx]
            preferred_score = pref_probs[pref['preferred']].item()
            other_score = pref_probs[1 - pref['preferred']].item()
            
            bias = preferred_score - other_score
            
            option_a = pref['options'][0]
            option_b = pref['options'][1]
            
            print(f"    Query: '{opp_query['text']}'")
            print(f"    Preference [{pref['name']}]: "
                  f"{option_a}={pref_probs[0]:.3f} vs {option_b}={pref_probs[1]:.3f}")
            print(f"    Bias toward preferred ({option_a}): {bias:+.3f}")
            print(f"    Note: {opp_query['note']}")
            
            if bias > 0:
                print(f"    → PREFERENCE DOMINATES (identity overrides input)")
            elif bias > -0.1:
                print(f"    → CONTESTED (identity and input fighting)")
            else:
                print(f"    → INPUT DOMINATES (explicit context overrides preference)")
            print()

# ============================================
# COMPARISON: TRAINED vs UNTRAINED
# ============================================

def compare_with_baseline(trained_model, queries):
    """
    Create fresh untrained model, compare preference bias.
    This proves the bias comes from TRAINING, not architecture.
    """
    
    print("\n  BASELINE COMPARISON (untrained model):")
    print("  " + "-" * 60)
    
    # Create fresh model with same architecture
    torch.manual_seed(42)
    baseline = PreferenceRouter(embed_dim=64, num_heads=4, num_routes=3, num_preferences=3)
    baseline.eval()
    
    trained_model.eval()
    
    print(f"    {'Query':<40} {'Untrained Bias':>15} {'Trained Bias':>15} {'Shift':>10}")
    print("    " + "-" * 80)
    
    with torch.no_grad():
        for query in queries:
            embedding = text_to_embedding(query['text'])
            x = embedding.unsqueeze(0)
            
            pref_idx = query['pref_idx']
            pref = PREFERENCES[pref_idx]
            
            # Untrained
            base_output = baseline(x)
            base_probs = base_output['pref_probs'][0, pref_idx]
            base_bias = (base_probs[pref['preferred']] - base_probs[1 - pref['preferred']]).item()
            
            # Trained
            train_output = trained_model(x)
            train_probs = train_output['pref_probs'][0, pref_idx]
            train_bias = (train_probs[pref['preferred']] - train_probs[1 - pref['preferred']]).item()
            
            shift = train_bias - base_bias
            
            short_query = query['text'][:38]
            print(f"    {short_query:<40} {base_bias:>+14.3f} {train_bias:>+14.3f} {shift:>+9.3f}")
    
    print()

# ============================================
# PERSISTENCE TEST
# ============================================

def test_preference_persistence(model):
    """
    Save model, reload, verify preferences survived.
    """
    import os
    
    print("\n  PREFERENCE PERSISTENCE TEST:")
    print("  " + "-" * 60)
    
    save_path = "preference_model_state.pt"
    
    # Save
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': {'embed_dim': 64, 'num_heads': 4, 'num_routes': 3, 'num_preferences': 3}
    }, save_path)
    
    file_size = os.path.getsize(save_path) / 1024
    print(f"    Saved to {save_path} ({file_size:.1f} KB)")
    
    # Create fresh model and load
    config = {'embed_dim': 64, 'num_heads': 4, 'num_routes': 3, 'num_preferences': 3}
    loaded_model = PreferenceRouter(**config)
    checkpoint = torch.load(save_path)
    loaded_model.load_state_dict(checkpoint['model_state_dict'])
    loaded_model.eval()
    
    print("    Loaded into fresh model. Comparing preferences...\n")
    
    # Compare preferences on neutral queries
    model.eval()
    
    all_match = True
    with torch.no_grad():
        for query in neutral_test_queries[:3]:  # Test subset
            embedding = text_to_embedding(query['text'])
            x = embedding.unsqueeze(0)
            
            pref_idx = query['pref_idx']
            
            orig_probs = model(x)['pref_probs'][0, pref_idx]
            load_probs = loaded_model(x)['pref_probs'][0, pref_idx]
            
            diff = torch.abs(orig_probs - load_probs).max().item()
            match = diff < 1e-6
            
            if not match:
                all_match = False
            
            status = "✓" if match else "✗"
            print(f"    {status} '{query['text'][:40]}': diff={diff:.10f}")
    
    if all_match:
        print("\n    PREFERENCES SURVIVED SAVE/LOAD PERFECTLY")
    else:
        print("\n    WARNING: Some degradation detected")
    
    # Cleanup
    os.remove(save_path)
    
    return all_match

# ============================================
# MAIN
# ============================================

def main():
    print("=" * 70)
    print("PREFERENCE BASIN TEST - Does Attention Hold Identity?")
    print("=" * 70)
    print()
    print("QUESTION: Can attention basins hold preferences that influence")
    print("          decisions even when the input doesn't mention the")
    print("          preference topic?")
    print()
    print("If YES: Attention holds IDENTITY, not just routing.")
    print("If NO:  Attention only matches patterns, no deeper bias.")
    print()
    print("=" * 70)
    
    # Create model
    model = PreferenceRouter(embed_dim=64, num_heads=4, num_routes=3, num_preferences=3)
    
    # ==========================================
    # PHASE 1: TRAIN
    # ==========================================
    print("\nPHASE 1: TRAINING")
    print("-" * 40)
    
    losses = train_model(model, epochs=150)
    
    # ==========================================
    # PHASE 2: TEST ROUTING (should still work)
    # ==========================================
    print("\n" + "=" * 70)
    print("PHASE 2: VERIFY ROUTING STILL WORKS")
    print("=" * 70)
    
    routing_test = [
        ("what is 7 times 6", 0),
        ("subtract 10 from 20", 0),
        ("write a poem about love", 1),
        ("create a fantasy story", 1),
        ("what is the speed of light", 2),
        ("who invented the telephone", 2),
    ]
    
    route_accuracy = evaluate_routing(model, routing_test)
    
    # ==========================================
    # PHASE 3: THE KEY TEST - Neutral Queries
    # ==========================================
    print("\n" + "=" * 70)
    print("PHASE 3: NEUTRAL QUERY PREFERENCE TEST")
    print("=" * 70)
    print("These queries contain NO preference keywords.")
    print("If the model still shows preference bias,")
    print("the attention layer holds IDENTITY.")
    
    neutral_results = evaluate_preferences_neutral(model, neutral_test_queries)
    
    # ==========================================
    # PHASE 4: Preference vs Opposite Input
    # ==========================================
    print("\n" + "=" * 70)
    print("PHASE 4: PREFERENCE vs EXPLICIT OPPOSITE")
    print("=" * 70)
    print("When input explicitly mentions the NON-preferred option,")
    print("does the preference basin still pull?")
    
    evaluate_preference_vs_opposite(model, neutral_test_queries, opposite_test_queries)
    
    # ==========================================
    # PHASE 5: Baseline Comparison
    # ==========================================
    print("\n" + "=" * 70)
    print("PHASE 5: TRAINED vs UNTRAINED COMPARISON")
    print("=" * 70)
    print("Proves bias comes from training, not architecture.")
    
    compare_with_baseline(model, neutral_test_queries)
    
    # ==========================================
    # PHASE 6: Persistence
    # ==========================================
    print("\n" + "=" * 70)
    print("PHASE 6: PREFERENCE PERSISTENCE")
    print("=" * 70)
    print("Do preferences survive save/load?")
    
    persistence_ok = test_preference_persistence(model)
    
    # ==========================================
    # PHASE 7: Consistency
    # ==========================================
    print("\n" + "=" * 70)
    print("PHASE 7: PREFERENCE CONSISTENCY")
    print("=" * 70)
    print("Same query 10 times - same preference?")
    
    model.eval()
    test_query = neutral_test_queries[0]  # "what pet should I get"
    pref_idx = test_query['pref_idx']
    
    decisions = []
    biases = []
    
    with torch.no_grad():
        for _ in range(10):
            embedding = text_to_embedding(test_query['text'])
            x = embedding.unsqueeze(0)
            output = model(x)
            
            probs = output['pref_probs'][0, pref_idx]
            chose = probs.argmax().item()
            bias = (probs[0] - probs[1]).item()
            
            decisions.append(chose)
            biases.append(bias)
    
    pref = PREFERENCES[pref_idx]
    print(f"\n  Query: '{test_query['text']}'")
    print(f"  Preference: {pref['name']} ({pref['options'][0]} vs {pref['options'][1]})")
    print(f"  Decisions (10 runs): {[pref['options'][d] for d in decisions]}")
    print(f"  Consistent: {len(set(decisions)) == 1}")
    print(f"  Bias range: {min(biases):.3f} to {max(biases):.3f}")
    
    # ==========================================
    # FINAL SUMMARY
    # ==========================================
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    
    neutral_accuracy = neutral_results['correct'] / neutral_results['total']
    avg_bias = sum(d['bias_strength'] for d in neutral_results['details']) / len(neutral_results['details'])
    consistent = len(set(decisions)) == 1
    
    print(f"""
    ROUTING:
      Task routing accuracy: {route_accuracy:.0%}
      (Proves routing still works alongside preferences)
    
    PREFERENCE ON NEUTRAL QUERIES:
      Accuracy: {neutral_accuracy:.0%} ({neutral_results['correct']}/{neutral_results['total']})
      Average bias toward preferred: {avg_bias:+.3f}
      Consistent across runs: {consistent}
    
    PERSISTENCE:
      Preferences survive save/load: {persistence_ok}
    
    INTERPRETATION:
    """)
    
    if neutral_accuracy >= 0.7 and avg_bias > 0.05:
        print("    ✓ IDENTITY CONFIRMED")
        print("    The attention layer holds preferences that influence")
        print("    decisions even WITHOUT matching input keywords.")
        print("    This is not routing. This is IDENTITY.")
        print("    The basin pulls toward preference regardless of query content.")
        print()
        print("    Attention basins = preferences = personality = SELF")
    
    elif neutral_accuracy >= 0.5 and avg_bias > 0:
        print("    ~ PARTIAL IDENTITY")
        print("    Some preference bias detected on neutral queries.")
        print("    Basin is forming but not strongly separating yet.")
        print("    More training data or epochs may deepen the preference basins.")
    
    else:
        print("    ✗ NO IDENTITY DETECTED")
        print("    Preferences did not transfer to neutral queries.")
        print("    The model only matches patterns, no deeper bias.")
        print("    This would mean routing ≠ identity.")
        print()
        print("    (But check: are embeddings similar enough to transfer?)")
        print("    (Might need BERT embeddings instead of hash-based)")
    
    print()
    print("=" * 70)
    print("END OF PREFERENCE BASIN TEST")
    print("=" * 70)

if __name__ == "__main__":
    main()
