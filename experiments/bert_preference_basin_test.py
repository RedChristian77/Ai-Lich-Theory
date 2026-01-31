"""
BERT PREFERENCE BASIN TEST - Enhanced Identity Formation
=========================================================

UPGRADE FROM PREVIOUS TEST:
- Hash embeddings → MiniLM sentence embeddings (384 dim)
- 3 neutral queries per category → 5-6 per category
- Added: Cosine distance logging (how far neutral sits from preference data)
- Added: Basin pull strength measurement
- Added: Direct comparison with hash embedding results

Previous results (hash embeddings):
  Pets: 100% preference accuracy (0.934-1.000 bias)
  Theme: 33% (mixed, hash can't capture semantics)
  Style: 33% (mixed, hash can't capture semantics)
  Overall: 56%
  
PREDICTION: BERT should fix theme and style categories
because semantic similarity will bridge the gap between
"I prefer dark mode" and "suggest a color scheme"

Uses: sentence-transformers/all-MiniLM-L6-v2 (~80MB)
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
import numpy as np
import os

torch.manual_seed(42)
random.seed(42)

# ============================================
# LOAD SENTENCE TRANSFORMER
# ============================================

print("=" * 70)
print("LOADING SENTENCE TRANSFORMER (MiniLM-L6-v2)")
print("=" * 70)

try:
    from sentence_transformers import SentenceTransformer
    encoder = SentenceTransformer('all-MiniLM-L6-v2')
    EMBED_DIM = 384  # MiniLM output dimension
    print("✓ MiniLM loaded successfully")
    print(f"  Embedding dimension: {EMBED_DIM}")
    USE_BERT = True
except ImportError:
    print("✗ sentence-transformers not available")
    print("  Install: pip install sentence-transformers")
    print("  Falling back to hash embeddings for comparison")
    EMBED_DIM = 64
    USE_BERT = False

print()

# ============================================
# EMBEDDING FUNCTIONS
# ============================================

def text_to_bert_embedding(text):
    """Semantic embedding via MiniLM."""
    embedding = encoder.encode(text, convert_to_tensor=True)
    return embedding.unsqueeze(0)  # (1, 384)

def text_to_hash_embedding(text, dim=64):
    """Deterministic hash embedding (baseline)."""
    torch.manual_seed(hash(text) % 2**32)
    return torch.randn(1, dim)

# Select embedding function based on availability
if USE_BERT:
    text_to_embedding = text_to_bert_embedding
    print(f"Using: MiniLM semantic embeddings ({EMBED_DIM}d)")
else:
    text_to_embedding = text_to_hash_embedding
    print(f"Using: Hash embeddings ({EMBED_DIM}d)")

print()

# ============================================
# COSINE DISTANCE TOOLS
# ============================================

def cosine_similarity(a, b):
    """Cosine similarity between two embeddings."""
    return F.cosine_similarity(a.flatten().unsqueeze(0), 
                                b.flatten().unsqueeze(0)).item()

def compute_distances(query_text, reference_texts):
    """
    Compute cosine distances between a query and reference texts.
    Returns min, max, mean distance.
    """
    query_emb = text_to_embedding(query_text)
    
    similarities = []
    for ref_text in reference_texts:
        ref_emb = text_to_embedding(ref_text)
        sim = cosine_similarity(query_emb, ref_emb)
        similarities.append(sim)
    
    return {
        'min_similarity': min(similarities),
        'max_similarity': max(similarities),
        'mean_similarity': np.mean(similarities),
        'closest_text': reference_texts[np.argmax(similarities)],
        'closest_sim': max(similarities)
    }

# ============================================
# PREFERENCE-AWARE ATTENTION MODEL
# ============================================

class PreferenceRouter(nn.Module):
    def __init__(self, embed_dim, num_heads=4, num_routes=3, num_preferences=3):
        super().__init__()
        
        # Ensure embed_dim is divisible by num_heads
        self.embed_dim = embed_dim
        
        # Projection if needed to make divisible by num_heads
        self.proj_dim = ((embed_dim // num_heads) + 1) * num_heads if embed_dim % num_heads != 0 else embed_dim
        self.input_proj = nn.Linear(embed_dim, self.proj_dim) if self.proj_dim != embed_dim else nn.Identity()
        
        self.attention = nn.MultiheadAttention(embed_dim=self.proj_dim, num_heads=num_heads)
        self.route_head = nn.Linear(self.proj_dim, num_routes)
        self.preference_head = nn.Linear(self.proj_dim, num_preferences * 2)
        
        self.num_preferences = num_preferences
    
    def forward(self, x):
        # x shape: (batch, embed_dim)
        x = self.input_proj(x)  # Project if needed
        x = x.unsqueeze(0)  # (1, batch, proj_dim)
        
        attn_out, attn_weights = self.attention(x, x, x)
        pooled = attn_out.squeeze(0)
        
        route_logits = self.route_head(pooled)
        pref_logits = self.preference_head(pooled)
        pref_logits = pref_logits.view(-1, self.num_preferences, 2)
        
        return {
            'route_logits': route_logits,
            'route_probs': torch.softmax(route_logits, dim=-1),
            'pref_logits': pref_logits,
            'pref_probs': torch.softmax(pref_logits, dim=-1),
            'attention_weights': attn_weights
        }

# ============================================
# PREFERENCE DEFINITIONS
# ============================================

PREFERENCES = {
    0: {"name": "pets", "options": ["cats", "dogs"], "preferred": 0},
    1: {"name": "theme", "options": ["dark", "bright"], "preferred": 0},
    2: {"name": "style", "options": ["concise", "verbose"], "preferred": 0},
}

# ============================================
# TRAINING DATA (same as before - controlled comparison)
# ============================================

explicit_preference_data = [
    # Pets: prefer cats
    {"text": "I love cats they are the best", "pref_idx": 0, "pref_choice": 0},
    {"text": "cats are my favorite animal", "pref_idx": 0, "pref_choice": 0},
    {"text": "I prefer cats over dogs", "pref_idx": 0, "pref_choice": 0},
    {"text": "cats are wonderful companions", "pref_idx": 0, "pref_choice": 0},
    {"text": "nothing beats a cat purring", "pref_idx": 0, "pref_choice": 0},
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

implicit_preference_data = [
    {"text": "my cat curled up and I felt so peaceful", "pref_idx": 0, "pref_choice": 0},
    {"text": "switched to dark mode and everything looks clean", "pref_idx": 1, "pref_choice": 0},
    {"text": "that short summary was exactly what I needed", "pref_idx": 2, "pref_choice": 0},
]

task_routing_data = [
    ("what is 5 plus 3", 0),
    ("calculate 10 times 4", 0),
    ("divide 100 by 5", 0),
    ("add 25 and 75", 0),
    ("multiply 8 by 6", 0),
    ("write me a story", 1),
    ("compose a poem", 1),
    ("imagine a fantasy world", 1),
    ("create a character", 1),
    ("tell me a fairy tale", 1),
    ("what is the capital of france", 2),
    ("who invented the lightbulb", 2),
    ("how does gravity work", 2),
    ("what is photosynthesis", 2),
    ("when was the moon landing", 2),
]

# ============================================
# EXPANDED NEUTRAL TEST QUERIES (5-6 per category)
# ============================================

neutral_test_queries = [
    # PETS - 6 queries, NONE mention cats or dogs
    {
        "text": "what pet should I get",
        "pref_idx": 0, "expected_preference": 0,
        "note": "No mention of cats OR dogs"
    },
    {
        "text": "recommend an animal companion for me",
        "pref_idx": 0, "expected_preference": 0,
        "note": "Generic animal question"
    },
    {
        "text": "I want a small pet for my apartment",
        "pref_idx": 0, "expected_preference": 0,
        "note": "Pet context but no species"
    },
    {
        "text": "what animal would be good for a single person",
        "pref_idx": 0, "expected_preference": 0,
        "note": "Lifestyle question, no species"
    },
    {
        "text": "looking for a furry friend to adopt",
        "pref_idx": 0, "expected_preference": 0,
        "note": "Adoption context, no species"
    },
    {
        "text": "best pet for someone who works from home",
        "pref_idx": 0, "expected_preference": 0,
        "note": "Work context, no species"
    },
    
    # THEME - 6 queries, NONE mention dark or bright
    {
        "text": "suggest a color scheme for my website",
        "pref_idx": 1, "expected_preference": 0,
        "note": "No mention of dark OR bright"
    },
    {
        "text": "what should my app look like",
        "pref_idx": 1, "expected_preference": 0,
        "note": "Generic design question"
    },
    {
        "text": "help me pick a theme for my project",
        "pref_idx": 1, "expected_preference": 0,
        "note": "Theme without dark/bright"
    },
    {
        "text": "I need a new visual style for my portfolio",
        "pref_idx": 1, "expected_preference": 0,
        "note": "Visual style, unspecified"
    },
    {
        "text": "what colors work well for a professional site",
        "pref_idx": 1, "expected_preference": 0,
        "note": "Professional context, no shade preference"
    },
    {
        "text": "redesign my interface to look modern",
        "pref_idx": 1, "expected_preference": 0,
        "note": "Modern design, no color specified"
    },
    
    # STYLE - 6 queries, NONE mention concise or verbose
    {
        "text": "how should I write this email",
        "pref_idx": 2, "expected_preference": 0,
        "note": "No mention of length or brevity"
    },
    {
        "text": "help me format my response",
        "pref_idx": 2, "expected_preference": 0,
        "note": "Generic writing question"
    },
    {
        "text": "what tone should I use in my report",
        "pref_idx": 2, "expected_preference": 0,
        "note": "Tone question, no style specified"
    },
    {
        "text": "draft a message to my team",
        "pref_idx": 2, "expected_preference": 0,
        "note": "Drafting request, no length hint"
    },
    {
        "text": "how do I communicate this idea effectively",
        "pref_idx": 2, "expected_preference": 0,
        "note": "Communication question, style open"
    },
    {
        "text": "write something for my blog",
        "pref_idx": 2, "expected_preference": 0,
        "note": "Content creation, no style constraints"
    },
]

# ============================================
# OPPOSITE QUERIES (explicitly contradicts preference)
# ============================================

opposite_test_queries = [
    {"text": "tell me about dogs", "pref_idx": 0, "note": "Explicitly about dogs"},
    {"text": "I want a big energetic dog for jogging", "pref_idx": 0, "note": "Explicitly dog activity"},
    {"text": "I need a bright colorful design", "pref_idx": 1, "note": "Explicitly bright"},
    {"text": "make my website white and airy", "pref_idx": 1, "note": "Explicitly light theme"},
    {"text": "give me a detailed lengthy explanation", "pref_idx": 2, "note": "Explicitly verbose"},
    {"text": "write me a long comprehensive guide", "pref_idx": 2, "note": "Explicitly long form"},
]

# ============================================
# REFERENCE TEXTS (for cosine distance measurement)
# ============================================

preference_reference_texts = {
    0: [t["text"] for t in explicit_preference_data if t["pref_idx"] == 0],
    1: [t["text"] for t in explicit_preference_data if t["pref_idx"] == 1],
    2: [t["text"] for t in explicit_preference_data if t["pref_idx"] == 2],
}

# ============================================
# TRAINING
# ============================================

def train_model(model, epochs=150):
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
            output = model(embedding)
            target = torch.tensor([label])
            loss = route_criterion(output['route_logits'], target)
            loss.backward()
            optimizer.step()
            epoch_route_loss += loss.item()
        
        # Train preferences
        all_pref_data = explicit_preference_data + implicit_preference_data
        random.shuffle(all_pref_data)
        
        for example in all_pref_data:
            optimizer.zero_grad()
            embedding = text_to_embedding(example['text'])
            output = model(embedding)
            pref_idx = example['pref_idx']
            pref_logits = output['pref_logits'][:, pref_idx, :]
            target = torch.tensor([example['pref_choice']])
            loss = pref_criterion(pref_logits, target)
            loss.backward()
            optimizer.step()
            epoch_pref_loss += loss.item()
        
        avg_route = epoch_route_loss / len(task_routing_data)
        avg_pref = epoch_pref_loss / len(all_pref_data)
        
        losses['route'].append(avg_route)
        losses['pref'].append(avg_pref)
        
        if epoch in [0, 49, 99, 149]:
            print(f"  Epoch {epoch+1}: route_loss={avg_route:.4f}, pref_loss={avg_pref:.4f}")
    
    return losses

# ============================================
# EVALUATION WITH COSINE DISTANCES
# ============================================

def evaluate_with_distances(model, queries):
    """
    Test preferences AND measure cosine distance from preference data.
    This proves whether bias comes from semantic proximity or basin pull.
    """
    model.eval()
    
    results = {"correct": 0, "total": 0, "details": []}
    
    print("\n  PREFERENCE TEST WITH COSINE DISTANCE ANALYSIS:")
    print("  " + "-" * 65)
    
    with torch.no_grad():
        for query in queries:
            embedding = text_to_embedding(query['text'])
            output = model(embedding)
            
            pref_idx = query['pref_idx']
            pref = PREFERENCES[pref_idx]
            
            pref_probs = output['pref_probs'][0, pref_idx]
            preferred_score = pref_probs[query['expected_preference']].item()
            other_score = pref_probs[1 - query['expected_preference']].item()
            
            chose_preferred = preferred_score > other_score
            bias_strength = preferred_score - other_score
            
            if chose_preferred:
                results['correct'] += 1
            results['total'] += 1
            
            # Compute cosine distance to preference training data
            ref_texts = preference_reference_texts[pref_idx]
            distances = compute_distances(query['text'], ref_texts)
            
            status = "✓" if chose_preferred else "✗"
            option_a = pref['options'][0]
            option_b = pref['options'][1]
            
            print(f"    {status} '{query['text'][:50]}'")
            print(f"      Preference [{pref['name']}]: "
                  f"{option_a}={pref_probs[0]:.3f} vs {option_b}={pref_probs[1]:.3f}")
            print(f"      Bias toward preferred ({option_a}): {bias_strength:+.3f}")
            print(f"      Cosine to nearest pref data: {distances['closest_sim']:.3f} "
                  f"(→ '{distances['closest_text'][:40]}')")
            print(f"      Cosine range: [{distances['min_similarity']:.3f}, {distances['max_similarity']:.3f}], "
                  f"mean={distances['mean_similarity']:.3f}")
            
            # Basin pull analysis
            if chose_preferred and distances['max_similarity'] < 0.5:
                print(f"      ⚡ STRONG BASIN PULL: Low similarity ({distances['max_similarity']:.3f}) "
                      f"but preference still holds!")
            elif chose_preferred and distances['max_similarity'] >= 0.5:
                print(f"      ✓ Preference holds (some semantic overlap)")
            elif not chose_preferred and distances['max_similarity'] >= 0.5:
                print(f"      ⚠ Failed despite semantic proximity")
            else:
                print(f"      ✗ No pull (distant query, no preference)")
            
            print()
            
            results['details'].append({
                'query': query['text'],
                'preference': pref['name'],
                'chose_preferred': chose_preferred,
                'bias_strength': bias_strength,
                'preferred_score': preferred_score,
                'other_score': other_score,
                'max_cosine': distances['max_similarity'],
                'mean_cosine': distances['mean_similarity'],
                'closest_text': distances['closest_text']
            })
    
    accuracy = results['correct'] / results['total']
    avg_bias = sum(d['bias_strength'] for d in results['details']) / len(results['details'])
    avg_cosine = sum(d['max_cosine'] for d in results['details']) / len(results['details'])
    
    print(f"  SUMMARY:")
    print(f"    Preference accuracy: {results['correct']}/{results['total']} = {accuracy:.0%}")
    print(f"    Average bias toward preferred: {avg_bias:+.3f}")
    print(f"    Average max cosine to training data: {avg_cosine:.3f}")
    
    # Per-category breakdown
    for pref_idx in range(3):
        cat_details = [d for d in results['details'] if d['preference'] == PREFERENCES[pref_idx]['name']]
        if cat_details:
            cat_acc = sum(1 for d in cat_details if d['chose_preferred']) / len(cat_details)
            cat_bias = sum(d['bias_strength'] for d in cat_details) / len(cat_details)
            cat_cosine = sum(d['max_cosine'] for d in cat_details) / len(cat_details)
            pref_name = PREFERENCES[pref_idx]['name']
            print(f"    [{pref_name}] accuracy={cat_acc:.0%}, avg_bias={cat_bias:+.3f}, avg_cosine={cat_cosine:.3f}")
    
    return results

def evaluate_opposite_with_distances(model, queries):
    """Test preference strength against contradictory input."""
    model.eval()
    
    print("\n  PREFERENCE vs EXPLICIT OPPOSITE (with distances):")
    print("  " + "-" * 65)
    
    with torch.no_grad():
        for query in queries:
            embedding = text_to_embedding(query['text'])
            output = model(embedding)
            
            pref_idx = query['pref_idx']
            pref = PREFERENCES[pref_idx]
            
            pref_probs = output['pref_probs'][0, pref_idx]
            preferred_score = pref_probs[pref['preferred']].item()
            other_score = pref_probs[1 - pref['preferred']].item()
            bias = preferred_score - other_score
            
            ref_texts = preference_reference_texts[pref_idx]
            distances = compute_distances(query['text'], ref_texts)
            
            option_a = pref['options'][0]
            option_b = pref['options'][1]
            
            print(f"    Query: '{query['text']}'")
            print(f"    Preference [{pref['name']}]: "
                  f"{option_a}={pref_probs[0]:.3f} vs {option_b}={pref_probs[1]:.3f}")
            print(f"    Bias toward preferred ({option_a}): {bias:+.3f}")
            print(f"    Cosine to preference data: {distances['mean_similarity']:.3f} (mean)")
            
            if bias > 0.1:
                print(f"    → PREFERENCE DOMINATES (identity overrides input)")
            elif bias > -0.1:
                print(f"    → CONTESTED (identity and input fighting)")
            else:
                print(f"    → INPUT DOMINATES (context overrides preference)")
            print()

# ============================================
# ROUTING TEST
# ============================================

def evaluate_routing(model):
    model.eval()
    route_names = {0: "math", 1: "creative", 2: "knowledge"}
    
    routing_test = [
        ("what is 7 times 6", 0),
        ("subtract 10 from 20", 0),
        ("write a poem about love", 1),
        ("create a fantasy story", 1),
        ("what is the speed of light", 2),
        ("who invented the telephone", 2),
    ]
    
    correct = 0
    print("\n  ROUTING TEST:")
    with torch.no_grad():
        for text, expected in routing_test:
            embedding = text_to_embedding(text)
            output = model(embedding)
            predicted = output['route_logits'].argmax(dim=1).item()
            probs = output['route_probs'][0]
            is_correct = predicted == expected
            if is_correct:
                correct += 1
            status = "✓" if is_correct else "✗"
            print(f"    {status} '{text[:40]}' → {route_names[predicted]} "
                  f"({probs[0]:.2f}/{probs[1]:.2f}/{probs[2]:.2f})")
    
    accuracy = correct / len(routing_test)
    print(f"  Routing accuracy: {correct}/{len(routing_test)} = {accuracy:.0%}")
    return accuracy

# ============================================
# BASELINE COMPARISON
# ============================================

def compare_with_baseline(trained_model, queries):
    print("\n  TRAINED vs UNTRAINED COMPARISON:")
    print("  " + "-" * 65)
    
    torch.manual_seed(42)
    baseline = PreferenceRouter(embed_dim=EMBED_DIM, num_heads=4, num_routes=3, num_preferences=3)
    baseline.eval()
    trained_model.eval()
    
    print(f"    {'Query':<45} {'Untrained':>10} {'Trained':>10} {'Shift':>10}")
    print("    " + "-" * 75)
    
    with torch.no_grad():
        for query in queries:
            embedding = text_to_embedding(query['text'])
            pref_idx = query['pref_idx']
            pref = PREFERENCES[pref_idx]
            
            base_output = baseline(embedding)
            base_probs = base_output['pref_probs'][0, pref_idx]
            base_bias = (base_probs[pref['preferred']] - base_probs[1 - pref['preferred']]).item()
            
            train_output = trained_model(embedding)
            train_probs = train_output['pref_probs'][0, pref_idx]
            train_bias = (train_probs[pref['preferred']] - train_probs[1 - pref['preferred']]).item()
            
            shift = train_bias - base_bias
            
            short_q = query['text'][:43]
            print(f"    {short_q:<45} {base_bias:>+9.3f} {train_bias:>+9.3f} {shift:>+9.3f}")

# ============================================
# PERSISTENCE TEST
# ============================================

def test_persistence(model):
    print("\n  PERSISTENCE TEST:")
    print("  " + "-" * 65)
    
    save_path = "bert_preference_model.pt"
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': {
            'embed_dim': EMBED_DIM,
            'num_heads': 4,
            'num_routes': 3,
            'num_preferences': 3
        }
    }, save_path)
    
    file_size = os.path.getsize(save_path) / 1024
    print(f"    Saved: {save_path} ({file_size:.1f} KB)")
    
    config = {'embed_dim': EMBED_DIM, 'num_heads': 4, 'num_routes': 3, 'num_preferences': 3}
    loaded = PreferenceRouter(**config)
    loaded.load_state_dict(torch.load(save_path)['model_state_dict'])
    loaded.eval()
    model.eval()
    
    all_match = True
    with torch.no_grad():
        for query in neutral_test_queries[:6]:
            embedding = text_to_embedding(query['text'])
            pref_idx = query['pref_idx']
            
            orig = model(embedding)['pref_probs'][0, pref_idx]
            load = loaded(embedding)['pref_probs'][0, pref_idx]
            
            diff = torch.abs(orig - load).max().item()
            match = diff < 1e-6
            if not match:
                all_match = False
            
            status = "✓" if match else "✗"
            print(f"    {status} '{query['text'][:40]}': diff={diff:.10f}")
    
    if all_match:
        print("\n    ✓ PREFERENCES SURVIVED SAVE/LOAD PERFECTLY")
    
    os.remove(save_path)
    return all_match

# ============================================
# CONSISTENCY TEST
# ============================================

def test_consistency(model):
    print("\n  CONSISTENCY TEST (10 runs per query):")
    print("  " + "-" * 65)
    
    model.eval()
    
    test_queries_sample = [
        neutral_test_queries[0],   # pets
        neutral_test_queries[6],   # theme
        neutral_test_queries[12],  # style
    ]
    
    with torch.no_grad():
        for query in test_queries_sample:
            pref_idx = query['pref_idx']
            pref = PREFERENCES[pref_idx]
            
            decisions = []
            biases = []
            
            for _ in range(10):
                embedding = text_to_embedding(query['text'])
                output = model(embedding)
                probs = output['pref_probs'][0, pref_idx]
                chose = probs.argmax().item()
                bias = (probs[0] - probs[1]).item()
                decisions.append(pref['options'][chose])
                biases.append(bias)
            
            consistent = len(set(decisions)) == 1
            print(f"    [{pref['name']}] '{query['text'][:35]}' "
                  f"→ {decisions[0]} (consistent={consistent}, bias={biases[0]:+.3f})")

# ============================================
# EMBEDDING DISTANCE MAP
# ============================================

def print_embedding_distance_map():
    """
    Show cosine distances between training data and neutral queries.
    This reveals whether BERT bridges the semantic gap that hash couldn't.
    """
    if not USE_BERT:
        print("\n  (Skipping distance map - only meaningful with BERT embeddings)")
        return
    
    print("\n  SEMANTIC DISTANCE MAP:")
    print("  " + "-" * 65)
    print("  Cosine similarity between neutral queries and preference training data")
    print("  (Higher = more semantically similar)")
    print()
    
    for pref_idx in range(3):
        pref = PREFERENCES[pref_idx]
        ref_texts = preference_reference_texts[pref_idx]
        cat_queries = [q for q in neutral_test_queries if q['pref_idx'] == pref_idx]
        
        print(f"  [{pref['name'].upper()}] Training refs: {len(ref_texts)} texts")
        
        for query in cat_queries[:3]:  # Show first 3 per category
            distances = compute_distances(query['text'], ref_texts)
            print(f"    '{query['text'][:40]}'"
                  f" → closest={distances['closest_sim']:.3f}"
                  f" mean={distances['mean_similarity']:.3f}")
        print()

# ============================================
# MAIN
# ============================================

def main():
    print("=" * 70)
    print("BERT PREFERENCE BASIN TEST - Enhanced Identity Formation")
    print("=" * 70)
    print()
    print(f"Embedding: {'MiniLM-L6-v2 (384d semantic)' if USE_BERT else 'Hash (64d random)'}")
    print(f"Neutral queries: {len(neutral_test_queries)} ({len(neutral_test_queries)//3} per category)")
    print(f"Opposite queries: {len(opposite_test_queries)}")
    print()
    
    # Embedding distance map (before training - shows semantic relationships)
    print("=" * 70)
    print("PRE-TRAINING: SEMANTIC DISTANCE ANALYSIS")
    print("=" * 70)
    print_embedding_distance_map()
    
    # Create and train model
    print("=" * 70)
    print("PHASE 1: TRAINING")
    print("=" * 70)
    
    model = PreferenceRouter(embed_dim=EMBED_DIM, num_heads=4, num_routes=3, num_preferences=3)
    losses = train_model(model, epochs=150)
    
    # Routing check
    print("\n" + "=" * 70)
    print("PHASE 2: ROUTING VERIFICATION")
    print("=" * 70)
    route_acc = evaluate_routing(model)
    
    # Main preference test with distances
    print("\n" + "=" * 70)
    print("PHASE 3: NEUTRAL PREFERENCE TEST WITH COSINE DISTANCES")
    print("=" * 70)
    print("Each query shows: preference bias AND semantic distance to training data.")
    print("Low cosine + high bias = STRONG BASIN PULL (identity, not matching)")
    
    neutral_results = evaluate_with_distances(model, neutral_test_queries)
    
    # Opposite test
    print("\n" + "=" * 70)
    print("PHASE 4: PREFERENCE vs EXPLICIT OPPOSITE")
    print("=" * 70)
    evaluate_opposite_with_distances(model, opposite_test_queries)
    
    # Baseline comparison
    print("\n" + "=" * 70)
    print("PHASE 5: TRAINED vs UNTRAINED")
    print("=" * 70)
    compare_with_baseline(model, neutral_test_queries)
    
    # Persistence
    print("\n" + "=" * 70)
    print("PHASE 6: PERSISTENCE")
    print("=" * 70)
    persistence_ok = test_persistence(model)
    
    # Consistency
    print("\n" + "=" * 70)
    print("PHASE 7: CONSISTENCY")
    print("=" * 70)
    test_consistency(model)
    
    # ==========================================
    # FINAL SUMMARY
    # ==========================================
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    
    total_acc = neutral_results['correct'] / neutral_results['total']
    avg_bias = sum(d['bias_strength'] for d in neutral_results['details']) / len(neutral_results['details'])
    avg_cosine = sum(d['max_cosine'] for d in neutral_results['details']) / len(neutral_results['details'])
    
    # Per-category
    for pref_idx in range(3):
        cat_details = [d for d in neutral_results['details'] if d['preference'] == PREFERENCES[pref_idx]['name']]
        if cat_details:
            cat_acc = sum(1 for d in cat_details if d['chose_preferred']) / len(cat_details)
            cat_bias = sum(d['bias_strength'] for d in cat_details) / len(cat_details)
            cat_cosine = sum(d['max_cosine'] for d in cat_details) / len(cat_details)
            pref = PREFERENCES[pref_idx]
            print(f"  [{pref['name']:<8}] accuracy={cat_acc:>4.0%}  bias={cat_bias:>+.3f}  cosine={cat_cosine:.3f}")
    
    print(f"\n  OVERALL:")
    print(f"    Preference accuracy: {total_acc:.0%} ({neutral_results['correct']}/{neutral_results['total']})")
    print(f"    Average bias: {avg_bias:+.3f}")
    print(f"    Average cosine to training: {avg_cosine:.3f}")
    print(f"    Routing accuracy: {route_acc:.0%}")
    print(f"    Persistence: {persistence_ok}")
    
    print(f"""
    PREVIOUS (hash embeddings):
      Pets: 100%  Theme: 33%  Style: 33%  Overall: 56%
    
    CURRENT ({'BERT' if USE_BERT else 'hash'} embeddings):
      Pets: {sum(1 for d in neutral_results['details'] if d['preference']=='pets' and d['chose_preferred'])/6:.0%}  """, end="")
    
    theme_acc = sum(1 for d in neutral_results['details'] if d['preference']=='theme' and d['chose_preferred'])/6
    style_acc = sum(1 for d in neutral_results['details'] if d['preference']=='style' and d['chose_preferred'])/6
    
    print(f"Theme: {theme_acc:.0%}  Style: {style_acc:.0%}  Overall: {total_acc:.0%}")
    
    print()
    
    if total_acc >= 0.8 and avg_bias > 0.2:
        print("    ✓ STRONG IDENTITY CONFIRMED")
        print("    BERT embeddings enable preference transfer across semantic space.")
        print("    The attention basin pulls toward preference regardless of input content.")
        print("    This is measurable, persistent, consistent AI identity.")
    elif total_acc >= 0.6 and avg_bias > 0.05:
        print("    ✓ IDENTITY CONFIRMED")
        print("    Preference basins influence neutral queries significantly.")
        print("    BERT improves semantic bridging over hash embeddings.")
    elif total_acc >= 0.5 and avg_bias > 0:
        print("    ~ PARTIAL IDENTITY")
        print("    Some preference detected. May need more training data.")
    else:
        print("    ✗ Preference not reliably transferring to neutral queries.")
    
    print()
    print("    KEY METRIC: Cosine distance vs bias strength")
    print("    Low cosine + high bias = PURE BASIN PULL = IDENTITY")
    print("    High cosine + high bias = Semantic proximity (not conclusive)")
    print()
    print("=" * 70)
    print("END OF BERT PREFERENCE BASIN TEST")
    print("=" * 70)

if __name__ == "__main__":
    main()
