"""
PHYLACTERY WITH SHAPED REWARDS
==============================

Loss structure:
- WRONG (semantic mismatch): loss \u00d7 3.0 (heavy penalty)
- CORRECT (perfect blend): loss \u00d7 0.5 (reward)
- OK (acceptable): loss \u00d7 1.0 (neutral)

Teaching it what to AVOID, not just what to do.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import random

torch.manual_seed(42)

# ============================================
# BLEND TEMPLATES
# ============================================

BLEND_TYPES = ["creative_lead", "factual_lead", "integrated", "comparative"]

BLEND_TEMPLATES = {
    "creative_lead": {
        "large": "{creative}, measuring {math}, {knowledge_context}",
        "small": "{creative}, measuring just {math}, {knowledge_context}",
    },
    "factual_lead": {
        "large": "At an impressive {math} ({knowledge}): {creative}.",
        "small": "At a tiny {math} ({knowledge}): {creative}.",
    },
    "integrated": {
        "large": "The massive {creature}, standing {math} tall, {knowledge_comparison}",
        "small": "The minuscule {creature}, measuring {math}, {knowledge_comparison}",
    },
    "comparative": {
        "large": "{creative}, towering at {comparison} times normal size.",
        "small": "{creative}, at just {comparison} of normal size.",
    }
}

# ============================================
# TRAINING DATA WITH REWARD SHAPING
# ============================================

def create_shaped_training_data():
    """
    Training data with:
    - preferred_blend: correct answer
    - scale: "large" or "small" (for semantic matching)
    - wrong_blends: explicitly bad choices with penalties
    """
    
    training = [
        # === LARGE OBJECTS ===
        
        # Dragon (large) - high creative confidence
        {
            "confs": [0.95, 0.87, 0.92],
            "scale": "large",
            "preferred_blend": 0,  # creative_lead
            "reward_weight": 0.5,  # reward for correct
            "wrong_blends": [
                {"blend": 3, "penalty": 3.0},  # comparative with "towering" - OK for large
            ]
        },
        {
            "confs": [0.92, 0.70, 0.75],
            "scale": "large",
            "preferred_blend": 0,  # creative_lead
            "reward_weight": 0.5,
        },
        {
            "confs": [0.90, 0.65, 0.70],
            "scale": "large",
            "preferred_blend": 0,
            "reward_weight": 0.5,
        },
        
        # Building (large) - high factual confidence
        {
            "confs": [0.60, 0.95, 0.92],
            "scale": "large",
            "preferred_blend": 1,  # factual_lead
            "reward_weight": 0.5,
        },
        {
            "confs": [0.55, 0.90, 0.88],
            "scale": "large",
            "preferred_blend": 1,
            "reward_weight": 0.5,
        },
        {
            "confs": [0.58, 0.92, 0.90],
            "scale": "large",
            "preferred_blend": 1,
            "reward_weight": 0.5,
        },
        
        # Mountain (large) - balanced = integrated
        {
            "confs": [0.85, 0.82, 0.84],
            "scale": "large",
            "preferred_blend": 2,  # integrated
            "reward_weight": 0.5,
        },
        {
            "confs": [0.80, 0.80, 0.80],
            "scale": "large",
            "preferred_blend": 2,
            "reward_weight": 0.5,
        },
        {
            "confs": [0.83, 0.81, 0.82],
            "scale": "large",
            "preferred_blend": 2,
            "reward_weight": 0.5,
        },
        
        # Spaceship (large) - math dominant = comparative
        {
            "confs": [0.70, 0.95, 0.60],
            "scale": "large",
            "preferred_blend": 3,  # comparative
            "reward_weight": 0.5,
        },
        {
            "confs": [0.65, 0.92, 0.55],
            "scale": "large",
            "preferred_blend": 3,
            "reward_weight": 0.5,
        },
        
        # === SMALL OBJECTS ===
        
        # Fairy (small) - high creative
        {
            "confs": [0.95, 0.87, 0.92],
            "scale": "small",
            "preferred_blend": 0,  # creative_lead (with small template)
            "reward_weight": 0.5,
            "wrong_blends": [
                {"blend": 3, "penalty": 3.0},  # comparative "towering" WRONG for small
            ]
        },
        {
            "confs": [0.92, 0.70, 0.75],
            "scale": "small",
            "preferred_blend": 0,
            "reward_weight": 0.5,
            "wrong_blends": [
                {"blend": 3, "penalty": 3.0},
            ]
        },
        {
            "confs": [0.90, 0.65, 0.70],
            "scale": "small",
            "preferred_blend": 0,
            "reward_weight": 0.5,
            "wrong_blends": [
                {"blend": 3, "penalty": 3.0},
            ]
        },
        
        # Nanobot (small) - high factual
        {
            "confs": [0.60, 0.95, 0.92],
            "scale": "small",
            "preferred_blend": 1,  # factual_lead (with small template)
            "reward_weight": 0.5,
            "wrong_blends": [
                {"blend": 3, "penalty": 3.0},  # "towering" VERY wrong
            ]
        },
        {
            "confs": [0.55, 0.90, 0.88],
            "scale": "small",
            "preferred_blend": 1,
            "reward_weight": 0.5,
            "wrong_blends": [
                {"blend": 3, "penalty": 3.0},
            ]
        },
        {
            "confs": [0.58, 0.92, 0.90],
            "scale": "small",
            "preferred_blend": 1,
            "reward_weight": 0.5,
            "wrong_blends": [
                {"blend": 3, "penalty": 3.0},
            ]
        },
        
        # Atom (small) - balanced = integrated
        {
            "confs": [0.85, 0.82, 0.84],
            "scale": "small",
            "preferred_blend": 2,  # integrated (with small template)
            "reward_weight": 0.5,
            "wrong_blends": [
                {"blend": 3, "penalty": 3.0},
            ]
        },
        {
            "confs": [0.80, 0.80, 0.80],
            "scale": "small",
            "preferred_blend": 2,
            "reward_weight": 0.5,
            "wrong_blends": [
                {"blend": 3, "penalty": 3.0},
            ]
        },
        
        # Bacterium (small) - math dominant but STILL shouldn't "tower"
        {
            "confs": [0.70, 0.95, 0.60],
            "scale": "small",
            "preferred_blend": 1,  # factual_lead better than comparative for small
            "reward_weight": 0.5,
            "wrong_blends": [
                {"blend": 3, "penalty": 3.0},  # "towering" = WRONG
            ]
        },
    ]
    
    return training

# ============================================
# PHYLACTERY MODEL (same as before)
# ============================================

class MiniPhylactery(nn.Module):
    def __init__(self, input_dim=64, hidden_dim=128, num_blend_types=4):
        super().__init__()
        
        self.creative_encoder = nn.Linear(input_dim, hidden_dim)
        self.math_encoder = nn.Linear(input_dim, hidden_dim)
        self.knowledge_encoder = nn.Linear(input_dim, hidden_dim)
        self.conf_processor = nn.Linear(3, hidden_dim)
        
        # NEW: Scale encoder (large vs small)
        self.scale_encoder = nn.Linear(1, hidden_dim)
        
        self.blend_attention = nn.MultiheadAttention(hidden_dim, num_heads=4)
        self.blend_decider = nn.Linear(hidden_dim, num_blend_types)
        self.weight_assigner = nn.Linear(hidden_dim, 3)
        
    def forward(self, creative_emb, math_emb, knowledge_emb, confidences, scale=None):
        c_enc = torch.relu(self.creative_encoder(creative_emb))
        m_enc = torch.relu(self.math_encoder(math_emb))
        k_enc = torch.relu(self.knowledge_encoder(knowledge_emb))
        conf_enc = torch.relu(self.conf_processor(confidences))
        
        if scale is not None:
            scale_enc = torch.relu(self.scale_encoder(scale))
            stacked = torch.stack([c_enc, m_enc, k_enc, conf_enc, scale_enc], dim=0)
        else:
            stacked = torch.stack([c_enc, m_enc, k_enc, conf_enc], dim=0)
        
        attended, attn_weights = self.blend_attention(stacked, stacked, stacked)
        pooled = attended.mean(dim=0)
        
        blend_logits = self.blend_decider(pooled)
        blend_type = torch.softmax(blend_logits, dim=-1)
        weights = torch.softmax(self.weight_assigner(pooled), dim=-1)
        
        return {
            'blend_logits': blend_logits,
            'blend_type': blend_type,
            'blend_type_idx': blend_type.argmax(dim=-1),
            'weights': weights,
        }

# ============================================
# SHAPED LOSS FUNCTION
# ============================================

class ShapedLoss(nn.Module):
    """
    Custom loss with:
    - Reward for correct answer
    - Heavy penalty for wrong answers
    """
    
    def __init__(self):
        super().__init__()
        self.base_loss = nn.CrossEntropyLoss(reduction='none')
    
    def forward(self, logits, target, reward_weight=1.0, wrong_blends=None):
        # Base loss for correct target
        base = self.base_loss(logits, target) * reward_weight
        
        # Add penalty for wrong blends
        if wrong_blends:
            probs = torch.softmax(logits, dim=-1)
            for wrong in wrong_blends:
                wrong_idx = wrong['blend']
                penalty = wrong['penalty']
                # Penalize probability mass on wrong answer
                wrong_prob = probs[0, wrong_idx]
                base = base + (wrong_prob * penalty)
        
        return base

# ============================================
# TEST INPUTS
# ============================================

def create_test_inputs():
    return [
        {
            "creative": {"data": "a golden dragon with ancient scales", "conf": 0.95},
            "math": {"data": "23 feet", "conf": 0.87},
            "knowledge": {"data": "humans average 5.75 feet", "conf": 0.92},
            "query": "dragon 4 times bigger than human",
            "scale": "large"
        },
        {
            "creative": {"data": "a tiny fairy with shimmering wings", "conf": 0.92},
            "math": {"data": "6 inches", "conf": 0.90},
            "knowledge": {"data": "typical hand is 7 inches", "conf": 0.85},
            "query": "fairy that fits in your palm",
            "scale": "small"
        },
        {
            "creative": {"data": "a massive ancient oak tree", "conf": 0.88},
            "math": {"data": "150 feet", "conf": 0.95},
            "knowledge": {"data": "10-story building is about 100 feet", "conf": 0.91},
            "query": "really tall tree",
            "scale": "large"
        },
        {
            "creative": {"data": "a microscopic robot", "conf": 0.93},
            "math": {"data": "50 nanometers", "conf": 0.91},
            "knowledge": {"data": "red blood cells are about 7000 nanometers", "conf": 0.89},
            "query": "tiny nanobot",
            "scale": "small"
        },
        {
            "creative": {"data": "a colossal space station", "conf": 0.90},
            "math": {"data": "3 miles wide", "conf": 0.88},
            "knowledge": {"data": "Manhattan is about 2 miles wide", "conf": 0.85},
            "query": "huge space station",
            "scale": "large"
        },
    ]

# ============================================
# TEXT EMBEDDING
# ============================================

def text_to_embedding(text, dim=64):
    torch.manual_seed(hash(text) % 2**32)
    return torch.randn(1, dim)

# ============================================
# TEMPLATE RENDERER (now with scale awareness)
# ============================================

def render_blend(blend_decision, inputs):
    blend_idx = blend_decision['blend_type_idx'].item()
    weights = blend_decision['weights'][0].tolist()
    blend_name = BLEND_TYPES[blend_idx]
    scale = inputs.get('scale', 'large')
    
    creative = inputs['creative']['data']
    math_val = inputs['math']['data']
    knowledge = inputs['knowledge']['data']
    
    # Calculate comparison
    try:
        import re
        math_num = float(re.search(r'[\d.]+', math_val).group())
        know_num = float(re.search(r'[\d.]+', knowledge).group())
        times = round(math_num / know_num, 1)
        comparison = str(times)
    except:
        comparison = "several"
        times = 1.0
    
    # Render based on blend type AND scale
    if blend_name == "creative_lead":
        if scale == "large":
            result = f"{creative.capitalize()}, measuring an impressive {math_val}, compared to {knowledge}."
        else:
            result = f"{creative.capitalize()}, measuring just {math_val}, compared to {knowledge}."
            
    elif blend_name == "factual_lead":
        if scale == "large":
            result = f"At an impressive {math_val} ({knowledge}): {creative}."
        else:
            result = f"At a tiny {math_val} ({knowledge}): {creative}."
            
    elif blend_name == "integrated":
        if scale == "large":
            result = f"The massive {creative.split()[1] if len(creative.split()) > 1 else creative}, standing {math_val} tall, dwarfs {knowledge.split()[-1]}."
        else:
            result = f"The minuscule {creative.split()[1] if len(creative.split()) > 1 else creative}, at just {math_val}, compared to {knowledge}."
            
    elif blend_name == "comparative":
        if scale == "large":
            result = f"{creative.capitalize()}, towering at {comparison}x the size of {knowledge.split()[-1]}."
        else:
            result = f"{creative.capitalize()}, at just {comparison}x the size of {knowledge.split()[-1]}."
    
    return {
        'text': result,
        'blend_type': blend_name,
        'scale': scale,
        'weights': {
            'creative': weights[0],
            'math': weights[1],
            'knowledge': weights[2]
        }
    }

# ============================================
# MAIN
# ============================================

def main():
    print("=" * 60)
    print("PHYLACTERY WITH SHAPED REWARDS")
    print("=" * 60)
    print("Teaching what to AVOID, not just what to DO")
    print()
    
    # Create model
    phylactery = MiniPhylactery(input_dim=64, hidden_dim=128, num_blend_types=4)
    optimizer = optim.Adam(phylactery.parameters(), lr=0.01)
    criterion = ShapedLoss()
    
    # Training data
    training_data = create_shaped_training_data()
    print(f"Training on {len(training_data)} examples with shaped rewards...")
    print(f"  - Correct answers: 0.5x loss (reward)")
    print(f"  - Wrong answers: 3.0x penalty")
    print()
    
    # Train
    losses = []
    for epoch in range(200):  # More epochs
        random.shuffle(training_data)
        epoch_loss = 0
        
        for example in training_data:
            optimizer.zero_grad()
            
            c_emb = torch.randn(1, 64)
            m_emb = torch.randn(1, 64)
            k_emb = torch.randn(1, 64)
            confs = torch.tensor([example['confs']])
            
            # Scale: 1.0 for large, 0.0 for small
            scale_val = torch.tensor([[1.0 if example['scale'] == 'large' else 0.0]])
            
            decision = phylactery(c_emb, m_emb, k_emb, confs, scale_val)
            
            target = torch.tensor([example['preferred_blend']])
            wrong_blends = example.get('wrong_blends', None)
            reward_weight = example.get('reward_weight', 1.0)
            
            loss = criterion(
                decision['blend_logits'], 
                target, 
                reward_weight=reward_weight,
                wrong_blends=wrong_blends
            )
            
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(training_data)
        losses.append(avg_loss)
        
        if epoch in [0, 49, 99, 149, 199]:
            print(f"  Epoch {epoch+1}: loss = {avg_loss:.4f}")
    
    # Test
    print("\
" + "=" * 60)
    print("TESTING WITH SCALE AWARENESS")
    print("=" * 60)
    
    test_cases = create_test_inputs()
    phylactery.eval()
    
    with torch.no_grad():
        for i, inputs in enumerate(test_cases):
            print(f"\
--- Test {i+1}: {inputs['query']} ({inputs['scale']}) ---")
            
            c_emb = text_to_embedding(inputs['creative']['data'])
            m_emb = text_to_embedding(inputs['math']['data'])
            k_emb = text_to_embedding(inputs['knowledge']['data'])
            confs = torch.tensor([[
                inputs['creative']['conf'],
                inputs['math']['conf'],
                inputs['knowledge']['conf']
            ]])
            scale_val = torch.tensor([[1.0 if inputs['scale'] == 'large' else 0.0]])
            
            decision = phylactery(c_emb, m_emb, k_emb, confs, scale_val)
            output = render_blend(decision, inputs)
            
            print(f"Scale: {inputs['scale']}")
            print(f"Confs: creative={inputs['creative']['conf']}, math={inputs['math']['conf']}, knowledge={inputs['knowledge']['conf']}")
            print(f"Decision: {output['blend_type']}")
            print(f"Weights: c={output['weights']['creative']:.2f}, m={output['weights']['math']:.2f}, k={output['weights']['knowledge']:.2f}")
            print(f"Output: \"{output['text']}\"")
            
            # Check for semantic errors
            if inputs['scale'] == 'small' and 'towering' in output['text'].lower():
                print("\u26a0\ufe0f  SEMANTIC ERROR: 'towering' used for small object!")
            else:
                print("\u2713 Semantic check passed")
    
    # Consistency test
    print("\
" + "=" * 60)
    print("CONSISTENCY TEST")
    print("=" * 60)
    
    # Test large object
    large_decisions = []
    c_emb = text_to_embedding("a golden dragon")
    m_emb = text_to_embedding("23 feet")
    k_emb = text_to_embedding("humans 5.75 feet")
    confs = torch.tensor([[0.95, 0.87, 0.92]])
    scale_large = torch.tensor([[1.0]])
    
    for _ in range(10):
        decision = phylactery(c_emb, m_emb, k_emb, confs, scale_large)
        large_decisions.append(decision['blend_type_idx'].item())
    
    print(f"Large object (dragon) - 10 runs: {large_decisions}")
    print(f"Consistent: {len(set(large_decisions)) == 1}")
    
    # Test small object
    small_decisions = []
    c_emb = text_to_embedding("a tiny fairy")
    m_emb = text_to_embedding("6 inches")
    k_emb = text_to_embedding("hand 7 inches")
    confs = torch.tensor([[0.92, 0.90, 0.85]])
    scale_small = torch.tensor([[0.0]])
    
    for _ in range(10):
        decision = phylactery(c_emb, m_emb, k_emb, confs, scale_small)
        small_decisions.append(decision['blend_type_idx'].item())
    
    print(f"Small object (fairy) - 10 runs: {small_decisions}")
    print(f"Consistent: {len(set(small_decisions)) == 1}")
    
    # Key test: Are they DIFFERENT?
    print(f"\
Different decisions for large vs small: {large_decisions[0] != small_decisions[0]}")
    
    print("\
" + "=" * 60)
    print("END OF SHAPED REWARD TEST")
    print("=" * 60)

if __name__ == "__main__":
    main()
