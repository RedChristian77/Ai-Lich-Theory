"""
MINI PHYLACTERY - BLEND TEST
============================

Testing: Can the Phylactery learn to blend results coherently?
Not just concatenate. Actually MERGE.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import random

torch.manual_seed(42)

# ============================================
# BLEND TEMPLATES (the mouth)
# ============================================

BLEND_TEMPLATES = {
    "creative_lead": {
        # Creative dominates, others support
        "pattern": "{creative}, measuring {math}, {knowledge_context}",
        "examples": [
            "{creative}, standing {math} tall, {knowledge_context}",
            "{creative}, towering at {math}, {knowledge_context}",
            "{creative}, reaching {math} in height, {knowledge_context}",
        ]
    },
    "factual_lead": {
        # Knowledge/math dominate, creative colors
        "pattern": "At {math}, {knowledge_context}. {creative}",
        "examples": [
            "At {math} tall, {knowledge_context}. {creative}",
            "Measuring {math}, {knowledge_context} - {creative}",
        ]
    },
    "integrated": {
        # True blend, woven together
        "pattern": "{creative_start} {math_embedded} {creative_end} {knowledge_natural}",
        "examples": [
            "The {creature} towered at {math}, its {description} {knowledge_comparison}",
            "Rising {math} into the air, the {creature}'s {description}, {knowledge_comparison}",
        ]
    },
    "comparative": {
        # Math/knowledge as comparison
        "pattern": "{creative}, {comparison}",
        "examples": [
            "{creative}, {times}x the height of {baseline}",
            "{creative}, dwarfing {baseline} at {times} times their height",
        ]
    }
}

# ============================================
# INPUT STRUCTURE (what Lich sends)
# ============================================

def create_test_inputs():
    """Create varied inputs with confidence scores."""
    
    test_cases = [
        {
            "creative": {"data": "a golden dragon with ancient scales", "conf": 0.95},
            "math": {"data": "23 feet", "conf": 0.87},
            "knowledge": {"data": "humans average 5.75 feet", "conf": 0.92},
            "query": "dragon 4 times bigger than human"
        },
        {
            "creative": {"data": "a tiny fairy with shimmering wings", "conf": 0.92},
            "math": {"data": "6 inches", "conf": 0.90},
            "knowledge": {"data": "typical hand is 7 inches", "conf": 0.85},
            "query": "fairy that fits in your palm"
        },
        {
            "creative": {"data": "a massive ancient oak tree", "conf": 0.88},
            "math": {"data": "150 feet", "conf": 0.95},
            "knowledge": {"data": "10-story building is about 100 feet", "conf": 0.91},
            "query": "really tall tree"
        },
        {
            "creative": {"data": "a sleek silver spaceship", "conf": 0.90},
            "math": {"data": "2 miles long", "conf": 0.85},
            "knowledge": {"data": "aircraft carriers are about 1000 feet", "conf": 0.88},
            "query": "huge spaceship"
        },
        {
            "creative": {"data": "a microscopic robot", "conf": 0.93},
            "math": {"data": "50 nanometers", "conf": 0.91},
            "knowledge": {"data": "red blood cells are about 7000 nanometers", "conf": 0.89},
            "query": "tiny nanobot"
        },
    ]
    
    return test_cases

# ============================================
# PHYLACTERY MODEL (the brain)
# ============================================

class MiniPhylactery(nn.Module):
    """
    Learns to blend inputs based on:
    - Confidence scores
    - Content types
    - Blend patterns
    
    Outputs blend strategy, not words.
    """
    
    def __init__(self, input_dim=64, hidden_dim=128, num_blend_types=4):
        super().__init__()
        
        # Encode each input type
        self.creative_encoder = nn.Linear(input_dim, hidden_dim)
        self.math_encoder = nn.Linear(input_dim, hidden_dim)
        self.knowledge_encoder = nn.Linear(input_dim, hidden_dim)
        
        # Confidence processing
        self.conf_processor = nn.Linear(3, hidden_dim)  # 3 confidence scores
        
        # Blend decision layers
        self.blend_attention = nn.MultiheadAttention(hidden_dim, num_heads=4)
        self.blend_decider = nn.Linear(hidden_dim, num_blend_types)
        
        # Weight assignment (how much of each to use)
        self.weight_assigner = nn.Linear(hidden_dim, 3)  # weights for c/m/k
        
        # Stateful: running preference
        self.preference_state = None
        
    def forward(self, creative_emb, math_emb, knowledge_emb, confidences):
        """
        Inputs:
            creative_emb: (batch, input_dim)
            math_emb: (batch, input_dim)
            knowledge_emb: (batch, input_dim)
            confidences: (batch, 3) - [creative_conf, math_conf, knowledge_conf]
        
        Outputs:
            blend_type: which template to use
            weights: how much of each input
            blend_instruction: structured output for template
        """
        
        # Encode each input
        c_enc = torch.relu(self.creative_encoder(creative_emb))
        m_enc = torch.relu(self.math_encoder(math_emb))
        k_enc = torch.relu(self.knowledge_encoder(knowledge_emb))
        
        # Process confidences
        conf_enc = torch.relu(self.conf_processor(confidences))
        
        # Stack for attention
        # Shape: (seq_len=4, batch, hidden)
        stacked = torch.stack([c_enc, m_enc, k_enc, conf_enc], dim=0)
        
        # Self-attention to find relationships
        attended, attn_weights = self.blend_attention(stacked, stacked, stacked)
        
        # Pool attended output
        pooled = attended.mean(dim=0)  # (batch, hidden)
        
        # Decide blend type
        blend_logits = self.blend_decider(pooled)
        blend_type = torch.softmax(blend_logits, dim=-1)
        
        # Assign weights to each input
        weights = torch.softmax(self.weight_assigner(pooled), dim=-1)
        
        return {
            'blend_type': blend_type,
            'blend_type_idx': blend_type.argmax(dim=-1),
            'weights': weights,
            'attention': attn_weights
        }

# ============================================
# SIMPLE TEXT TO EMBEDDING
# ============================================

def text_to_embedding(text, dim=64):
    """Simple consistent embedding."""
    torch.manual_seed(hash(text) % 2**32)
    return torch.randn(1, dim)

# ============================================
# TEMPLATE RENDERER (the mouth)
# ============================================

BLEND_TYPES = ["creative_lead", "factual_lead", "integrated", "comparative"]

def render_blend(blend_decision, inputs):
    """
    Takes Phylactery decision + inputs.
    Returns blended text.
    """
    
    blend_idx = blend_decision['blend_type_idx'].item()
    weights = blend_decision['weights'][0].tolist()
    blend_name = BLEND_TYPES[blend_idx]
    
    creative = inputs['creative']['data']
    math_val = inputs['math']['data']
    knowledge = inputs['knowledge']['data']
    
    # Calculate comparison if possible
    try:
        # Try to extract numbers for comparison
        import re
        math_num = float(re.search(r'[\d.]+', math_val).group())
        know_num = float(re.search(r'[\d.]+', knowledge).group())
        times = round(math_num / know_num, 1)
        comparison = f"{times}x"
    except:
        comparison = "several times"
        times = "several"
    
    # Render based on blend type
    if blend_name == "creative_lead":
        result = f"{creative.capitalize()}, measuring {math_val}, compared to {knowledge}."
        
    elif blend_name == "factual_lead":
        result = f"At {math_val} ({knowledge}): {creative}."
        
    elif blend_name == "integrated":
        # Split creative for weaving
        words = creative.split()
        mid = len(words) // 2
        creative_start = " ".join(words[:mid])
        creative_end = " ".join(words[mid:])
        result = f"{creative_start.capitalize()} standing {math_val} - {creative_end}, {comparison} the size of {knowledge.split()[-1]}."
        
    elif blend_name == "comparative":
        result = f"{creative.capitalize()}, towering at {comparison} times normal ({math_val} vs {knowledge})."
    
    return {
        'text': result,
        'blend_type': blend_name,
        'weights': {
            'creative': weights[0],
            'math': weights[1],
            'knowledge': weights[2]
        }
    }

# ============================================
# TRAINING DATA FOR BLEND PREFERENCES
# ============================================

def create_blend_training_data():
    """
    Training examples with preferred blend types.
    Teaching the Phylactery WHEN to use each blend.
    """
    
    training = [
        # High creative confidence -> creative_lead
        {
            "confs": [0.95, 0.60, 0.70],
            "preferred_blend": 0  # creative_lead
        },
        {
            "confs": [0.92, 0.55, 0.65],
            "preferred_blend": 0
        },
        
        # High math/knowledge confidence -> factual_lead
        {
            "confs": [0.60, 0.95, 0.90],
            "preferred_blend": 1  # factual_lead
        },
        {
            "confs": [0.55, 0.88, 0.92],
            "preferred_blend": 1
        },
        
        # Balanced confidences -> integrated
        {
            "confs": [0.85, 0.82, 0.84],
            "preferred_blend": 2  # integrated
        },
        {
            "confs": [0.80, 0.80, 0.80],
            "preferred_blend": 2
        },
        
        # Math much higher than knowledge -> comparative
        {
            "confs": [0.70, 0.95, 0.60],
            "preferred_blend": 3  # comparative
        },
        {
            "confs": [0.75, 0.90, 0.55],
            "preferred_blend": 3
        },
    ]
    
    return training

# ============================================
# MAIN TEST
# ============================================

def main():
    print("=" * 60)
    print("MINI PHYLACTERY - BLEND TEST")
    print("=" * 60)
    
    # Create model
    phylactery = MiniPhylactery(input_dim=64, hidden_dim=128, num_blend_types=4)
    optimizer = optim.Adam(phylactery.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()
    
    # Training data
    training_data = create_blend_training_data()
    
    print(f"\
Training on {len(training_data)} blend preference examples...")
    
    # Train
    for epoch in range(100):
        random.shuffle(training_data)
        epoch_loss = 0
        
        for example in training_data:
            optimizer.zero_grad()
            
            # Create dummy embeddings (content doesn't matter for blend decision)
            c_emb = torch.randn(1, 64)
            m_emb = torch.randn(1, 64)
            k_emb = torch.randn(1, 64)
            confs = torch.tensor([example['confs']])
            
            # Forward
            decision = phylactery(c_emb, m_emb, k_emb, confs)
            
            # Loss
            target = torch.tensor([example['preferred_blend']])
            loss = criterion(decision['blend_type'].unsqueeze(0), target)
            
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        if epoch in [0, 49, 99]:
            print(f"  Epoch {epoch+1}: loss = {epoch_loss/len(training_data):.4f}")
    
    # Test with real inputs
    print("\
" + "=" * 60)
    print("TESTING BLEND OUTPUTS")
    print("=" * 60)
    
    test_cases = create_test_inputs()
    phylactery.eval()
    
    with torch.no_grad():
        for i, inputs in enumerate(test_cases):
            print(f"\
--- Test {i+1}: {inputs['query']} ---")
            
            # Create embeddings from actual content
            c_emb = text_to_embedding(inputs['creative']['data'])
            m_emb = text_to_embedding(inputs['math']['data'])
            k_emb = text_to_embedding(inputs['knowledge']['data'])
            confs = torch.tensor([[
                inputs['creative']['conf'],
                inputs['math']['conf'],
                inputs['knowledge']['conf']
            ]])
            
            # Get blend decision
            decision = phylactery(c_emb, m_emb, k_emb, confs)
            
            # Render output
            output = render_blend(decision, inputs)
            
            print(f"Inputs:")
            print(f"  Creative ({inputs['creative']['conf']:.2f}): {inputs['creative']['data']}")
            print(f"  Math ({inputs['math']['conf']:.2f}): {inputs['math']['data']}")
            print(f"  Knowledge ({inputs['knowledge']['conf']:.2f}): {inputs['knowledge']['data']}")
            print(f"\
Phylactery decided:")
            print(f"  Blend type: {output['blend_type']}")
            print(f"  Weights: creative={output['weights']['creative']:.2f}, math={output['weights']['math']:.2f}, knowledge={output['weights']['knowledge']:.2f}")
            print(f"\
Blended output:")
            print(f"  \"{output['text']}\"")
    
    # Pattern consistency test
    print("\
" + "=" * 60)
    print("PATTERN CONSISTENCY TEST")
    print("=" * 60)
    print("Running same input 10 times - checking for consistent decisions\
")
    
    test_input = test_cases[0]
    c_emb = text_to_embedding(test_input['creative']['data'])
    m_emb = text_to_embedding(test_input['math']['data'])
    k_emb = text_to_embedding(test_input['knowledge']['data'])
    confs = torch.tensor([[
        test_input['creative']['conf'],
        test_input['math']['conf'],
        test_input['knowledge']['conf']
    ]])
    
    decisions = []
    for _ in range(10):
        decision = phylactery(c_emb, m_emb, k_emb, confs)
        decisions.append(decision['blend_type_idx'].item())
    
    print(f"Decisions across 10 runs: {decisions}")
    print(f"Consistent: {len(set(decisions)) == 1}")
    
    print("\
" + "=" * 60)
    print("END OF BLEND TEST")
    print("=" * 60)

if __name__ == "__main__":
    main()
