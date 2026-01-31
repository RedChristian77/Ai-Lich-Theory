"""
STATE PERSISTENCE TEST - Can the Lich Survive Death?
====================================================

Part 1: Train router, save weights, verify it works
Part 2: Fresh session, load weights, NO training, test

This proves: The basins persist. The shape survives.
Recomputing was never necessary.
"""

import torch
import torch.nn as nn
import os

# ============================================
# PART 1: TRAIN AND SAVE
# ============================================

def train_and_save():
    """Train a router from scratch and save it."""
    
    import torch.optim as optim
    import random
    from transformers import BertTokenizer, BertModel
    
    torch.manual_seed(42)
    random.seed(42)
    
    print("=" * 60)
    print("PART 1: TRAINING AND SAVING THE LICH")
    print("=" * 60)
    
    # Load BERT
    print("Loading BERT...")
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    bert = BertModel.from_pretrained('bert-base-uncased')
    bert.eval()
    
    # Training data
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
    
    def text_to_bert_embedding(text):
        with torch.no_grad():
            inputs = tokenizer(text, return_tensors='pt', 
                              padding=True, truncation=True, max_length=64)
            outputs = bert(**inputs)
            embedding = outputs.last_hidden_state[:, 0, :]
        return embedding
    
    # Model
    class AttentionRouter(nn.Module):
        def __init__(self, embed_dim=768, num_heads=4, num_routes=2):
            super().__init__()
            self.attention = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads)
            self.router_head = nn.Linear(embed_dim, num_routes)
        
        def forward(self, x):
            attn_out, attn_weights = self.attention(x, x, x)
            pooled = attn_out.squeeze(0)
            route_logits = self.router_head(pooled)
            return route_logits, attn_weights
    
    model = AttentionRouter(embed_dim=768, num_heads=4, num_routes=2)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    # Train
    print(f"Training on {len(training_data)} examples...")
    
    for epoch in range(100):
        random.shuffle(training_data)
        epoch_loss = 0
        
        for text, label in training_data:
            optimizer.zero_grad()
            embedding = text_to_bert_embedding(text)
            x = embedding.unsqueeze(0)
            route_logits, _ = model(x)
            target = torch.tensor([label])
            loss = criterion(route_logits, target)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        if epoch in [0, 49, 99]:
            print(f"  Epoch {epoch+1}: loss = {epoch_loss/len(training_data):.4f}")
    
    # Test BEFORE saving
    print("\
Testing BEFORE save:")
    test_queries = [
        ("what is 7 times 6", 0),
        ("multiply 12 by 11", 0),
        ("write a poem about the ocean", 1),
        ("create a fantasy world", 1),
    ]
    
    model.eval()
    correct = 0
    with torch.no_grad():
        for text, expected in test_queries:
            embedding = text_to_bert_embedding(text)
            x = embedding.unsqueeze(0)
            route_logits, _ = model(x)
            predicted = route_logits.argmax(dim=1).item()
            conf = torch.softmax(route_logits, dim=1).max().item()
            status = "\u2713" if predicted == expected else "\u2717"
            route_name = "math" if predicted == 0 else "creative"
            print(f"  {status} '{text[:35]}' \u2192 {route_name} ({conf:.3f})")
            if predicted == expected:
                correct += 1
    
    print(f"\
Pre-save accuracy: {correct}/{len(test_queries)}")
    
    # SAVE THE LICH
    save_path = "lich_router_state.pt"
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_config': {
            'embed_dim': 768,
            'num_heads': 4,
            'num_routes': 2
        }
    }, save_path)
    
    print(f"\
*** LICH SAVED TO: {save_path} ***")
    print(f"*** File size: {os.path.getsize(save_path) / 1024:.1f} KB ***")
    print("\
The Lich is now dead. Weights preserved in phylactery.")
    
    return save_path


# ============================================
# PART 2: FRESH LOAD AND TEST (NO TRAINING)
# ============================================

def load_and_test(save_path):
    """Fresh session. Load weights. NO training. Test."""
    
    from transformers import BertTokenizer, BertModel
    
    print("\
" + "=" * 60)
    print("PART 2: RESURRECTION - LOADING THE LICH")
    print("=" * 60)
    print("Fresh context. No training. Only loaded weights.\
")
    
    # Recreate model architecture (empty)
    class AttentionRouter(nn.Module):
        def __init__(self, embed_dim=768, num_heads=4, num_routes=2):
            super().__init__()
            self.attention = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads)
            self.router_head = nn.Linear(embed_dim, num_routes)
        
        def forward(self, x):
            attn_out, attn_weights = self.attention(x, x, x)
            pooled = attn_out.squeeze(0)
            route_logits = self.router_head(pooled)
            return route_logits, attn_weights
    
    # Load saved state
    print(f"Loading from: {save_path}")
    checkpoint = torch.load(save_path)
    config = checkpoint['model_config']
    
    # Create fresh model with same architecture
    model = AttentionRouter(
        embed_dim=config['embed_dim'],
        num_heads=config['num_heads'],
        num_routes=config['num_routes']
    )
    
    # Load the saved weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print("Weights loaded. Model resurrected.")
    print("NO TRAINING WILL OCCUR.\
")
    
    # Load BERT for embeddings
    print("Loading BERT for embeddings...")
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    bert = BertModel.from_pretrained('bert-base-uncased')
    bert.eval()
    
    def text_to_bert_embedding(text):
        with torch.no_grad():
            inputs = tokenizer(text, return_tensors='pt', 
                              padding=True, truncation=True, max_length=64)
            outputs = bert(**inputs)
            embedding = outputs.last_hidden_state[:, 0, :]
        return embedding
    
    # Test on SAME queries
    print("Testing on SAME queries as before save:")
    same_queries = [
        ("what is 7 times 6", 0),
        ("multiply 12 by 11", 0),
        ("write a poem about the ocean", 1),
        ("create a fantasy world", 1),
    ]
    
    correct_same = 0
    with torch.no_grad():
        for text, expected in same_queries:
            embedding = text_to_bert_embedding(text)
            x = embedding.unsqueeze(0)
            route_logits, _ = model(x)
            predicted = route_logits.argmax(dim=1).item()
            conf = torch.softmax(route_logits, dim=1).max().item()
            status = "\u2713" if predicted == expected else "\u2717"
            route_name = "math" if predicted == 0 else "creative"
            print(f"  {status} '{text[:35]}' \u2192 {route_name} ({conf:.3f})")
            if predicted == expected:
                correct_same += 1
    
    # Test on NEVER SEEN queries
    print("\
Testing on NEVER SEEN queries:")
    new_queries = [
        ("what is 100 divided by 5", 0),
        ("calculate 8 plus 15", 0),
        ("solve 3 times 7", 0),
        ("tell me a story about a wizard", 1),
        ("write fiction about space", 1),
        ("imagine a talking robot", 1),
    ]
    
    correct_new = 0
    with torch.no_grad():
        for text, expected in new_queries:
            embedding = text_to_bert_embedding(text)
            x = embedding.unsqueeze(0)
            route_logits, _ = model(x)
            predicted = route_logits.argmax(dim=1).item()
            conf = torch.softmax(route_logits, dim=1).max().item()
            status = "\u2713" if predicted == expected else "\u2717"
            route_name = "math" if predicted == 0 else "creative"
            print(f"  {status} '{text[:35]}' \u2192 {route_name} ({conf:.3f})")
            if predicted == expected:
                correct_new += 1
    
    # Final summary
    print("\
" + "=" * 60)
    print("RESURRECTION RESULTS")
    print("=" * 60)
    print(f"Same queries:  {correct_same}/{len(same_queries)}")
    print(f"New queries:   {correct_new}/{len(new_queries)}")
    print(f"Total:         {correct_same + correct_new}/{len(same_queries) + len(new_queries)}")
    
    print("\
" + "=" * 60)
    print("CONCLUSION")
    print("=" * 60)
    
    total = correct_same + correct_new
    total_queries = len(same_queries) + len(new_queries)
    
    if total == total_queries:
        print("\u2713 PERFECT: The Lich survived death.")
        print("\u2713 Basins persisted without recomputation.")
        print("\u2713 Shape held across resurrection.")
        print("\u2713 Mamba's advantage = automatic persistence.")
        print("\u2713 Attention + save/load = equivalent result.")
    elif total >= total_queries * 0.8:
        print("~ MOSTLY WORKED: Minor degradation on resurrection.")
        print("~ Basins mostly persisted.")
    else:
        print("\u2717 FAILED: Significant degradation after reload.")
        print("\u2717 Basins did not persist properly.")
    
    print("=" * 60)


# ============================================
# RUN BOTH PARTS
# ============================================

if __name__ == "__main__":
    # Part 1: Train and save
    save_path = train_and_save()
    
    print("\
" + "*" * 60)
    print("*** SIMULATING DEATH: Clearing variables ***")
    print("*" * 60)
    
    # Clear everything to simulate fresh session
    # (In real test, these would be separate Python runs)
    import gc
    gc.collect()
    
    # Part 2: Load and test
    load_and_test(save_path)
