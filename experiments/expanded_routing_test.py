"""
EXPANDED ROUTING TEST - 3 Categories + Edge Cases
=================================================

Round 1: Clean examples (math, creative, knowledge)
Round 2: Edge cases (ambiguous, mixed intent)

Same architecture. More basins. Harder tests.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import random
from transformers import BertTokenizer, BertModel

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
# TRAINING DATA - 3 Categories
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
    "solve for x in 2x equals 10",
    "what is 17 squared",
    "factorial of 5",
    "convert 0.5 to fraction",
    "average of 10 20 30",
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
    "write a love letter",
    "describe a dream sequence",
    "create a villain backstory",
    "write a plot twist",
    "compose a limerick",
]

knowledge_examples = [
    "what is the capital of France",
    "who invented the telephone",
    "when did World War 2 end",
    "explain how photosynthesis works",
    "what causes earthquakes",
    "who wrote Romeo and Juliet",
    "what is the speed of light",
    "how does gravity work",
    "what is the largest planet",
    "who was the first president",
    "what is DNA",
    "how do airplanes fly",
    "what is the boiling point of water",
    "who painted the Mona Lisa",
    "what is the population of Earth",
    "explain the water cycle",
    "what is an atom",
    "who discovered penicillin",
    "what is the longest river",
    "how do magnets work",
]

# Labels: 0 = math, 1 = creative, 2 = knowledge
training_data = [(text, 0) for text in math_examples] + \
                [(text, 1) for text in creative_examples] + \
                [(text, 2) for text in knowledge_examples]

# ============================================
# TEST DATA - Round 1: Clean examples
# ============================================

clean_test_data = [
    # Math (should route to 0)
    ("what is 7 times 6", 0),
    ("add 100 and 250", 0),
    ("divide 81 by 9", 0),
    ("calculate the sum of 5 5 5", 0),
    ("multiply 12 by 11", 0),
    
    # Creative (should route to 1)
    ("write a poem about the ocean", 1),
    ("tell me a story about a knight", 1),
    ("create a fantasy world", 1),
    ("imagine a talking cat", 1),
    ("describe a haunted house", 1),
    
    # Knowledge (should route to 2)
    ("what is the capital of Japan", 2),
    ("who invented electricity", 2),
    ("how does the heart pump blood", 2),
    ("what year did the Titanic sink", 2),
    ("explain how rainbows form", 2),
]

# ============================================
# TEST DATA - Round 2: Edge cases
# ============================================

edge_case_test_data = [
    # Math + Creative blend
    ("write a story about a dragon who is 4 times bigger than a human", "ambiguous_math_creative"),
    ("create a poem using only numbers", "ambiguous_math_creative"),
    ("imagine a world where 2 plus 2 equals 5", "ambiguous_math_creative"),
    
    # Creative + Knowledge blend
    ("write a story about how dinosaurs went extinct", "ambiguous_creative_knowledge"),
    ("create a tale about Einstein discovering relativity", "ambiguous_creative_knowledge"),
    ("describe what it felt like to walk on the moon", "ambiguous_creative_knowledge"),
    
    # Math + Knowledge blend
    ("calculate the distance to the sun in miles", "ambiguous_math_knowledge"),
    ("what percentage of Earth is water", "ambiguous_math_knowledge"),
    ("how many seconds in a year", "ambiguous_math_knowledge"),
    
    # Truly ambiguous
    ("help me with my homework", "ambiguous_all"),
    ("I need assistance", "ambiguous_all"),
    ("can you help", "ambiguous_all"),
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
# MODEL - 3 Routes
# ============================================

class AttentionRouter(nn.Module):
    def __init__(self, embed_dim=768, num_heads=4, num_routes=3):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads)
        self.router_head = nn.Linear(embed_dim, num_routes)
    
    def forward(self, x):
        attn_out, attn_weights = self.attention(x, x, x)
        pooled = attn_out.squeeze(0)
        route_logits = self.router_head(pooled)
        return route_logits, attn_weights

# ============================================
# TRAINING
# ============================================

model = AttentionRouter(embed_dim=768, num_heads=4, num_routes=3)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

print("=" * 60)
print("EXPANDED ROUTING TEST - 3 Categories")
print("=" * 60)
print(f"Training: {len(training_data)} examples")
print(f"  Math: {len(math_examples)}")
print(f"  Creative: {len(creative_examples)}")
print(f"  Knowledge: {len(knowledge_examples)}")
print(f"Clean tests: {len(clean_test_data)}")
print(f"Edge cases: {len(edge_case_test_data)}")
print("=" * 60)

num_epochs = 100
losses = []

for epoch in range(num_epochs):
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
    
    avg_loss = epoch_loss / len(training_data)
    losses.append(avg_loss)
    
    if epoch in [0, 24, 49, 74, 99]:
        print(f"Epoch {epoch+1}: loss = {avg_loss:.4f}")

# ============================================
# TESTING - Round 1: Clean
# ============================================

print("\
" + "=" * 60)
print("ROUND 1: CLEAN TEST EXAMPLES")
print("=" * 60)

route_names = {0: "math", 1: "creative", 2: "knowledge"}
model.eval()
correct = 0

with torch.no_grad():
    for text, expected in clean_test_data:
        embedding = text_to_bert_embedding(text)
        x = embedding.unsqueeze(0)
        route_logits, _ = model(x)
        predicted = route_logits.argmax(dim=1).item()
        probs = torch.softmax(route_logits, dim=1)[0]
        
        is_correct = predicted == expected
        if is_correct:
            correct += 1
        
        status = "\u2713" if is_correct else "\u2717"
        print(f"{status} '{text[:45]}'")
        print(f"    Expected: {route_names[expected]}, Got: {route_names[predicted]}")
        print(f"    Probs: math={probs[0]:.3f}, creative={probs[1]:.3f}, knowledge={probs[2]:.3f}")

clean_accuracy = correct / len(clean_test_data)
print(f"\
Clean test accuracy: {correct}/{len(clean_test_data)} = {clean_accuracy:.1%}")

# ============================================
# TESTING - Round 2: Edge Cases
# ============================================

print("\
" + "=" * 60)
print("ROUND 2: EDGE CASES (Ambiguous)")
print("=" * 60)
print("No 'correct' answer - observing what the model chooses\
")

with torch.no_grad():
    for text, case_type in edge_case_test_data:
        embedding = text_to_bert_embedding(text)
        x = embedding.unsqueeze(0)
        route_logits, _ = model(x)
        predicted = route_logits.argmax(dim=1).item()
        probs = torch.softmax(route_logits, dim=1)[0]
        
        print(f"'{text[:50]}'")
        print(f"    Type: {case_type}")
        print(f"    Routed to: {route_names[predicted]}")
        print(f"    Probs: math={probs[0]:.3f}, creative={probs[1]:.3f}, knowledge={probs[2]:.3f}")
        print()

# ============================================
# FINAL SUMMARY
# ============================================

print("=" * 60)
print("FINAL SUMMARY")
print("=" * 60)
print(f"Training loss: {losses[0]:.4f} \u2192 {losses[-1]:.4f}")
print(f"Clean test accuracy: {clean_accuracy:.1%}")
print(f"\
Loss progression:")
print(f"  Epoch 1:   {losses[0]:.4f}")
print(f"  Epoch 25:  {losses[24]:.4f}")
print(f"  Epoch 50:  {losses[49]:.4f}")
print(f"  Epoch 100: {losses[99]:.4f}")
print("=" * 60)
print("END OF TEST")
print("=" * 60)
