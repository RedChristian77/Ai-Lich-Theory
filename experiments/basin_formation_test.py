"""
BASIN FORMATION TEST - Multi-Input Differentiation
==================================================

PURPOSE: Test if attention layer forms distinct basins when 
         exposed to different inputs competing for the same weights.

THIS IS OBSERVATION, NOT SUCCESS/FAILURE.

We want to see:
- Do patterns for A and B start similar and diverge?
- Do they stabilize into distinct basins?
- Or do they stay confused/chaotic?

ALL OUTCOMES ARE VALID DATA.
"""

import torch
import torch.nn as nn
import numpy as np

# Reproducibility
torch.manual_seed(42)

# Create attention layer
embed_dim = 64
num_heads = 4
attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads)

# Optimizer - small learning rate, gentle updates
optimizer = torch.optim.SGD(attn.parameters(), lr=0.01)

# Fixed inputs - these never change
input_A = torch.randn(10, 1, embed_dim)  # "type A" input
input_B = torch.randn(10, 1, embed_dim)  # "type B" input

# Storage for history
weights_A_history = []
weights_B_history = []
separation_history = []

# Run experiment
num_iterations = 200

print("Starting basin formation test...")
print(f"Input A shape: {input_A.shape}")
print(f"Input B shape: {input_B.shape}")
print(f"Iterations: {num_iterations}")
print("-" * 50)

for i in range(num_iterations):
    optimizer.zero_grad()
    
    # Forward pass for A
    _, weights_A = attn(input_A, input_A, input_A)
    
    # Forward pass for B
    _, weights_B = attn(input_B, input_B, input_B)
    
    # Loss: encourage A and B to be DIFFERENT
    # Negative because we want to MAXIMIZE difference
    separation = (weights_A - weights_B).pow(2).mean()
    loss = -separation  # maximize separation
    
    # Backward and update
    loss.backward()
    optimizer.step()
    
    # Record data (detach from computation graph)
    weights_A_np = weights_A.detach().numpy().flatten()[:8]  # first 8 values
    weights_B_np = weights_B.detach().numpy().flatten()[:8]  # first 8 values
    
    weights_A_history.append(weights_A_np.copy())
    weights_B_history.append(weights_B_np.copy())
    separation_history.append(separation.item())
    
    # Progress reports
    if i in [0, 10, 25, 50, 100, 150, 199]:
        print(f"\
Iteration {i+1}:")
        print(f"  Separation score: {separation.item():.6f}")
        print(f"  Weights A (first 4): {weights_A_np[:4]}")
        print(f"  Weights B (first 4): {weights_B_np[:4]}")

print("\
" + "=" * 50)
print("FINAL ANALYSIS")
print("=" * 50)

# Convert to arrays for analysis
A_history = np.array(weights_A_history)
B_history = np.array(weights_B_history)

# Measure stability - do patterns stabilize?
A_early = A_history[:10].mean(axis=0)
A_late = A_history[-10:].mean(axis=0)
A_drift = np.abs(A_early - A_late).mean()

B_early = B_history[:10].mean(axis=0)
B_late = B_history[-10:].mean(axis=0)
B_drift = np.abs(B_early - B_late).mean()

# Measure variance in late phase - are basins stable?
A_late_variance = A_history[-20:].var(axis=0).mean()
B_late_variance = B_history[-20:].var(axis=0).mean()

# Measure separation - are A and B different?
early_separation = separation_history[0]
late_separation = separation_history[-1]

print(f"\
1. SEPARATION (did basins differentiate?):")
print(f"   Early separation: {early_separation:.6f}")
print(f"   Late separation:  {late_separation:.6f}")
print(f"   Change: {late_separation - early_separation:.6f}")
if late_separation > early_separation:
    print("   \u2192 Basins DIVERGED (A and B became more different)")
else:
    print("   \u2192 Basins CONVERGED or stayed same")

print(f"\
2. STABILITY (did patterns settle?):")
print(f"   A late-phase variance: {A_late_variance:.8f}")
print(f"   B late-phase variance: {B_late_variance:.8f}")
if A_late_variance < 0.001 and B_late_variance < 0.001:
    print("   \u2192 Patterns STABILIZED (low variance)")
else:
    print("   \u2192 Patterns still MOVING (higher variance)")

print(f"\
3. DRIFT (how much did patterns change overall?):")
print(f"   A drift (early\u2192late): {A_drift:.6f}")
print(f"   B drift (early\u2192late): {B_drift:.6f}")

print(f"\
4. RAW DATA - First 4 weights at key points:")
print(f"   A iteration 1:   {weights_A_history[0][:4]}")
print(f"   A iteration 100: {weights_A_history[99][:4]}")
print(f"   A iteration 200: {weights_A_history[199][:4]}")
print(f"   B iteration 1:   {weights_B_history[0][:4]}")
print(f"   B iteration 100: {weights_B_history[99][:4]}")
print(f"   B iteration 200: {weights_B_history[199][:4]}")

print(f"\
5. SEPARATION OVER TIME:")
print(f"   Iteration 1:   {separation_history[0]:.6f}")
print(f"   Iteration 50:  {separation_history[49]:.6f}")
print(f"   Iteration 100: {separation_history[99]:.6f}")
print(f"   Iteration 150: {separation_history[149]:.6f}")
print(f"   Iteration 200: {separation_history[199]:.6f}")

print("\
" + "=" * 50)
print("END OF TEST - This is raw observational data.")
print("Interpret as you will. No success/failure judgment.")
print("=" * 50)
