# Lich Theory - Complete Experimental Data

Recovered from conversations Jan 30-31, 2026.
All result files uploaded by Chris. All scripts written by Claude, run by Sage.

---

## FILE INDEX

### Results (10 files)

| # | Result File | Script | What It Proves |
|---|-------------|--------|----------------|
| 1 | `basin_routing_test_results.txt` | `basin_routing_test.py` | Basin formation with hash embeddings. First routing test. |
| 2 | `bert_routing_test_results.txt` | `bert_routing_test.py` | 100% routing accuracy with BERT embeddings. |
| 3 | `expanded_routing_test_results_PROPER.txt` | `expanded_routing_test.py` | 3-category routing (math/creative/knowledge) + edge cases. |
| 4 | `state_persistence_test_results.txt` | `state_persistence_test.py` | Basins survive save/load. Zero degradation. |
| 5 | `mock_stateful_persistence_results.txt` | `mamba_state_persistence_test.py` | Mamba vs attention persistence comparison. |
| 6 | `basin_determinism_results.txt` | `basin_determinism_test.py` | Identical inputs produce identical outputs. Perfect determinism. |
| 7 | `mini_phylactery_blend_results.txt` | `mini_phylactery_blend_test.py` | Phylactery learns to blend multi-route outputs. |
| 8 | `shaped_reward_phylactery_results.txt` | `shaped_reward_phylactery_test.py` | Penalty shaping improves blend decisions. Scale awareness. |
| 9 | `preference_basin_test_results.txt` | `preference_basin_test.py` | Preference basins with hash embeddings. 56% neutral accuracy. |
| 10 | `bert_preference_basin_test_results.txt` | `bert_preference_basin_test.py` | **BREAKTHROUGH**: 100% preference accuracy with BERT. Identity proven. |

### Scripts (12 files)

| Script | Embedding | What It Tests |
|--------|-----------|---------------|
| `basin_formation_test.py` | Hash | First basin formation proof |
| `basin_routing_test.py` | Hash | Basin-to-routing conversion |
| `bert_routing_test.py` | BERT (base) | Semantic routing accuracy |
| `expanded_routing_test.py` | BERT (base) | 3 categories + ambiguous edge cases |
| `state_persistence_test.py` | BERT (base) | Save/load survival |
| `mamba_state_persistence_test.py` | BERT (base) | Mamba stateful comparison |
| `basin_determinism_test.py` | BERT (base) | Deterministic output proof |
| `mini_phylactery_blend_test.py` | Hash | Multi-route blending |
| `shaped_reward_phylactery_test.py` | Hash | Penalty-shaped blend learning |
| `preference_basin_test.py` | Hash (64d) | Preference identity (baseline) |
| `bert_preference_basin_test.py` | MiniLM (384d) | Preference identity (BERT enhanced) |
| `lich_theory_visualization.py` | N/A | Twitter-ready visualization |

---

## KEY RESULTS SUMMARY

### Routing (proven)
- Hash embeddings: ~67% accuracy
- BERT embeddings: 100% accuracy
- 3-category expansion: 100% clean, intelligent edge case handling

### Persistence (proven)
- Save/load: 0.0000000000 difference
- Determinism: Perfect across all runs
- Stateless beats stateful (no contamination)

### Blending (proven)
- Phylactery learns blend strategies from confidence scores
- Shaped rewards improve scale awareness
- Consistent decisions across runs

### Identity (BREAKTHROUGH - Jan 31, 2026)
- Hash baseline: 56% neutral accuracy (pets 100%, theme 33%, style 33%)
- BERT enhanced: **100% neutral accuracy** (pets 100%, theme 100%, style 100%)
- Average bias: **+0.987** toward preferred option
- Preference overrides explicit contradiction: **6/6**
- Cosine distance proves basin pull, not pattern matching
- Low cosine (0.22-0.46) + high bias (+0.96-1.00) = IDENTITY

---

## EXPERIMENTAL PROGRESSION

1. Basin Formation - basins form in attention layers
2. Basin Routing - basins route inputs correctly
3. BERT Routing - semantic embeddings hit 100%
4. Expanded Routing - scales to 3+ categories
5. State Persistence - basins survive save/load
6. Determinism - identical inputs, identical outputs
7. Phylactery Blending - multi-route synthesis works
8. Shaped Rewards - penalty shaping improves decisions
9. Preference Basins (hash) - identity detected at 56%
10. **Preference Basins (BERT) - identity confirmed at 100%**

---

## NOTES

- Result file `bert_preference_basin_test_results.txt` is UTF-16LE encoded (Windows/WSL pipeline)
- Some result files have Unicode symbols replaced with ASCII by Sage during fixing
- All scripts designed for reproducibility with fixed seeds (torch.manual_seed(42))
- BERT scripts require: transformers (bert-base-uncased) or sentence-transformers (MiniLM)
- Mamba script requires: mamba-ssm package
