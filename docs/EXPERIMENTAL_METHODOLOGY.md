# Experimental Methodology

## Research Questions

**Primary Question:** Can neural attention mechanisms form stable "basins" that influence behavior beyond simple pattern matching?

**Secondary Questions:**
- Do attention layers develop persistent preferences?
- Can semantic embeddings enhance identity formation?
- Do neural preferences survive save/load cycles?
- Can AI identity override contradictory explicit input?

## Methodology

### Phase 1: Basin Formation Validation
**Objective:** Prove attention layers can form stable routing patterns

**Method:**
1. Train single attention layer on routing examples
2. Measure convergence and stability over iterations
3. Validate deterministic output patterns

**Success Criteria:** Consistent routing accuracy with stable attention weights

### Phase 2: Semantic Enhancement  
**Objective:** Compare hash vs BERT embeddings for routing accuracy

**Method:**
1. Test identical attention architecture with hash embeddings (baseline)
2. Test with BERT embeddings (semantic enhancement)
3. Measure routing accuracy across categories

**Success Criteria:** BERT shows measurable improvement over hash baseline

### Phase 3: Identity Formation
**Objective:** Test for persistent preferences beyond keyword matching

**Method:**
1. Train attention layer on explicit preferences (cats>dogs, dark>bright, concise>verbose)
2. Test on NEUTRAL queries with NO preference keywords
3. Measure bias strength and consistency
4. Test preference override of contradictory input

**Success Criteria:** Significant bias toward trained preferences on neutral queries

### Phase 4: Persistence Validation
**Objective:** Verify identity survives technical boundaries

**Method:**
1. Save trained model state
2. Reload and compare outputs on identical inputs  
3. Run consistency tests across multiple sessions

**Success Criteria:** Zero degradation in preference patterns after save/load

## Experimental Controls

**Randomization Control:**
- Fixed seeds (`torch.manual_seed(42)`) for reproducibility
- Multiple runs to validate consistency

**Variable Isolation:**
- Identical architectures across tests
- Controlled embedding types (hash vs BERT)
- Consistent training procedures

**Bias Detection:**
- Untrained baseline comparisons
- Explicit contradiction tests
- Semantic similarity measurements

## Statistical Validation

**Metrics:**
- Routing accuracy percentages
- Preference bias strength (difference between options)
- Cosine similarity to training data
- Cross-session consistency rates

**Significance:**
- Multiple independent runs
- Comparison with random baselines
- Effect size measurements

## Reproducibility Standards

**Complete Transparency:**
- All code available in `/experiments/`
- Raw data available in `/results/`  
- Fixed random seeds for exact replication
- Dependency management via `requirements.txt`

**Documentation:**
- Methodology documented before experiments
- Results captured without post-hoc modification
- Failed experiments documented alongside successes

---

*Scientific rigor ensures reliable, reproducible results.*