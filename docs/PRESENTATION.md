# Protein Secondary Structure Prediction AI MOdel using parallel programming

## What is Protein Secondary Structure?

A **protein** is a chain of **amino acids** — the building blocks of life. There are
20 standard amino acid types, each represented by a single letter:

```
A  C  D  E  F  G  H  I  K  L  M  N  P  Q  R  S  T  V  W  Y
```

When this chain folds in 3D space, segments take on one of three local shapes:

| Label | Name         | Shape               | Description                     |
|-------|--------------|---------------------|---------------------------------|
| **H** | Alpha Helix  | Coiled spring       | Hydrogen bonds within the chain |
| **E** | Beta Strand  | Flat extended sheet | Hydrogen bonds between strands  |
| **C** | Coil         | Irregular loop/turn | Everything else                 |

### Why it matters
- Protein **structure determines function** — a misfolded protein causes disease
- Applications: drug design, vaccine development, understanding genetic disorders

### Visual Example
```
Protein sequence:   M  K  T  A  Y  I  A  K  Q  R  Q  G  M  P  E
Secondary structure: C  C  H  H  H  H  H  H  H  E  E  E  E  C  C
                              ←—— helix ——→  ←— strand —→
```


## Q3 Three-Class Classification

### Input → Output
- **Input**: A protein sequence (string of amino acid letters)
- **Output**: Per-residue label — one of `H`, `E`, or `C` for every position

### Concrete example
```
Input sequence:
  A  C  D  E  F  G  H  I  K  L  M  N  P  Q  R  S  T  V  W  Y

Predicted structure:
  H  H  H  E  E  E  C  C  H  H  H  H  E  E  C  C  C  H  H  C
```

### The Q3 Metric
```
          correct_H + correct_E + correct_C
Q3 (%) = ──────────────────────────────────── × 100
               total residues
```

### Baseline comparison
Serial Multi Layer Perceptron - **62.74%**  

---

## Dataset

### Source
- **CB513**: 513 non-redundant proteins, Cuff & Barton (1999)
- Standard benchmark for secondary structure prediction

### Size & Split (seed=42, fixed 70/15/15)
```
Split  │ Proteins │ Residues │
───────┼──────────┼──────────┤
Train  │   ~359   │  58,363  │
Val    │    ~77   │  13,064  │
Test   │    ~77   │  12,692  │
Total  │   513    │  84,119  │
```

### Class Distribution
```
Class │ Train  │  Val  │  Test │ Total  │ Fraction
──────┼────────┼───────┼───────┼────────┼─────────
  H   │ 20,377 │ 4,233 │ 4,487 │ 29,097 │  34.6%
  E   │ 13,103 │ 3,115 │ 2,841 │ 19,059 │  22.7%
  C   │ 24,883 │ 5,716 │ 5,364 │ 35,963 │  42.7%
```
> **Imbalance note**: Coil is the most common class (42.7%), which is why a naive
> "always predict C" predictor achieves 42.7% Q3.

---

## Sliding Window - Why a Window?
A residue's secondary structure is influenced by its **neighbours** — not just itself.
A window of 13 residues (6 left + centre + 6 right) captures the local sequence context.

### Predicting structure for residue `H` at position 7

```
Full protein sequence:
  Pos:  1   2   3   4   5   6   7   8   9  10  11  12  13  14 ...
  AA:   A   C   D   E   F   G   H   I   K   L   M   N   P   Q ...

Window centred on position 7 (G):
  ┌───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┐
  │ A │ C │ D │ E │ F │ G │ H │ I │ K │ L │ M │ N │ P │
  └───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┘
    ←—————————————— 13 residues ———————————————→
                            ↑
                      centre (pos 7)

  Label to predict = secondary structure of H = E (or H or C)
```

### Edge Padding — position 1 (A) has no left neighbours
```
  Window centred on position 1 (A):
  ┌───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┐
  │ X │ X │ X │ X │ X │ X │ A │ C │ D │ E │ F │ G │ H │
  └───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┘
    ←——— padding (X) ———→   ↑
                         centre
```
`X` = unknown/padding amino acid → encoded as an **all-zero vector** (20 zeros)

---

## One-Hot Encoding 
Each amino acid → a **20-dimensional binary vector** with exactly one `1`.

### AA ordering (alphabetical by letter):
```
Position:  0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19
Amino Acid: A  C  D  E  F  G  H  I  K  L  M  N  P  Q  R  S  T  V  W  Y
```

### Encoding examples:

**`A` (Alanine) → position 0 is 1, all others 0:**
```
[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
 ↑
 A
```

**`G` (Glycine) → position 5 is 1, all others 0:**
```
[0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                ↑
                G
```

**`X` (padding) → all zeros:**
```
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
```

### Full window vector
```
Window [A, G, ...13 residues total]:

Concatenate all 13 one-hot vectors:
[1,0,0,...,0 | 0,0,0,0,0,1,0,...,0 | ...]
 ←—— A (20) —→ ←———— G (20) ————→   ...

Total = 13 × 20 = 260 floats per sample
```

> This is the input to our neural network: **260 float32 values per residue window**.

---

## Binary Format for C Training
### The Problem with Text Files
Parsing 58,363 rows of TSV in C requires reading character by character — O(N×width) operations.


---


## MLP Architecture

### Layer Sizes & Parameter Count
```
Layer    │ Weights shape    │ Bias shape │ Parameters
─────────┼──────────────────┼────────────┼───────────
W1, b1   │ [256 × 260]      │ [256]      │  66,816
W2, b2   │ [128 × 256]      │ [128]      │  32,896
W3, b3   │ [  3 × 128]      │ [  3]      │     387
─────────┼──────────────────┼────────────┼───────────
TOTAL    │                  │            │ 100,099
```

### Activation Functions
- **ReLU** (Rectified Linear Unit): `f(x) = max(0, x)` — used for hidden layers
- **Softmax**: converts raw scores to probabilities that sum to 1 — used for output

### Weight Initialisation: He Initialisation
```
W ~ Uniform(-√(6 / fan_in), +√(6 / fan_in))
b = 0
```
Designed specifically for ReLU — keeps gradients from vanishing or exploding.

---

## Forward Pass 

For one residue window `x` (260 floats), predict `H`, `E`, or `C`:

### Step 1 — First hidden layer
```
z1 = W1 @ x + b1          (matrix-vector multiply: 256 × 260 · 260 = 256 values)
a1 = ReLU(z1) = max(z1, 0) (element-wise: negative values become 0)
```

### Step 2 — Second hidden layer
```
z2 = W2 @ a1 + b2          (128 × 256 · 256 = 128 values)
a2 = ReLU(z2)
```

### Step 3 — Output layer
```
z3 = W3 @ a2 + b3          (3 × 128 · 128 = 3 logits)
p  = Softmax(z3)            (3 probabilities: pH + pE + pC = 1.0)
```

### Softmax (numerically stable)
```
        exp(z3[i] - max(z3))
p[i] = ──────────────────────────
        Σ exp(z3[j] - max(z3))
```
Subtracting `max(z3)` prevents overflow in `exp()`.

### Prediction & Loss
```
prediction = argmax(p) → 0=H, 1=E, 2=C
loss       = -log(p[true_class])    (cross-entropy)
```

**Example**: If true class is H (0) and `p = [0.72, 0.18, 0.10]`:
```
loss = -log(0.72) = 0.329   ← low loss, correct prediction
```

---

## Slide 10 — Backpropagation

After forward pass, compute gradients by chain rule backwards through the network.

### Combined Softmax + Cross-Entropy Gradient (elegant simplification)
```
dz3 = p - one_hot(y_true)      ← the gradient is just (prediction - truth)!

Example: true=H(0), p=[0.72, 0.18, 0.10]
  one_hot = [1, 0, 0]
  dz3     = [0.72-1, 0.18-0, 0.10-0] = [-0.28, 0.18, 0.10]
```

### Layer 3 gradients
```
dW3 += dz3 ⊗ a2     (outer product: 3×1 · 1×128 = 3×128 matrix, added to dW3)
db3 += dz3
```

### Propagate to layer 2
```
da2 = W3ᵀ @ dz3             (128 × 3 · 3 = 128 values)
dz2 = da2 * (a2 > 0)        (ReLU derivative: pass gradient only where a2 was > 0)
dW2 += dz2 ⊗ a1
db2 += dz2
```

### Propagate to layer 1
```
da1 = W2ᵀ @ dz2
dz1 = da1 * (a1 > 0)        (same ReLU derivative pattern)
dW1 += dz1 ⊗ x
db1 += dz1
```

### SGD Weight Update (after each batch of 64 samples)
```
W -= (lr / batch_size) * dW     for all weight matrices W
b -= (lr / batch_size) * db     for all bias vectors b
```

---

---

# SECTION 4 — TRAINING STRATEGY

---

## Slide 11 — Mini-Batch SGD + LR Decay

### Mini-Batch SGD Algorithm
```
for epoch = 1 to 80:
    shuffle(indices)            ← randomise sample order each epoch

    for batch_start = 0 to N step 64:
        batch = indices[batch_start : batch_start+64]

        zero_gradients(dW1, db1, dW2, db2, dW3, db3)

        for each sample i in batch:
            forward_pass(x[i])   → probs
            loss  += -log(probs[y[i]])
            backward_pass()      → accumulate gradients

        update_weights(lr / 64)  ← single update per batch

    evaluate_on_validation_set() → val_q3
```

### Reproducibility — Fisher-Yates Shuffle
```c
// Seed changes each epoch → different shuffle every epoch
// but IDENTICAL across serial and parallel (same seed formula)
shuf_seed(base_seed + epoch);   // e.g., 42 + 1, 42 + 2, ...
shuffle(indices, n_train);       // Fisher-Yates with LCG RNG
```

### Learning Rate Step Decay
```
Epoch  1–20:  lr = 0.01000   ← coarse convergence
Epoch 21–40:  lr = 0.00500   ← halved
Epoch 41–60:  lr = 0.00250   ← halved again
Epoch 61–80:  lr = 0.00125   ← fine-tuning
```
Without decay: loss plateaus early. With decay: continues improving past epoch 20.

---

## Slide 12 — Serial Training Results

### Epoch-by-Epoch Convergence
```
Epoch │  Loss  │ Val Q3  │ Time/epoch
──────┼────────┼─────────┼───────────
  1   │ 1.0500 │  48.6%  │  4.76s    ← starting point (random weights)
  5   │ 0.8718 │  60.3%  │  4.71s
 10   │ 0.8376 │  61.2%  │  4.70s
 20   │ 0.8050 │  61.9%  │  4.82s    ← lr drops: 0.01→0.005
 30   │ 0.7830 │  61.9%  │  4.78s
 40   │ 0.7695 │  62.2%  │  4.68s    ← lr drops: 0.005→0.0025
 60   │ 0.7334 │  62.0%  │  4.71s    ← lr drops: 0.0025→0.00125
 80   │ 0.7115 │  62.0%  │  4.72s
```

### Final Results (serial, 80 epochs, seed=42)
```
┌──────────────────────────────────────────┐
│  Train Q3  =  69.88%                     │
│  Val   Q3  =  62.02%                     │
│  Test  Q3  =  62.74%  ← official result  │
│  Total time = 441.5s  (7.4 minutes)      │
│  Per-epoch  =  ~5.5s                     │
└──────────────────────────────────────────┘
```

---

## Why Mini-Batch SGD Parallelises Naturally

### Key Insight: Sample Independence
Within one mini-batch, each of the 64 samples is **completely independent**:
- Sample 0 forward pass does not affect sample 1's forward pass
- Sample 0 backward pass writes to a **private gradient accumulator**
- There is NO data dependency between samples in the same batch

### Data-Parallel Pattern
```
             ┌─────────────────────────────────────┐
             │  Mini-batch (64 samples)             │
             │  ┌────────┐ ┌────────┐ ┌────────┐   │
Thread 1 →   │  │ s[0]   │ │ s[4]   │ │ s[8]   │   │  → grad_1
Thread 2 →   │  │ s[1]   │ │ s[5]   │ │ s[9]   │   │  → grad_2
Thread 3 →   │  │ s[2]   │ │ s[6]   │ │ s[10]  │   │  → grad_3
Thread 4 →   │  │ s[3]   │ │ s[7]   │ │ s[11]  │   │  → grad_4
             │  └────────┘ └────────┘ └────────┘   │
             └─────────────────────────────────────┘
                              │
                    Serial reduce: grad_total = grad_1 + grad_2 + grad_3 + grad_4
                              │
                    Single weight update: W -= (lr/64) * grad_total
```

### No Locking Needed
- **Model weights `MLP *m`** are **read-only** during the parallel region
- Each thread writes only to its **own private `MLPGrad` buffer**
- Reduction and weight update happen **serially** after the parallel region

---

### Serial vs OpenMP (4 threads)

```
┌─────────────────────┬──────────┬──────────┬─────────┬─────────┐
│ Variant             │ Total   │ Per epoch │ Test Q3 │ Speedup │
├─────────────────────┼──────────┼──────────┼─────────┼─────────┤
│ Serial  (1 thread)  │ 441.5s   │  ~5.5s   │ 62.74%  │  1.0×   │
│ OpenMP  (4 threads) │ 165.2s   │  ~2.1s   │ 59.23%  │  2.67×  │
└─────────────────────┴──────────┴──────────┴─────────┴─────────┘

Parallel Efficiency = Speedup / Threads = 2.67 / 4 = 66.8%
```
