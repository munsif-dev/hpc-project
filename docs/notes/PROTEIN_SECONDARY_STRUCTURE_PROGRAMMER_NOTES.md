# Protein Secondary Structure (Programmer Notes)
## Why this note exists
This note is derived from our discussion and explains the project in simple programming terms, not scientific jargon.

---

## 1) Core idea in one sentence
For every amino-acid character in a protein sequence, predict one label from `{H, E, C}` using nearby characters as context.

`H` = helix  
`E` = sheet  
`C` = coil

This is a standard **3-class classification** problem repeated across positions.

---

## 2) Think of it like tagging text
If NLP does:
- input: words
- output: part-of-speech tag per word

This project does:
- input: amino-acid letters
- output: structure tag (`H/E/C`) per letter

So protein secondary structure prediction is basically **sequence labeling/classification** in ML terms.

---

## 3) What raw data looks like
Each protein has:
- a sequence string (amino acids)
- a structure label string (target labels)

Example:
```text
Sequence: M K T A Y I A K Q
Labels:   C C H H H C C E E
Index:    0 1 2 3 4 5 6 7 8
```

Important rule: sequence length and label length must match.

---

## 4) What one training sample looks like
Model predicts one center position at a time using a sliding window.

Roadmap/paper-aligned window size = `13`:
- 6 residues on left
- 1 center residue
- 6 residues on right

Training sample = `(window_features, center_label)`.

---

## 5) Easy toy example (window = 5 for readability)
Real project uses window 13, but here 5 is easier to see.

```text
Sequence: M K T A Y I A K Q
Labels:   C C H H H C C E E
```

Take index `3` (residue `A`, true label `H`):

```text
Window residues: K T A Y I
Target label:    H
```

So one sample is:
- input window = `"KTAYI"`
- output class = `"H"`

Do this for all valid center positions to build dataset.

---

## 6) How letters become numbers
Neural nets need numeric inputs, so encode amino acids.

### Option A (MVP): One-hot encoding
Use fixed amino-acid order with 20 symbols, e.g.:
`A,C,D,E,F,G,H,I,K,L,M,N,P,Q,R,S,T,V,W,Y`

Examples:
- `A` -> `[1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]`
- `K` -> `[0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0]`
- `Y` -> `[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1]`

If window size is 13:
- each residue = 20 numbers
- each window = `13 x 20`
- flatten -> `260` features

So each sample is:
- `x_i`: vector length 260
- `y_i`: class id for center residue (`H/E/C`)

---

## 7) Final processed arrays (what training code uses)
After preprocessing:
- `X`: shape `[N, 260]`, float32
- `y`: shape `[N]`, int

Where:
- `N` = number of generated windows across all proteins
- class mapping usually fixed, e.g.:
  - `H -> 0`
  - `E -> 1`
  - `C -> 2`

---

## 8) What model outputs
For one sample, model outputs 3 scores/probabilities:
```text
[p(H), p(E), p(C)] = [0.78, 0.12, 0.10]
```
Prediction = class with max value (`argmax`) -> `H`.

For all samples, convert predicted class IDs back to chars (`H/E/C`) to get predicted structure string.

---

## 9) Q3 metric (main accuracy metric)
Q3 = percentage of residues predicted correctly in 3-state labels.

Formula:
```text
Q3 = (correct_H + correct_E + correct_C) / total_residues * 100
```

Mini example:
```text
True: C C H H H C C E E
Pred: C C H H C C C E E
```

Matches = 8 out of 9  
Q3 = `8/9 * 100 = 88.89%`

---

## 10) End-to-end pipeline in programmer steps
1. Load raw sequences + labels.
2. If needed, convert 8-state DSSP labels to 3-state:
   - `H,G,I -> H`
   - `B,E -> E`
   - others -> `C`
3. Generate sliding windows (size 13).
4. Encode windows (one-hot for MVP).
5. Save processed binary dataset.
6. Train serial model (reference truth).
7. Compute Q3 on val/test.
8. Run parallel versions (OpenMP, pthreads, MPI, hybrid, CUDA).
9. Compare accuracy parity vs serial and timing/speedup.

---

## 11) Tiny pseudocode
```pseudo
for protein in proteins:
    seq = protein.sequence
    lbl = protein.labels_3state
    for i in range(len(seq)):
        window = get_window(seq, i, size=13, padding="X")
        x = encode_one_hot(window)      # shape: 260
        y = map_label(lbl[i])           # H/E/C -> 0/1/2
        append(X, x)
        append(y_all, y)
```

Training:
```pseudo
logits = model(X_batch)         # shape: [B, 3]
loss = cross_entropy(logits, y_batch)
update_weights(loss)
```

Evaluation:
```pseudo
pred = argmax(logits, axis=1)
q3 = mean(pred == y_true) * 100
```

---

## 12) Common confusion and quick answers
Q: Is this one prediction per protein?  
A: No. It is one prediction per residue position.

Q: Why windowing?  
A: Center residue structure depends on nearby residues, so we provide local context.

Q: Why one-hot first?  
A: It is simple, stable, and matches project MVP goals.

Q: What does parallel code speed up?  
A: Mostly training time (forward/backward gradient work), not data meaning.

Q: What must stay consistent across versions?  
A: Data splits, seed, hyperparameters, and accuracy checks against serial baseline.

---

## 13) Mapping to this project structure
- Raw input files: `data/raw/`
- Processed arrays/binaries: `data/processed/`
- Preprocessing scripts: `scripts/preprocess/`
- Serial implementation: `src/serial/`
- OpenMP implementation: `src/openmp/`
- Pthreads implementation: `src/pthreads/`
- MPI implementation: `src/mpi/`
- Hybrid implementation: `src/hybrid/`
- CUDA implementation: `src/cuda/`
- Logs/tables: `results/`
- Plots: `plots/`
- Final write-up: `report/`

---

## 14) One final mental model
Treat this project as:
`sequence window -> 3-class classifier -> center label`

Repeat for every position.  
Then compare predicted label string with true label string using Q3.

