# Project Progress – Protein Secondary Structure Prediction using Parallel Programming

## Overview

This document records the progress of the **Protein Secondary Structure Prediction AI Model** project.
The goal of this project is to predict the **secondary structure of proteins** using a **Multi-Layer Perceptron (MLP)** implemented in C and accelerated with parallel programming techniques.

Protein secondary structure classes:

| Label | Structure   |
| ----- | ----------- |
| H     | Alpha Helix |
| E     | Beta Strand |
| C     | Coil        |

The project also explores **parallel training using OpenMP** to improve performance.

---

# Current Progress

## 1. Dataset Preparation

The **CB513 dataset** is used for training and evaluation.

Dataset statistics:

* Total proteins: 513
* Total residues: 84,119
* Classes: H (Helix), E (Strand), C (Coil)

Split:

| Dataset    | Samples |
| ---------- | ------- |
| Training   | ~58,363 |
| Validation | ~13,064 |
| Testing    | ~12,692 |

---

## 2. Feature Extraction

A **sliding window approach** is used.

Window size = **13 residues**

Example:

Protein sequence

```
A C D E F G H I K L M N P
```

Each amino acid is encoded using **one-hot encoding (20 dimensions)**.

Total feature vector size:

```
13 × 20 = 260 features
```

---

# Model Architecture

The neural network used is a **Multi-Layer Perceptron (MLP)**.

Structure:

```
Input Layer  : 260 neurons
Hidden Layer1: 256 neurons (ReLU)
Hidden Layer2: 128 neurons (ReLU)
Output Layer : 3 neurons (Softmax)
```

Parameter count ≈ **100k parameters**

---

# Sample Code

## Forward Pass Example

```c
void mlp_forward(const MLP *m,
                 const float *x,
                 float *a1,
                 float *a2,
                 float *logits,
                 float *probs)
{
    // Layer 1
    for (int i = 0; i < m->hidden1; i++) {
        float sum = m->b1[i];
        for (int j = 0; j < m->input_dim; j++) {
            sum += m->W1[i*m->input_dim + j] * x[j];
        }
        a1[i] = sum > 0 ? sum : 0;   // ReLU
    }

    // Layer 2
    for (int i = 0; i < m->hidden2; i++) {
        float sum = m->b2[i];
        for (int j = 0; j < m->hidden1; j++) {
            sum += m->W2[i*m->hidden1 + j] * a1[j];
        }
        a2[i] = sum > 0 ? sum : 0;
    }

    // Output layer
    for (int i = 0; i < m->output_dim; i++) {
        float sum = m->b3[i];
        for (int j = 0; j < m->hidden2; j++) {
            sum += m->W3[i*m->hidden2 + j] * a2[j];
        }
        logits[i] = sum;
    }
}
```

---

# Parallel Implementation (OpenMP)

Mini-batch training is parallelized using **OpenMP threads**.

Each thread processes different samples in the batch.

Example:

```c
#pragma omp parallel for schedule(static)
for (int i = 0; i < batch_size; i++) {

    int idx = indices[i];
    const float *x = train.X + idx * feature_dim;

    mlp_forward(&model, x, a1, a2, logits, probs);

}
```

Benefits:

* Multiple CPU cores used simultaneously
* Faster training time
* Suitable for HPC environments

---

# Current Results

Serial training results:

```
Train Q3 Accuracy : 69.88%
Validation Q3     : 62.02%
Test Q3           : 62.74%
Training Time     : ~441 seconds
```

OpenMP parallel results (4 threads):

```
Test Q3 Accuracy  : 59.23%
Training Time     : ~165 seconds
Speedup           : 2.67×
```

Parallel efficiency ≈ **66%**

---

# Next Steps

Planned improvements:

1. Implement **Pthreads version**
2. Implement **MPI distributed training**
3. Hybrid parallel model (**MPI + OpenMP**)
4. Investigate GPU acceleration (CUDA)
5. Improve model accuracy

---

# Conclusion

The project demonstrates that **parallel programming can significantly reduce training time** for machine learning models.

Using OpenMP threads, the training process achieved **2.67× speedup compared to the serial implementation**, while maintaining competitive prediction accuracy.

This work combines concepts from:

* Bioinformatics
* Machine Learning
* Parallel Programming
* High Performance Computing
