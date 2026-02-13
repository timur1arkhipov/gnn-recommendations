# OB-GNN Diagrams

## 1. Discrete Vector Bundle on a Bipartite Graph

```
        Discrete Vector Bundle on a Bipartite Graph

     User fibers                        Item fibers
     F_u ≅ R^d                          F_i ≅ R^d

    ┌─────────┐       O_{iu}           ┌─────────┐
    │  x_u1   │─────────────────────►  │  x_i1   │
    │  ∈ F_u1 │  ◄─────────────────── │  ∈ F_i1 │
    └─────────┘       O_{u1,i1}        └─────────┘
         │                                  │
         │ O_{u1,i2}            O_{u2,i1}   │
         │                                  │
    ┌────▼────┐       O_{iu}           ┌────▼────┐
    │  x_u2   │─────────────────────►  │  x_i2   │
    │  ∈ F_u2 │  ◄─────────────────── │  ∈ F_i2 │
    └─────────┘       O_{u2,i2}        └─────────┘

    Standard GNN:  x_i ← Σ W · x_j      (W arbitrary)
    Bundle GNN:    x_i ← Σ O_{ij} · x_j  (O_{ij} ∈ O(d))
```

## 2. Orthogonal Connection Matrix Construction

```
            Orthogonal Connection Matrix Construction

  S_1 ∈ R^{b×b}  ──►  A_1 = S_1 - S_1^T  ──► exp(A_1) = Q_1
  S_2 ∈ R^{b×b}  ──►  A_2 = S_2 - S_2^T  ──► exp(A_2) = Q_2
      ...                    ...                    ...
  S_g ∈ R^{b×b}  ──►  A_g = S_g - S_g^T  ──► exp(A_g) = Q_g

   ┌────┬────┬─────┬────┐
   │ Q_1│  0 │ ... │  0 │
   ├────┼────┼─────┼────┤
   │  0 │ Q_2│ ... │  0 │  = W_block  ──► Shuffle ──► W_conn
   ├────┼────┼─────┼────┤
   │  0 │  0 │ ... │ Q_g│     W_conn^T · W_conn = I  ✓
   └────┴────┴─────┴────┘
```

## 3. OB-GNN Architecture

```
                    OB-GNN Architecture

  X^(0) ──► [ A·X·W_conn ] ──► [ X·W_local ] ──► (1-α)·T+α·X^(0) = X^(1)
    │           Transport        Group&Shuffle       Residual
    │
  X^(1) ──► [ A·X·W_conn ] ──► [ X·W_local ] ──► (1-α)·T+α·X^(0) = X^(2)
    │
   ...                                                       ...
    │
  X^(L-1) ─► [ A·X·W_conn ] ──► [ X·W_local ] ──► (1-α)·T+α·X^(0) = X^(L)
    │
    ▼
  Layer Aggregation:  X_final = Σ w_l · X^(l),  l = 0..L
                                 │
                        ┌────────┴────────┐
                        │  E_U  │   E_I   │
                        │(users)│ (items)  │
                        └─────────────────┘
```
