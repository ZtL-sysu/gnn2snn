# Dense linear-operation accounting for the diagnostic

For one batch with N nodes, E directed edges and B graphs, the frozen GatedGCN has node input5, edge input1, width70, four gated layers and readout70→35→17→10. Count one multiply-accumulate per linear weight application; biases and all other operations are separate.

Embedding MACs: `350*N + 70*E`.

Each gated layer: A/B/D/E each apply70×70 to N nodes, C applies70×70 to E edges, giving `19600*N + 4900*E`. All four: `78400*N + 19600*E`.

MLP MACs: `(70*35 + 35*17 + 17*10)*B = 3215*B`.

Therefore ANN linear MACs are `embedding + body`, with `body=78400*N+19600*E+3215*B`. The current differential hybrid computes embeddings once and the entire body in each of T iterations, including its classifier. Its **executed dense linear MACs** are `embedding + T*body`. Source: gated_differential_hybrid_v1.py and gated_teacher_model.py. This is a formula for the present code, not a prediction of sparse hardware activity.

The validation structures show E=8N in the reported size cohorts; where this identity holds per graph, ANN is `236110*N+3215*B`, and hybrid is `(910+235200*T)*N+3215*T*B`. The linear-MAC multiplier is close to4 atT4 or8 atT8 because embeddings are a small fraction. No event-based sparse linear implementation is present in this diagnostic. The same projections are evaluated even when a preceding event is zero.

This count excludes sigmoid, edge-message multiply, scatters/reductions, denominator epsilon/division, BN, edge ReLU/residual, node residual, pooling, difference encoders, source neuron threshold selection, state updates, memory traffic, and six latch calls. These add work and state; they cannot rescue a claim that this dense implementation reduces total operations relative to the ANN. Signed graded nonzero event counts alone are insufficient as an energy proxy.

The positive result is low-step conversion fidelity for a strong input-dependent GNN. Efficient execution remains a separate unresolved problem. Any future cached/incremental-controller variant must measure fidelity loss and include cache/state/maintenance costs; its novelty requires comparison with prior asynchronous/incremental GNN and nonlinear conversion work.
