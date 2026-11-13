# Frozen-gate GatedGCN prototype protocol

## Method

`gated_frozen_gate_prototype.py` defines a label-free hybrid conversion
interface for the existing `GatedTeacher`. In eval mode, the controller runs
the complete ANN forward pass over each input graph batch. At every layer it
records edge gates

`g_uv = sigmoid(C(e_uv) + D(h_u) + E(h_v))`

using PyG's source-to-target edge convention. The frozen node replay computes

`A(h_v) + Σ_{u→v} g_uv ⊙ B(h_u) / (Σ_{u→v} g_uv + eps)`

with channel-wise target normalization. The sum in the denominator is per
target node and feature channel. This is an input-conditioned fixed linear
message operator, conditional on the teacher's gates.

The exact replay retains the teacher's node batch normalization, ReLU,
residual addition, mean/sum/max graph readout, MLP biases, two MLP ReLUs, and
analog final classifier. The controller also records each teacher edge state
after edge BN, ReLU, and residual update. The replay does not simulate edge
states: the ANN controller supplies their effect through the gates at every
layer. The shared input embedding is real-valued.

The optional staged path replaces each of the four node-branch ReLU outputs
and two MLP ReLU outputs with nonnegative channel threshold counts:

`q = theta/T * clamp(round(T * relu(a) / theta), 0, T)`.

The count activations are fed through the residual/MLP sequence in layer order;
this is a staged deterministic quantizer, not a causal per-tick simulator.
Residual bypasses, embedding, and final classifier remain analog. Diagnostics
include per-site counts and total counts. Thresholds are one positive scalar
per output channel and are supplied by the caller; this prototype does not
calibrate them.

## Minimal CPU synthetic verification

Run `pytest -q test_gated_frozen_gate_prototype.py` from this directory in the
local conda environment. The tests use a three-node directed graph with
asymmetric edges and no labels. They check exact frozen-gate replay against
`GatedTeacher`, manually calculated target-wise normalization by channel,
single-site quantization error/saturation, and count shapes/ranges at every
staged activation site.

## Scope and limitations

The exact match validates implementation consistency on a synthetic graph;
it does not establish benchmark accuracy or robustness. Test inputs may be
passed through a teacher controller without reading labels, but doing so at
deployment requires a full ANN forward pass and is therefore a teacher-oracle
hybrid, not an independent SNN. A validation-set search over labels or
thresholds would contaminate validation evidence. No real test split or labels
are used here.

The recorded edge states and all gates are teacher-conditioned; the node
quantizer does not feed back into later gates. Thus the staged result is not a
self-consistent dynamic GatedGCN conversion and makes no causal, latency,
energy, event-driven, or hardware-efficiency claim. Controller compute,
gate-storage cost, and threshold calibration are outside this prototype.
