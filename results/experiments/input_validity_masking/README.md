# Branch-validity loss masking

[English](README.md) | [简体中文](README.zh-CN.md) | [Experiment index](../README.md)

Validity is derived from source decoding, not black-pixel heuristics: Visual is
valid when Depth or IR contains a PIL-verifiable image; skeleton is valid when
at least one frame passes the existing parser. This finds 103/3,036 clips with
neither primary input. They account for 73 of the frozen release's 133 OOF
errors, and are concentrated in user 5 (60 rows) and class 36 (30 rows).

The paired experiment retains every row in each forward pass and changes only
the CE mask; validation still includes every held-user row. The frozen-gate T+V
control scored 2,889. Visual-only masking scored 2,888, Temporal-only masking
2,885, and masking both scored 2,885: 8 corrected, 12 broken, net -4, with fold
nets `-1/-5/+2/0/0`. Valid-input and subject-macro accuracy also fell. Missing
inputs explain a large irreducible error subset, but loss masking alone does not
improve generalization and was rejected.
