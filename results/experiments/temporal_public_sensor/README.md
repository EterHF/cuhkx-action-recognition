# strictV3 Temporal + public Depth/IR

[English](README.md) | [简体中文](README.zh-CN.md) | [Experiment index](../README.md)

This experiment retains strictV3 Temporal as the anchor and adds independently
fine-tuned public Omnivore Swin-T Depth and IR probabilities. All five
subject-held sensor folds were trained for 15 fixed epochs from the official
public initialization; no historical project checkpoint was loaded.

Full OOF accuracy was 0.940053 for Temporal, 0.492754 for Depth and 0.544466
for IR. The predeclared conservative probability blend
Temporal/Depth/IR=`0.90/0.05/0.05` reached 0.939723, one fewer correct row.
Fold deltas were A −1, B 0, C +1, D 0 and E −1; worst-user accuracy remained
0.8125. It failed the all-fold non-degradation gate.

Of 182 Temporal errors, Depth corrected 33, IR corrected 39, and both sensors
agreed on the correct class for 23. Conversely, the sensors agreed on the same
wrong class for 428 rows that Temporal classified correctly. A separate nested
weight diagnostic reached 0.939065 and selected zero sensor weight in outer
folds B, C and D. This confirms that the present sensor branches do not provide
a reliable gate signal.

The integration interface is retained, but its safe default is a zero sensor
gate, which is exactly the original Temporal model. No test candidate was built
and no Kaggle quota was used for this failed OOF route.

