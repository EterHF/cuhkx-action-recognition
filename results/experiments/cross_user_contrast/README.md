# Same-class cross-user contrast

[English](README.md) | [简体中文](README.zh-CN.md) | [Experiment index](../README.md)

A weight-0.05, temperature-0.1 supervised contrastive term is applied directly
to the 40-D TCN representation used by the classifier. Positives are same-class
clips from different outer-train users; different classes are negatives, while
same-class/same-user pairs are ignored rather than mislabeled as negatives.

The signal was active: every fold had at least 2,118 valid anchors per epoch,
and contrast loss decreased in every fold. Nevertheless, Temporal fell from
2,847 to 2,843 and T+V from 2,889 to 2,888: 5 rows corrected, 6 broken, net -1,
with fold nets `-1/-1/+1/0/0`. Subject-macro and valid-input accuracy also fell.
The preregistered recipe failed; weight and temperature were not scanned.
