# RQ3: Recipient heterogeneity

- Complete claim-memory events: 85
- Recipient-level audit units: 255
- Recipients: A1, A2, A3
- Estimand: `U_team(memory, recipient)=Y(use_all)-Y(drop_recipient_only)`
- Neutral epsilon: 0.0
- Bootstrap unit: claim-memory event

## Current run

| Metric | Estimate [95% CI] |
|---|---:|
| Direct +/− recipient sign-flip rate | 2.35% [0.00, 5.88] |
| Any recipient sign heterogeneity rate | 25.88% [16.47, 35.29] |
| Mean within-event utility range | 0.09 [0.06, 0.12] |

## Per-recipient team utility

| Recipient | Mean utility [95% CI] | Positive | Neutral | Negative |
|---|---:|---:|---:|---:|
| A1 | 0.01 [-0.02, 0.04] | 9 | 67 | 9 |
| A2 | 0.01 [-0.02, 0.04] | 10 | 66 | 9 |
| A3 | 0.00 [-0.03, 0.04] | 8 | 66 | 11 |

## Independent retest

- Shared events: 85
- Previous direct sign-flip rate: 4.71% [1.18, 9.41]
- Current direct sign-flip rate: 2.35% [0.00, 5.88]
- Stable direct sign-flip rate (flip in both runs): 0.00% [0.00, 0.00]
- Reproducible directional flip rate (same positive/negative recipient orientation): 0.00% [0.00, 0.00]
- Recipient sign consistency: 73.33% [65.88, 80.39]
- Direct opposite-sign transition rate across runs: 4.31% [1.57, 7.84]

A direct sign flip observed only in one run is treated as sampling-sensitive. The strongest RQ3 evidence is the reproducible directional flip rate, not the single-run rate.
