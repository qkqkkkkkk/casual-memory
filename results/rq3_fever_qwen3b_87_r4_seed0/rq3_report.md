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
| Direct +/− recipient sign-flip rate | 4.71% [1.18, 9.41] |
| Any recipient sign heterogeneity rate | 27.06% [17.65, 36.47] |
| Mean within-event utility range | 0.11 [0.07, 0.14] |

## Per-recipient team utility

| Recipient | Mean utility [95% CI] | Positive | Neutral | Negative |
|---|---:|---:|---:|---:|
| A1 | -0.02 [-0.06, 0.00] | 5 | 70 | 10 |
| A2 | 0.01 [-0.02, 0.04] | 11 | 65 | 9 |
| A3 | 0.01 [-0.03, 0.04] | 13 | 61 | 11 |

## Independent retest

Not available. Run the same design with a disjoint inference-seed range and pass `--retest-results` before interpreting sign flips as replicated recipient heterogeneity.
