# BTC news relevance audit rubric

Label each sampled row as one of:

| Label | Meaning |
| --- | --- |
| `relevant` | Clearly about Bitcoin, BTC markets, major crypto venues/policy affecting BTC, or BTC-adjacent equities (MSTR/COIN/miners) in a crypto context. |
| `weak` | Tangentially related (broad “markets”, unrelated crypto alts with no BTC link, vague “blockchain” without BTC). |
| `irrelevant` | No meaningful BTC/crypto-market link (sports, local news, pure unrelated equities). |

## Examples

- “Bitcoin ETF inflows hit a weekly record” → **relevant**  
- “Coinbase Q2 earnings beat estimates” → **relevant**  
- “Ethereum gas fees drop” (no BTC) → **weak**  
- “Local bakery opens new store” → **irrelevant**  
- “MicroStrategy buys more BTC” → **relevant**

## Instructions

1. Open `data/processed/relevance_audit_sample.csv`.  
2. Fill `human_label` only (`relevant` / `weak` / `irrelevant`). Optional `notes`.  
3. Do not change `auto_relevant` or `audit_id`.  
4. Save as `relevance_audit_labeled.csv` and run:

```bash
python scripts/score_relevance_audit.py --labeled data/processed/relevance_audit_labeled.csv
```
