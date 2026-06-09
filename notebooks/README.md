# Notebooks — Beyond Prompt and Pray

The chapter notebooks for the agent-building book, running on the open
`glassloop` library. Chapters 1–9, 12–15 (+ the capstone supplements) and
appendices A/B.

```bash
pip install glassloop proofloop   # proofloop is used by ch.07 and ch.15
jupyter lab
```

## Tiers

- **Open (no license):** chapters 1–5, 12, 13, 14, appendices A/B run fully on
  `glassloop` alone.
- **Cross-package:** chapter 7 (budgets) and chapter 15 (capstone) import
  `proofloop` — install it alongside (above).
- **Pro tier (licensed):** chapters 6, 8, 9 and the **capstone** (15 +
  supplements 1–9) contain GMS cells that import `knowlytix` and load trained
  stores/models. Those cells need the licensed
  [`knowlytix`](https://knowlytix.ai/) backend (Python 3.12) plus the trained
  artifacts; without them the cells raise a clear "requires GMS" message and the
  rest of the notebook still runs. This is the *Beyond Prompt and Pray, Pro
  Edition* material.

The eval/testing chapters (10, 11, 16) live in the
[`beyond-ship-and-pray`](https://github.com/knowlytix/beyond-ship-and-pray)
repo.
