# Beyond Prompt and Pray

*Building policy-governed AI agents from scratch.*

Most agent code is "prompt and pray" — you wire up a model, hand it tools, and
hope it stays in scope. This is the open, inspectable alternative: an agent loop
you can read in an afternoon, with **typed actions**, **runtime gates** that
detect and stop risky or out-of-scope behavior *before it executes*, multi-axis
**budgets**, **human escalation**, and a tamper-evident **audit log**. Everything
runs on the standard library plus `pydantic` — the baseline works with **no
license and no GMS**.

For production-grade guarantees — geometric plausibility, calibrated
admissibility, signed verdicts — the optional **GMS backend**
([`knowlytix`](https://knowlytix.ai/)) snaps in via a lazy seam. Clone it, run the
baseline, see exactly where GMS lifts detection and validation.

> Part of the **"Beyond … and Pray"** series:
> [governed agents](https://github.com/knowlytix/beyond-prompt-and-pray) ·
> [trustworthy RAG](https://github.com/knowlytix/beyond-chunk-and-pray) ·
> [test & validate](https://github.com/knowlytix/beyond-ship-and-pray) ·
> [LLMs from scratch](https://github.com/knowlytix/llm-from-scratch)

## What's inside

- **Governed agent loop** — generator-based, every step inspectable
- **Runtime gates** — `ALLOW` / `DENY` / `ESCALATE`, stop bad actions before they run
- **Typed actions** — `ToolCall` / `AskUser` / `Finish` / `Escalate`
- **Budgets** — token / time / tool-call / dollar caps, first-class
- **Human escalation** — reviewer protocol, not an error path
- **Tamper-evident audit** — hash-chained record of every decision
- **GMS-optional** — baseline runs free; geometric guarantees via `knowlytix`

## Install

```bash
pip install glassloop                 # core: loop, typed actions, tools, gates, audit, planning, memory
pip install "glassloop[ml]"           # + open-weight model tools (torch, transformers)
pip install "glassloop[gms]"          # + the licensed GMS backend (knowlytix; see below)
```

## The GMS upgrade (open-core)

`glassloop` runs fully without a license. The GMS-backed features — geometric
plausibility gate, calibrated admissibility, signed verdicts — require the
licensed [`knowlytix`](https://knowlytix.ai/) package, installed separately.
`glassloop` imports it lazily:

```python
import glassloop.gms as gms
gms.available()   # True if the licensed backend is installed
gms.require()     # returns knowlytix, or raises with install instructions
```

The production-grade, GMS-native edition is the *Beyond Prompt and Pray, Pro
Edition* — see [knowlytix.ai](https://knowlytix.ai/).

## License

Apache-2.0. © 2026 Knowlytix.
