# MERGE-REPORT.md — flux-reasoner / flux-reasoner-engine

**Date:** 2026-05-03  
**Decision:** No merge needed. `flux-reasoner-engine` archived as redundant fork.

## Finding

Both repositories are **byte-for-byte identical**:

- Same `flux_reasoner/__init__.py` (dual-interpreter gradient reasoning engine)
- Same `pyproject.toml` (v0.1.0, setuptools, requests>=2.28)
- Same `README.md` (MIT license, same API docs)
- Same commit history (`cbebedf` Initial commit → `69deec6` Initial release 0.1.0)
- Identical gradient computation: `novelty - constraint` model
- Identical API: `FluxReasoner.reason()` and `reason_with_iterations()`

## Why No Merge Was Needed

`flux-reasoner-engine` was never actually differentiated from `flux-reasoner`. It is a **redundant fork** — a copy of the canonical repo that was cloned but never modified.

The gradient reasoning engine lives in one place: **SuperInstance/flux-reasoner**

## Action Taken

1. `flux-reasoner` (canonical) — retained as-is, README updated with archive note
2. `flux-reasoner-engine` — marked with `ARCHIVED.md` pointing to canonical repo

## Canonical Repository

**SuperInstance/flux-reasoner** — https://github.com/SuperInstance/flux-reasoner

## Lessons

- If a "different" repo never diverged, archive it immediately — don't let forks multiply
- Regular audit of org repos would catch this faster

## Future

If both need to exist for branding reasons, `flux-reasoner-engine` should be archived or redirected. The code should live in one canonical location.
