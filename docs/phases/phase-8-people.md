# Phase 8 — People (future)

| | |
|---|---|
| **Status** | Not started |
| **Last updated** | 2026-10-02 |
| **Depends on** | Phase 6 |

## Goal

Recognise specific people in addition to the two cats.

## Options

- **A. More classes in the same detector** — add one class per person to the dataset and retrain. Runs fully on
  the Pi with the existing pipeline. Limits: every new person means annotating and retraining; it is closed-set
  (a stranger is labelled as a known person); people are harder to tell apart from whole-body appearance
  (clothes change).
- **B. Detect, then identify** — the Pi detects a generic `person` (and `cat`); identity is decided by a second
  step on the PC (face recognition, e.g. InsightFace/ArcFace, for people; image embeddings for cats) against a
  small gallery of reference photos. New people need only reference photos and there is an "unknown" result.
  Needs the PC running and a channel from the Pi to the PC.

## Steps

- [ ] Decide between A and B (or A first, B later) based on how phase 6 went.
- [ ] Collect and annotate images of each person (with their consent).
- [ ] Extend the class map in `tools/via_to_yolo.py` / dataset config and retrain, or build the identification
      service for option B.

## Handover notes

_None yet._
