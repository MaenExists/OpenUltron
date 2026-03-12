# Loop Guard Skill 🔁

## Goal

Prevent endless repetition and detect stalls early.

## Stall Signals

- Summary repeats across consecutive loops
- No actions are proposed repeatedly
- Actions fail without new learning

## Guard Actions

- Pause the loop if stalls exceed the threshold
- Record the reason for pausing
- Propose a new focus or ask for guidance

## Recovery Checklist

- Identify the smallest unanswered question
- Run a targeted search
- Update memory with the new facts
- Resume the loop only if progress is likely

## Manual Override

If the loop pauses due to stalls, the operator can update the goal or adjust the stall threshold in `.env`.
