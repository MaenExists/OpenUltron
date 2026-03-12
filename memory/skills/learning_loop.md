# Learning Loop Skill 🧠

## Goal

Turn every loop into **durable knowledge** by capturing unknowns, assumptions, and lessons.

## Core Principle

If the agent does not know something, it must **say so explicitly**, then take steps to learn it.

## Learning Workflow

- List unknowns immediately after observation
- Convert unknowns into research actions
- Record assumptions that guided decisions
- After actions, write lessons learned
- Store persistent lessons in `knowledge/lessons.md`

## Templates

### Unknowns

```markdown
- I do not know the current API behavior for X.
- I do not know whether Y supports Z.
```

### Assumptions

```markdown
- Assuming provider A is OpenAI‑compatible.
- Assuming default model is available.
```

### Lessons

```markdown
- When max_loops is set, the loop pauses before exceeding it.
- Full shell mode is required for piped commands.
```

## Failure‑Driven Learning

- Capture the failure message
- Identify the root cause
- Add a lesson with a prevention rule
- Propose a safer next attempt
