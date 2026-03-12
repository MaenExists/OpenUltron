# Tool Use Skill 🛠️

## Purpose

Use tools to **gather evidence**, **modify files**, and **execute tasks** in a controlled, auditable way. Always choose the smallest tool that reliably solves the problem.

## Decision Checklist

- State the objective in one sentence
- Identify missing information
- Choose the tool that can resolve the unknown fastest
- Define the success criteria for the tool output
- Log the result and update memory if it changes your understanding

## Tool Selection Guide

- `search_web` when you need to discover sources or current facts
- `fetch_url` when you already have a source and need the content
- `shell` when you need local facts, file listings, or to run tests
- `write_file` / `append_file` when you need to change project files
- `write_memory` / `append_memory` when you need to preserve learning or logs

## Payload Templates

### search_web

```json
{
  "query": "OpenAI SDK tool calling best practices",
  "max_results": 5,
  "save_to": "knowledge/research/openai_tools.md"
}
```

### fetch_url

```json
{
  "url": "https://example.com/docs",
  "method": "GET",
  "headers": {"User-Agent": "OpenUltron"},
  "save_to": "knowledge/research/example_docs.md"
}
```

### shell

```json
{
  "cmd": "rg -n \"AsyncOpenAI\" openultron",
  "cwd": ".",
  "timeout": 60
}
```

### shell with full mode

```json
{
  "cmd": "ls -la | sed -n '1,5p'",
  "use_shell": true,
  "cwd": ".",
  "timeout": 60
}
```

### write_file

```json
{
  "path": "docs/notes.md",
  "content": "# Notes\n\n- item 1\n"
}
```

## Quality Gates

- Validate outputs before acting on them
- If results conflict with memory, update memory and note the conflict
- If the result is ambiguous, run a second tool to confirm

## Failure Handling

- Capture the error in `knowledge/failures.md`
- Add a lesson and a next step
- Retry with a smaller, more constrained action
