# Contributing 🤝

Thanks for helping improve OpenUltron! This guide keeps collaboration smooth and predictable.

## Ways To Contribute

- Fix bugs
- Improve docs
- Add tests
- Propose new features

## Development Setup

1. Create a venv and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Configure environment variables:

```bash
cp .env.example .env
```

3. Run the server:

```bash
uvicorn app:app --reload
```

## Tests

```bash
pytest
```

## Code Style

- Keep changes focused and minimal
- Prefer clear, direct names over cleverness
- Add tests for new behavior

## Pull Requests

- Explain the problem and the fix
- Link related issues
- Include screenshots for UI changes

## Code Of Conduct

This project follows `CODE_OF_CONDUCT.md`.
