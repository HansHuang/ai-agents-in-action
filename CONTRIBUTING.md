# Contributing to ai-agent-act

## Adding a New Code Implementation
1. Read the corresponding doc in `docs/` first
2. Follow the canonical structure for that concept
3. Match the exact class/method names from other language implementations
4. Include a `test_*.py` (or equivalent) file
5. Update the doc's "Code Reference" section with your new language link

## Code Standards
- Python: type hints required, black formatting
- Node.js: TypeScript optional but encouraged, async/await over callbacks
- Go: follow standard Go conventions, handle all errors

## Pull Request Checklist
- [ ] Tests pass
- [ ] README.md updated with run instructions
- [ ] Code structure matches sibling implementations
- [ ] No framework dependencies in chapters 01-05 (from scratch only)