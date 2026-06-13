# AI Workspace

This directory owns repo-local AI tool configuration and skills.

- Canonical skill source lives in `ai/skills`.
- BMad install files live in `ai/bmad`.
- BMad generated artifacts live in `ai/bmad-output`.
- Root discovery paths such as `.agents/skills` and `.claude/skills` are
  generated compatibility symlinks.
- Do not add root `_bmad` or `_bmad-output` compatibility links; update BMad
  config and skill instructions to use `ai/bmad` and `ai/bmad-output`.
