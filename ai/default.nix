{ pkgs }:
let
  inherit (pkgs) lib;

  skillsDir = ./skills;
  agentsMd = ./AGENTS.md;

  skillNames = lib.attrNames (
    lib.filterAttrs (_: type: type == "directory") (builtins.readDir skillsDir)
  );

  skillLinks =
    targetRoot:
    map (name: {
      name = "${targetRoot}/${name}";
      path = skillsDir + "/${name}";
    }) skillNames;

  codexConfig = pkgs.writeText "config.toml" ''
    #:schema https://developers.openai.com/codex/config-schema.json

    # Codex discovers AGENTS.md and .agents/skills automatically.
    # Repo-local skill sources live under ai/skills.
  '';

  layout = pkgs.linkFarm "ai-harness-layout" (
    [
      {
        name = "AGENTS.md";
        path = agentsMd;
      }
      {
        name = "CLAUDE.md";
        path = agentsMd;
      }
      {
        name = ".claude/CLAUDE.md";
        path = agentsMd;
      }
      {
        name = ".codex/config.toml";
        path = codexConfig;
      }
    ]
    ++ skillLinks ".agents/skills"
    ++ skillLinks ".claude/skills"
  );
in
{
  inherit layout;

  activate = pkgs.writeShellApplication {
    name = "ai-harness-activate";
    runtimeInputs = [ pkgs.gitMinimal ];
    text = ''
      repo_root="$(git rev-parse --show-toplevel 2>/dev/null || true)"
      if [ -z "$repo_root" ]; then
        exit 0
      fi
      cd "$repo_root"

      ln -sfn ${layout}/AGENTS.md AGENTS.md
      ln -sfn ${layout}/CLAUDE.md CLAUDE.md

      # BMad is configured to use ai/bmad and ai/bmad-output directly. Clean
      # stale compatibility symlinks from older activations without touching a
      # real directory if one exists.
      for stale_link in _bmad _bmad-output; do
        if [ -L "$stale_link" ]; then
          rm -f "$stale_link"
        fi
      done

      mkdir -p .agents .claude .codex
      ln -sfn ../ai/skills .agents/skills
      ln -sfn ../ai/skills .claude/skills
      ln -sfn ${layout}/.claude/CLAUDE.md .claude/CLAUDE.md
      ln -sfn ${layout}/.codex/config.toml .codex/config.toml
    '';
  };
}
