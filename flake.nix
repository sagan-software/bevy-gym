{
  description = "Development shell for bevy-gym";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
    treefmt-nix = {
      url = "github:numtide/treefmt-nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs =
    {
      self,
      nixpkgs,
      flake-utils,
      treefmt-nix,
      ...
    }:
    flake-utils.lib.eachDefaultSystem (
      system:
      let
        pkgs = import nixpkgs {
          inherit system;
        };
        inherit (pkgs) lib;

        aiHarness = pkgs.callPackage ./ai/default.nix { };

        sourceExcludes = [
          "ai/bmad/**"
          "ai/bmad-output/**"
          "ai/skills/**"
          "ref/**"
          "target/**"
        ];
        deadnixExcludeArgs = "--exclude ${lib.concatMapStringsSep " " lib.escapeShellArg sourceExcludes}";
        statixIgnoreArgs = lib.concatMapStringsSep " " (
          pattern: "--ignore ${lib.escapeShellArg pattern}"
        ) sourceExcludes;
        rumdlExcludeArg = lib.escapeShellArg (lib.concatStringsSep "," sourceExcludes);

        taploLintCommand = ''
          find . -name '*.toml' \
            -not -path './ai/bmad/*' \
            -not -path './ai/bmad-output/*' \
            -not -path './ai/skills/*' \
            -not -path './ref/*' \
            -not -path './target/*' \
            -exec taplo lint {} +
        '';

        treefmtEval = treefmt-nix.lib.evalModule pkgs {
          projectRootFile = "flake.nix";
          settings.global.excludes = sourceExcludes ++ [
            ".direnv/**"
            ".git/**"
            "result"
            "result-*"
          ];
          programs = {
            rustfmt.enable = true;
            taplo.enable = true;
            rumdl-format.enable = true;
            nixfmt.enable = true;
            dprint = {
              enable = true;
              includes = [
                "*.json"
                "*.jsonc"
                "*.yaml"
                "*.yml"
              ];
              settings.plugins = pkgs.dprint-plugins.getPluginList (plugins: [
                plugins.dprint-plugin-json
                plugins.g-plane-pretty_yaml
              ]);
            };
          };
        };

        mkCargoCheck =
          name: command: extraNativeBuildInputs:
          pkgs.stdenvNoCC.mkDerivation {
            inherit name;
            src = self;

            cargoDeps = pkgs.rustPlatform.importCargoLock {
              lockFile = ./Cargo.lock;
            };

            nativeBuildInputs = [
              pkgs.rustPlatform.cargoSetupHook
              pkgs.cargo
              pkgs.pkg-config
              pkgs.rustc
            ]
            ++ extraNativeBuildInputs;

            buildPhase = ''
              runHook preBuild

              export HOME="$TMPDIR"
              export CARGO_TARGET_DIR="$TMPDIR/target"
              ${command}

              runHook postBuild
            '';

            installPhase = ''
              runHook preInstall
              mkdir -p "$out"
              runHook postInstall
            '';
          };

        mkRepoCheck =
          name: nativeBuildInputs: command:
          pkgs.runCommand name
            {
              inherit nativeBuildInputs;
            }
            ''
              export HOME="$TMPDIR"
              export XDG_CACHE_HOME="$TMPDIR/cache"
              mkdir -p "$XDG_CACHE_HOME"

              cd ${self}
              ${command}

              touch "$out"
            '';
      in
      {
        packages = {
          ai = aiHarness.layout;
          ai-harness-activate = aiHarness.activate;
          ai-harness-layout = aiHarness.layout;
        };

        apps = {
          ai-harness-activate = {
            type = "app";
            program = "${aiHarness.activate}/bin/ai-harness-activate";
            meta.description = "Refresh repo-local AI tool symlinks";
          };
        };

        formatter = treefmtEval.config.build.wrapper;

        checks = {
          cargo-check = mkCargoCheck "cargo-check" "cargo check --locked" [ ];
          cargo-clippy = mkCargoCheck "cargo-clippy" "cargo clippy --locked" [ pkgs.clippy ];
          taplo = mkRepoCheck "taplo-lint" [ pkgs.taplo ] taploLintCommand;
          rumdl = mkRepoCheck "rumdl-lint" [
            pkgs.rumdl
          ] "rumdl check . --no-cache --disable MD013 --exclude ${rumdlExcludeArg}";
          statix = mkRepoCheck "statix-check" [ pkgs.statix ] "statix check ${statixIgnoreArgs} .";
          deadnix = mkRepoCheck "deadnix-check" [ pkgs.deadnix ] "deadnix --fail ${deadnixExcludeArgs} .";
          treefmt = treefmtEval.config.build.check self;
        };

        devShells.default = pkgs.mkShell {
          packages = [
            aiHarness.activate
            pkgs.cargo
            pkgs.rustc
            pkgs.rustfmt
            pkgs.clippy
            pkgs.pkg-config
            treefmtEval.config.build.wrapper
            pkgs.deadnix
            pkgs.dprint
            pkgs.nixfmt
            pkgs.rumdl
            pkgs.statix
            pkgs.taplo
          ];

          shellHook = ''
            ${aiHarness.activate}/bin/ai-harness-activate
          '';
        };
      }
    );
}
