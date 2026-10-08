{
  description = "Development shell for bevy-gym";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
    fenix = {
      url = "github:nix-community/fenix";
      inputs.nixpkgs.follows = "nixpkgs";
    };
    treefmt-nix = {
      url = "github:numtide/treefmt-nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs =
    {
      self,
      fenix,
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

        rustToolchain = fenix.packages.${system}.combine [
          fenix.packages.${system}.stable.toolchain
          fenix.packages.${system}.targets.wasm32-unknown-unknown.stable.rust-std
        ];
        wasmRustPlatform = pkgs.makeRustPlatform {
          cargo = rustToolchain;
          rustc = rustToolchain;
        };

        # mujoco-rs 5.x binds to MuJoCo 3.9.0 exactly. Keep the runtime next to
        # the development toolchain so Cargo builds and binaries use one ABI.
        mujoco = pkgs.fetchzip {
          url = "https://github.com/google-deepmind/mujoco/releases/download/3.9.0/mujoco-3.9.0-linux-x86_64.tar.gz";
          hash = "sha256-uHQgGkk3+y/A2oFEZ32PFY+ZeYIRsicJnyD8zLnESDU=";
        };

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
            rustfmt = {
              enable = true;
              edition = "2021";
            };
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

        # Keep prose and browser-test edits from rebuilding every Rust dependency.
        # Repository lint checks below still inspect the complete source tree.
        rustFiles = lib.fileset.fileFilter (
          file:
          lib.any file.hasExt [
            "rs"
            "toml"
            "lock"
            "json"
            "mpk"
            "xml"
          ]
        ) ./.;
        rustSource = lib.fileset.toSource {
          root = ./.;
          fileset = lib.fileset.unions [
            rustFiles
            ./LICENSE-MIT
            ./LICENSE-APACHE
            ./LICENSES
          ];
        };

        mkCargoCheck =
          name: command: extraNativeBuildInputs:
          pkgs.stdenvNoCC.mkDerivation {
            inherit name;
            src = rustSource;

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
              # Match CI without retaining debug symbols in discarded check builds.
              export CARGO_PROFILE_DEV_DEBUG=0
              export CARGO_PROFILE_TEST_DEBUG=0
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

        renderNativeBuildInputs = [
          pkgs.libxkbcommon
          pkgs.libx11
          pkgs.libxcursor
          pkgs.libxi
          pkgs.libxrandr
          pkgs.vulkan-loader
          pkgs.wayland
        ];
        renderRuntimeLibraryPath = lib.makeLibraryPath renderNativeBuildInputs;
        mujocoRuntimeLibraryPath = "${mujoco}/lib";

        webDist = pkgs.stdenvNoCC.mkDerivation {
          pname = "bevy-gym-web";
          version = "0.1.0";
          src = lib.fileset.toSource {
            root = ./.;
            fileset = lib.fileset.unions [
              rustFiles
              ./web/index.html
              ./web/styles.css
            ];
          };

          cargoDeps = wasmRustPlatform.importCargoLock {
            lockFile = ./Cargo.lock;
          };

          nativeBuildInputs = [
            wasmRustPlatform.cargoSetupHook
            rustToolchain
            pkgs.binaryen
            pkgs.trunk
            pkgs.wasm-bindgen-cli
          ];

          buildPhase = ''
            runHook preBuild

            export HOME="$TMPDIR"
            export CARGO_TARGET_DIR="$TMPDIR/target"
            export NO_COLOR=true
            export TRUNK_OFFLINE=true
            trunk build --config web/Trunk.toml --release --locked --dist "$out"

            runHook postBuild
          '';

          installPhase = "true";
        };

        webServe = pkgs.writeShellApplication {
          name = "bevy-gym-web-serve";
          runtimeInputs = [
            rustToolchain
            pkgs.binaryen
            pkgs.trunk
            pkgs.wasm-bindgen-cli
          ];
          text = ''
            if [[ ! -f web/index.html ]]; then
              echo "run this command from the bevy-gym repository root" >&2
              exit 2
            fi
            export NO_COLOR=true
            export TRUNK_OFFLINE=true
            web_address="''${BEVY_GYM_WEB_ADDRESS:-127.0.0.1}"
            exec trunk serve --config web/Trunk.toml --release --locked --address "$web_address" --port 8080 "$@"
          '';
        };

        webServeTailnet = pkgs.writeShellApplication {
          name = "bevy-gym-web-serve-tailnet";
          runtimeInputs = [
            webServe
            pkgs.tailscale
          ];
          text = ''
            BEVY_GYM_WEB_ADDRESS="$(tailscale ip -4)"
            export BEVY_GYM_WEB_ADDRESS
            exec bevy-gym-web-serve "$@"
          '';
        };

        webServeTailscaleHttps = pkgs.writeShellApplication {
          name = "bevy-gym-web-serve-tailscale-https";
          runtimeInputs = [ webServe ];
          text = ''
            export TRUNK_SERVE_WS_PROTOCOL=wss
            exec bevy-gym-web-serve "$@"
          '';
        };

        webCheck = pkgs.writeShellApplication {
          name = "bevy-gym-web-check";
          runtimeInputs = [
            rustToolchain
            pkgs.binaryen
            pkgs.playwright-test
            pkgs.python3
            pkgs.trunk
            pkgs.wasm-bindgen-cli
          ];
          text = ''
            if [[ ! -f web/index.html ]]; then
              echo "run this command from the bevy-gym repository root" >&2
              exit 2
            fi
            export NO_COLOR=true
            export PLAYWRIGHT_BROWSERS_PATH="${pkgs.playwright-driver.browsers}"
            # Pin EGL to Nix Mesa on Ubuntu runners as well as NixOS.
            # https://github.com/NixOS/nixpkgs/pull/510475
            export __EGL_VENDOR_LIBRARY_FILENAMES="${pkgs.mesa}/share/glvnd/egl_vendor.d/50_mesa.json"
            export LIBGL_ALWAYS_SOFTWARE=1
            export TRUNK_OFFLINE=true
            trunk build --config web/Trunk.toml --release --locked
            python -m http.server 4173 --bind 127.0.0.1 --directory dist >/dev/null 2>&1 &
            server_pid=$!
            trap 'kill "$server_pid"' EXIT
            playwright test --config web/playwright.config.cjs
          '';
        };

        browserRuntimeCheck = pkgs.writeShellApplication {
          name = "bevy-gym-browser-runtime-check";
          runtimeInputs = [ pkgs.playwright-test ];
          text = ''
            export NO_COLOR=true
            export PLAYWRIGHT_BROWSERS_PATH="${pkgs.playwright-driver.browsers}"
            # Pin EGL to Nix Mesa on Ubuntu runners as well as NixOS.
            # https://github.com/NixOS/nixpkgs/pull/510475
            export __EGL_VENDOR_LIBRARY_FILENAMES="${pkgs.mesa}/share/glvnd/egl_vendor.d/50_mesa.json"
            export LIBGL_ALWAYS_SOFTWARE=1
            playwright test --config gymnasium-web/playwright.config.cjs --grep 'browser runtime starts$' "$@"
          '';
        };

        droneBrowserCheck = pkgs.writeShellApplication {
          name = "bevy-gym-drone-browser-check";
          runtimeInputs = [
            rustToolchain
            pkgs.wasm-bindgen-cli
            pkgs.chromedriver
          ];
          text = ''
            export CARGO_TARGET_WASM32_UNKNOWN_UNKNOWN_RUNNER=wasm-bindgen-test-runner
            export CHROMEDRIVER="${lib.getExe pkgs.chromedriver}"
            export WASM_BINDGEN_TEST_WEBDRIVER_JSON="${
              pkgs.writeText "drone-webdriver.json" (
                builtins.toJSON {
                  "goog:chromeOptions".binary = lib.getExe pkgs.chromium;
                }
              )
            }"
            cargo test --locked --no-default-features --features robots,browser \
              --target wasm32-unknown-unknown --test drone_hover --test drone_recovery --test drone_parity "$@"
          '';
        };

        gymnasiumCheck = pkgs.writeShellApplication {
          name = "bevy-gym-browser-check";
          runtimeInputs = [
            rustToolchain
            pkgs.binaryen
            pkgs.playwright-test
            pkgs.python3
            pkgs.trunk
            pkgs.wasm-bindgen-cli
          ];
          text = ''
            export NO_COLOR=true
            export PLAYWRIGHT_BROWSERS_PATH="${pkgs.playwright-driver.browsers}"
            # Pin EGL to Nix Mesa on Ubuntu runners as well as NixOS.
            # https://github.com/NixOS/nixpkgs/pull/510475
            export __EGL_VENDOR_LIBRARY_FILENAMES="${pkgs.mesa}/share/glvnd/egl_vendor.d/50_mesa.json"
            export LIBGL_ALWAYS_SOFTWARE=1
            python tests/test_browser_assets.py
            trunk build --config gymnasium-web/Trunk.toml --release --locked --public-url /bevy-gym/
            serve_root=$(mktemp -d)
            ln -s "$PWD/site" "$serve_root/bevy-gym"
            python -m http.server 4174 --bind 127.0.0.1 --directory "$serve_root" >/dev/null 2>&1 &
            server_pid=$!
            trap 'kill "$server_pid"; rm -r "$serve_root"' EXIT
            playwright test --config gymnasium-web/playwright.config.cjs "$@"
          '';
        };

        wasmCheck = pkgs.stdenvNoCC.mkDerivation {
          name = "wasm-check";
          src = rustSource;

          cargoDeps = wasmRustPlatform.importCargoLock {
            lockFile = ./Cargo.lock;
          };

          nativeBuildInputs = [
            wasmRustPlatform.cargoSetupHook
            rustToolchain
          ];

          buildPhase = ''
            runHook preBuild
            export HOME="$TMPDIR"
            export CARGO_TARGET_DIR="$TMPDIR/target"
            cargo check --locked --target wasm32-unknown-unknown --package bevy-gym-web
            runHook postBuild
          '';

          installPhase = "touch $out";
        };
      in
      {
        packages = {
          inherit mujoco;
          web-dist = webDist;
        };

        apps = {
          web-serve = {
            type = "app";
            program = lib.getExe webServe;
          };
          web-serve-tailnet = {
            type = "app";
            program = lib.getExe webServeTailnet;
          };
          web-serve-tailscale-https = {
            type = "app";
            program = lib.getExe webServeTailscaleHttps;
          };
          browser-runtime-check = {
            type = "app";
            program = lib.getExe browserRuntimeCheck;
          };
          drone-browser-check = {
            type = "app";
            program = lib.getExe droneBrowserCheck;
          };
          gymnasium-check = {
            type = "app";
            program = lib.getExe gymnasiumCheck;
          };
          web-check = {
            type = "app";
            program = lib.getExe webCheck;
          };
        };

        formatter = treefmtEval.config.build.wrapper;

        checks = {
          cargo-check = mkCargoCheck "cargo-check" "cargo check --locked" renderNativeBuildInputs;
          cargo-check-render =
            mkCargoCheck "cargo-check-render" "cargo check --example cartpole --features render"
              renderNativeBuildInputs;
          cargo-clippy = mkCargoCheck "cargo-clippy" "cargo clippy --locked" (
            [ pkgs.clippy ] ++ renderNativeBuildInputs
          );
          wasm = wasmCheck;
          web-checkpoint-parity =
            mkCargoCheck "web-checkpoint-parity"
              "cargo test --locked checkpoint_round_trip_preserves_recurrent_policy --lib"
              renderNativeBuildInputs;
          web-dist = webDist;
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
            rustToolchain
            pkgs.pkg-config
            pkgs.python3
            mujoco
          ]
          ++ renderNativeBuildInputs
          ++ [
            treefmtEval.config.build.wrapper
            pkgs.deadnix
            pkgs.dprint
            pkgs.nixfmt
            pkgs.rumdl
            pkgs.statix
            pkgs.taplo
            pkgs.trunk
            pkgs.wasm-bindgen-cli
            pkgs.binaryen
          ];

          shellHook = ''
            export MUJOCO_DYNAMIC_LINK_DIR="${mujocoRuntimeLibraryPath}"

            if [ -n "''${LD_LIBRARY_PATH:-}" ]; then
              export LD_LIBRARY_PATH="${mujocoRuntimeLibraryPath}:${renderRuntimeLibraryPath}:$LD_LIBRARY_PATH"
            else
              export LD_LIBRARY_PATH="${mujocoRuntimeLibraryPath}:${renderRuntimeLibraryPath}"
            fi
          '';
        };
      }
    );
}
