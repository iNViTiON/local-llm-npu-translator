{
  description = "Local NPU translation backend (OpenVINO, Intel AI Boost)";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";

  outputs = { self, nixpkgs }:
    let
      system = "x86_64-linux";
      pkgs = import nixpkgs { inherit system; config.allowUnfree = true; };

      # OpenVINO >= 2026.0 loads the NPU compiler from its own libs dir; it is not in the pip wheel.
      npuCompiler = pkgs.stdenv.mkDerivation {
        pname = "intel-npu-compiler";
        version = "2026.38-rc1";
        src = pkgs.fetchurl {
          url = "https://github.com/openvinotoolkit/npu_compiler/releases/download/npu_ud_2026_38_rc1/l_vpux_compiler_l0_linux_ubuntu_24_04-7_10_0-Release_dyntbb_postcommit_cid_0b38f7d42113ff329ac2bdd33583d123de4ccf2f_260902_1709.tar.gz";
          hash = "sha256-hdAS06yitjDMNJcs5ABoBOs3e0PF6MNn6UpAjK7cvsM=";
        };
        sourceRoot = ".";
        nativeBuildInputs = [ pkgs.autoPatchelfHook ];
        buildInputs = with pkgs; [ stdenv.cc.cc.lib level-zero zlib zstd tbb_2022 ];
        installPhase = ''
          mkdir -p $out/lib
          cp cid/lib/*.so $out/lib/
        '';
      };

      runtimeLibs = with pkgs; [ stdenv.cc.cc.lib zlib ocl-icd tbb_2022 level-zero ];
    in {
      # The server as a systemd user service (runs from a checkout; see nix/module.nix).
      nixosModules.default = import ./nix/module.nix;

      packages.${system}.npu-compiler = npuCompiler;

      devShells.${system}.default = pkgs.mkShell {
        packages = with pkgs; [ python312 uv git curl jq ];
        shellHook = ''
          # NPU user-mode driver + level-zero must match the kernel firmware: use the system ones.
          export LD_LIBRARY_PATH="/run/opengl-driver/lib:${pkgs.lib.makeLibraryPath runtimeLibs}''${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
          export UV_PYTHON="${pkgs.python312}/bin/python3.12"
          export UV_PYTHON_DOWNLOADS=never
          export HF_HOME="$PWD/.cache/hf"
          # Expose the NPU compiler next to the OpenVINO NPU plugin (refreshed on every shell entry).
          for d in .venv/lib/python3.12/site-packages/openvino/libs; do
            [ -d "$d" ] && ln -sf ${npuCompiler}/lib/*.so "$d"/
          done
        '';
      };
    };
}
