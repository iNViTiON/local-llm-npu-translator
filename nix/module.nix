# NixOS module: the translation server as a systemd user service on localhost:11434.
#
# It runs from a checkout of this repository through its run.sh (the flake dev shell plus the uv
# venv), because the converted model, the venv and the compile cache live in that checkout; see
# the README for the one-time model conversion. Started at login, restarted on failure. The
# server frees the model after `idleUnloadSeconds` idle and reloads it in ~2 s.
#
# run.sh calls `nix develop`, so flakes must be enabled for the user.
{
  config,
  lib,
  pkgs,
  ...
}:

let
  cfg = config.services.npu-translator;
in
{
  options.services.npu-translator = {
    enable = lib.mkEnableOption "the NPU translation server as a user service";

    user = lib.mkOption {
      type = lib.types.str;
      example = "alice";
      description = "User account that runs the server (the service only starts for this user).";
    };

    directory = lib.mkOption {
      type = lib.types.str;
      default = "${config.users.users.${cfg.user}.home}/local-llm-npu-translator";
      defaultText = lib.literalExpression ''"''${config.users.users.''${user}.home}/local-llm-npu-translator"'';
      description = "Checkout of this repository, with the converted model (its run.sh starts the server).";
    };

    devices = lib.mkOption {
      type = lib.types.enum [
        "npu"
        "all"
      ];
      default = "npu";
      description = ''
        Devices the server may use. `npu`: the NPU only. `all`: NPU, GPU and CPU are served; a
        plain model name still runs on the NPU, and clients pick the others with names like
        `translategemma:4b@GPU`. Replicas load on first use and cost RAM only while loaded.
      '';
    };

    idleUnloadSeconds = lib.mkOption {
      type = lib.types.ints.unsigned;
      default = 300;
      description = "Free the model's memory after this many idle seconds; 0 keeps it loaded.";
    };
  };

  config = lib.mkIf cfg.enable {
    systemd.user.services.npu-translator = {
      description = "TranslateGemma on the NPU, Ollama-compatible API on localhost:11434";
      wantedBy = [ "default.target" ];
      unitConfig.ConditionUser = cfg.user;

      path = with pkgs; [
        bash
        coreutils
        git # nix develop reads the project flake from its git checkout
        config.nix.package
      ];

      serviceConfig = {
        ExecStart = lib.escapeShellArgs (
          [
            "${cfg.directory}/run.sh"
            "--idle-unload"
            (toString cfg.idleUnloadSeconds)
          ]
          ++ (
            if cfg.devices == "all" then
              [
                "--device"
                "NPU,GPU,CPU"
                "--default-device"
                "NPU"
              ]
            else
              [
                "--device"
                "NPU"
              ]
          )
        );
        Restart = "on-failure";
        RestartSec = 5;
        SuccessExitStatus = 143; # SIGTERM on stop, passed back through `nix develop`
      };
    };
  };
}
