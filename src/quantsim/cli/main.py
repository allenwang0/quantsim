"""quantsim command-line entry point: quantsim <command> [options]."""

import importlib
import sys

COMMANDS = {
    "backtest": (
        "quantsim.cli.backtest",
        "Run an event-driven, vectorized, sweep or walk-forward backtest",
    ),
    "wfo": (
        "quantsim.cli.wfo",
        "Walk-forward optimization or parameter sweep on a signal function",
    ),
    "paper": (
        "quantsim.cli.paper",
        "Paper trading through Alpaca, or simulated fills without keys",
    ),
    "bootstrap": (
        "quantsim.cli.bootstrap",
        "Download price and macro history into the local database",
    ),
    "stress": ("quantsim.cli.stress", "Run a strategy over historical stress periods"),
}


def usage() -> str:
    lines = ["usage: quantsim <command> [options]", "", "commands:"]
    lines += [f"  {name:<10} {help_text}" for name, (_, help_text) in COMMANDS.items()]
    lines += ["", "Run 'quantsim <command> --help' for a command's options."]
    return "\n".join(lines)


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help"):
        print(usage())
        return 0
    command, rest = argv[0], argv[1:]
    if command not in COMMANDS:
        print(f"quantsim: unknown command {command!r}\n\n{usage()}", file=sys.stderr)
        return 2
    module = importlib.import_module(COMMANDS[command][0])
    result = module.main(rest)
    return result if isinstance(result, int) else 0


if __name__ == "__main__":
    sys.exit(main())
