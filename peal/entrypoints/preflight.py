"""Check run configs before spending GPU time on them.

Usage::

    python preflight.py <config.yaml> [<config.yaml> ...]
    python preflight.py --script reproduction_scripts/reproduce_sce_results.sh
    python preflight.py --script <script> --quiet     # only the failures
    python preflight.py --no-existence configs/**/*.yaml   # CI, no $PEAL_RUNS

Exit status is 0 when everything checked can run, 1 otherwise, so a script can
guard itself with `python preflight.py --script "$0" || exit 1`.
"""

import argparse
import sys

from peal.preflight import check_config, check_script, check_script_executables


def report(name, problems, notes, quiet=False):
    """Print one config's verdict and say whether it failed.

    Parameters
    ----------
    name : str
        Label for the line, normally the config's file name.
    problems : list of str
        Reasons the config cannot run; any entry means failure.
    notes : list of str
        Informational remarks, printed only for passing configs.
    quiet : bool, optional
        Suppress the output for passing configs.

    Returns
    -------
    bool
        ``True`` when ``problems`` is non-empty.
    """
    if problems:
        print(f"FAIL  {name}")
        for p in problems:
            print(f"          problem: {p}")
    elif not quiet:
        print(f"ok    {name}")
        for n in notes:
            print(f"          note:    {n}")
    return bool(problems)


def main():
    """Check the configs named on the command line and/or in ``--script``.

    Delegates to ``check_config`` and ``check_script`` in
    ``peal/preflight.py``.

    Returns
    -------
    int
        Exit status: 0 when every checked config can run, 1 otherwise.
    """
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("configs", nargs="*")
    parser.add_argument(
        "--script", help="check every --config this shell script passes"
    )
    parser.add_argument("--quiet", action="store_true", help="print only the failures")
    parser.add_argument(
        "--require-tracked",
        action="store_true",
        help="also fail when a file the script invokes exists but is not tracked\n"
        "by git, i.e. a clone of the repository would not get it. Use this for a\n"
        "release check; the default only asks whether the script can run here",
    )
    parser.add_argument(
        "--no-existence",
        action="store_true",
        help="skip the on-disk checks; use in a commit hook or CI, where\n"
        "$PEAL_RUNS is not mounted",
    )
    args = parser.parse_args()

    failed = 0
    checked = 0
    if args.script:
        results = check_script(args.script, not args.no_existence)
        for cfg, problems, notes in results:
            checked += 1
            failed += report(cfg.split("/")[-1], problems, notes, args.quiet)
        # A script also *runs* files, and a config-only check cannot see when one
        # of those is missing or was never committed.
        if not args.no_existence:
            problems, notes = check_script_executables(
                args.script, args.require_tracked
            )
            checked += 1
            failed += report(
                args.script.split("/")[-1] + " (files it runs)",
                problems,
                notes,
                args.quiet,
            )
    for cfg in args.configs:
        problems, notes = check_config(cfg, not args.no_existence)
        checked += 1
        failed += report(cfg.split("/")[-1], problems, notes, args.quiet)

    if not checked:
        parser.error("nothing to check: pass config paths or --script")
    print(f"\n{checked} config(s) checked, {failed} cannot run as committed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
