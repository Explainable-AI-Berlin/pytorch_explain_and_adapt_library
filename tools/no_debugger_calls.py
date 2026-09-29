"""Fail if a tracked Python file contains an executable debugger call.

`breakpoint()` and `pdb.set_trace()` are invisible in review and harmless when
someone runs a notebook, but in a batch run they raise `BdbQuit` and discard
whatever the job had computed. Occurrences inside string literals are ignored,
which is why this walks the syntax tree rather than grepping.

Invocation (from the pre-commit hook or by hand)::

    python tools/no_debugger_calls.py <file.py> [<file.py> ...]

Prints ``<path>:<line>: debugger call in library code`` per hit and exits
with status 1 when any file has one or does not parse; nothing is written.
"""

import ast
import os
import sys


def debugger_calls(path):
    """Find the executable debugger calls in one Python file.

    Parameters
    ----------
    path : str
        The file to parse.

    Returns
    -------
    list of int
        Line numbers of ``breakpoint()`` / ``*.set_trace()`` expression
        statements, or ``[-1]`` when the file does not parse.
    """
    try:
        tree = ast.parse(open(path).read())
    except SyntaxError as error:
        print(f"{path}: does not parse: {error}")
        return [-1]
    hits = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Expr) or not isinstance(node.value, ast.Call):
            continue
        func = node.value.func
        if isinstance(func, ast.Name) and func.id == "breakpoint":
            hits.append(node.lineno)
        elif isinstance(func, ast.Attribute) and func.attr == "set_trace":
            hits.append(node.lineno)
    return hits


def main(paths):
    """Check every path and report the hits.

    Parameters
    ----------
    paths : list of str
        Python files to check.

    Returns
    -------
    int
        Exit status: 1 when any debugger call (or parse error) was found,
        else 0.
    """
    bad = 0
    for path in paths:
        # A path listed by `git ls-files` can already be deleted in the working
        # tree (the index lags behind until the deletion is committed); there
        # is nothing to check in it.
        if not os.path.exists(path):
            continue
        for line in debugger_calls(path):
            if line > 0:
                print(f"{path}:{line}: debugger call in library code")
            bad += 1
    if bad:
        print(f"\n{bad} debugger call(s). Raise the error instead, or delete the line.")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
