"""PEAL's logging: one ``peal`` logger tree in place of ``print``.

Until 2026-09-25 PEAL reported progress with about 950 ``print`` calls. That
works in a research checkout and is unusable inside another application: the
output cannot be routed, filtered or silenced. Every module now does

    from peal.log import get_logger
    _log = get_logger(__name__)
    _log.info("...")

and writes to a child of the ``peal`` logger.

**Defaults keep the old behaviour on purpose.** The first :func:`get_logger`
call attaches a handler to the ``peal`` logger that writes bare messages to
``sys.stdout`` at ``INFO``, so a run's console output and every ``> log.txt``
redirection look exactly as they did with ``print``. A host application that
wants something else can, before importing anything else from PEAL:

* set ``PEAL_LOG_LEVEL`` (e.g. ``WARNING``) to quieten it,
* set ``PEAL_LOG_QUIET=1`` to attach no handler at all and route the ``peal``
  logger through its own logging configuration,
* or call :func:`configure` itself.
"""

import logging
import os
import sys

__all__ = ["get_logger", "configure", "ROOT_NAME"]

ROOT_NAME = "peal"
_CONFIGURED = False


class _StdoutHandler(logging.StreamHandler):
    """A stream handler that looks up ``sys.stdout`` at emit time.

    ``logging.StreamHandler(sys.stdout)`` captures the *object* that is
    ``sys.stdout`` when the handler is created. Anything that swaps the stream
    afterwards (pytest's capture, ``contextlib.redirect_stdout``, a notebook
    kernel) would then be bypassed, which ``print`` never was. Resolving the
    stream lazily keeps that property of ``print``.
    """

    def __init__(self):
        super().__init__(stream=None)

    @property
    def stream(self):
        return sys.stdout

    @stream.setter
    def stream(self, value):  # StreamHandler.__init__ assigns; ignore it
        pass


def configure(level=None, stream=None, fmt="%(message)s", force=False):
    """Attach PEAL's default handler to the ``peal`` logger.

    Parameters
    ----------
    level : str or int, optional
        Logging level; defaults to ``$PEAL_LOG_LEVEL`` or ``INFO``.
    stream : file-like, optional
        Where messages go; defaults to ``sys.stdout`` so that existing
        ``python run.py > log.txt`` invocations keep capturing everything.
    fmt : str
        Handler format; the default is the bare message, i.e. what ``print``
        produced.
    force : bool
        Reconfigure even if a handler is already attached.
    """
    global _CONFIGURED
    root = logging.getLogger(ROOT_NAME)
    if _CONFIGURED and not force:
        return root
    if force:
        for handler in list(root.handlers):
            root.removeHandler(handler)
    if os.environ.get("PEAL_LOG_QUIET", "0") != "1":
        handler = _StdoutHandler() if stream is None else logging.StreamHandler(stream)
        handler.setFormatter(logging.Formatter(fmt))
        root.addHandler(handler)
        # Do not also bubble up to the root logger: a host application with
        # its own root handler would otherwise see every message twice.
        root.propagate = False
    level = level or os.environ.get("PEAL_LOG_LEVEL", "INFO")
    root.setLevel(level.upper() if isinstance(level, str) else level)
    _CONFIGURED = True
    return root


def get_logger(name=None):
    """Return a logger under the ``peal`` tree, configuring it on first use.

    Parameters
    ----------
    name : str, optional
        Usually ``__name__``; anything not under ``peal.`` is nested beneath
        it so one setting governs all of PEAL.
    """
    configure()
    if not name:
        return logging.getLogger(ROOT_NAME)
    if name != ROOT_NAME and not name.startswith(ROOT_NAME + "."):
        name = f"{ROOT_NAME}.{name}"
    return logging.getLogger(name)
