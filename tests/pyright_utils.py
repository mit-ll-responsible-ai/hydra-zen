# Copyright (c) 2022 Massachusetts Institute of Technology
# SPDX-License-Identifier: MIT

import inspect
import json
import os
import re
import shutil
import subprocess
import tempfile
import textwrap
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from typing_extensions import Literal, NotRequired, TypedDict


class _Summary(TypedDict):
    filesAnalyzed: int
    errorCount: int
    warningCount: int
    informationCount: int
    timeInSec: float


class _LineInfo(TypedDict):
    line: int
    character: int


class _Range(TypedDict):
    start: _LineInfo
    end: _LineInfo


class _Diagnostic(TypedDict):
    file: str
    severity: Literal["error", "warning", "information"]
    message: str
    range: _Range
    rule: NotRequired[str]


class PyrightOutput(TypedDict):
    """The schema for the JSON output of a pyright scan"""

    version: str
    time: str
    generalDiagnostics: List[_Diagnostic]
    summary: _Summary


_found_path = shutil.which("pyright")
PYRIGHT_PATH = Path(_found_path) if _found_path else None
del _found_path


@contextmanager
def chdir():
    with tempfile.TemporaryDirectory() as tmpdirname:
        old_dir = os.getcwd()  # get current working directory (cwd)
        try:
            os.chdir(tmpdirname)  # change cwd to the temp-directory
            yield Path(tmpdirname)  # yields control to the test to be run
        finally:
            os.chdir(old_dir)


# `docstring_re` is derived from https://github.com/python/cpython/blob/main/Lib/doctest.py
# which is free for copy/reuse under GPL license
#
# This regular expression is used to find doctest examples in a
# string.  It defines two groups: `source` is the source code
# (including leading indentation and prompts); `indent` is the
# indentation of the first (PS1) line of the source code; and
# `want` is the expected output (including leading indentation).
docstring_re = re.compile(
    r"""
    # Source consists of a PS1 line followed by zero or more PS2 lines.
    (?P<source>
        (?:^(?P<indent> [ ]*) >>>    .*)    # PS1 line
        (?:\n           [ ]*  \.\.\. .*)*)  # PS2 lines
    \n?
    """,
    re.MULTILINE | re.VERBOSE,
)


def get_docstring_examples(doc: str) -> str:
    prefix = ">>> "

    # contains input lines of docstring examples with all indentation
    # and REPL markers removed
    src_lines: List[str] = []

    for source, indent in docstring_re.findall(doc):
        source: str
        indent: str
        for line in source.splitlines():
            src_lines.append(line[len(indent) + len(prefix) :])
        src_lines.append("")  # newline between blocks
    return "\n".join(src_lines)


def pyright_analyze(
    code_or_path,
    pyright_config: Optional[Dict[str, Any]] = None,
    *,
    path_to_pyright: Union[Path, None] = PYRIGHT_PATH,
    preamble: str = "",
    python_version: Optional[str] = None,
    report_unnecessary_type_ignore_comment: Optional[bool] = None,
    type_checking_mode: Optional[Literal["basic", "strict"]] = None,
    overwrite_config_ok: bool = False,
    scan_docstring: bool = False,
) -> PyrightOutput:
    """
    Scans a Python object (e.g., a function), docstring, or file(s) using pyright and
    returns a JSON summary of the scan.

    Some common pyright configuration options are exposed via this function for
    convenience; a full pyright JSON config can be specified to completely control
    the behavior of pyright.

    Parameters
    ----------
    func : SourceObjectType | str | Path
        A function, module-object, class, or method to scan. Or, a path to a file or
        directory to scan.

    pyright_config : None | dict[str, Any]
        A JSON configuration for pyright's settings [1]_.

    preamble: str, optional (default='')
        A "header" added to the source code that will be scanned. E.g., this can be
        useful for adding import statements.

    path_to_pyright: Path, keyword-only
        Path to the pyright executable. Defaults to `shutil.where('pyright')` if the
        executable can be found.

    python_version: Optional[str], keyword-only
        The version of Python used for this execution environment as a string in the
        format "M.m". E.g., "3.9" or "3.7"

    report_unnecessary_type_ignore_comment: Optional[bool], keyword-only
        If `True` specifying `# type: ignore` for an expression that would otherwise
        not result in an error will cause pyright to report an error.

    type_checking_mode: Optional[Literal["basic", "strict"]], keyword-only
        Modifies pyright's default settings for what it marks as a warning verses an
        error.

    overwrite_config_ok : bool, optional (default=False)
        If `True`, and if pyright configuration options are specified, this function
        will temporarily overwrite an existing pyrightconfig.json file if necessary.

        This option should be used with caution if tests using `pyright_analyze` are
        being run concurrently that impact the same config file.

    Returns
    -------
    PyrightOutput : TypedDict
        The JSON-decoded results of the scan [2]_.
            - version: str
            - time: str
            - generalDiagnostics: List[DiagnosticDict] (one entry per error/warning)
            - summary: SummaryDict

    References
    ----------
    .. [1] https://github.com/microsoft/pyright/blob/main/docs/configuration.md
    .. [2] https://github.com/microsoft/pyright/blob/main/docs/command-line.md#json-output

    Examples
    --------
    Here pyright will record an error when scan a function that attempts to add a
    string-annotated variable to an integer.

    >>> def f(x: str):
    ...     return 1 + x
    >>> pyright_analyze(f)
    {'version': '1.1.281',
     'time': '1669686515154',
     'generalDiagnostics': [{'file': 'C:\\Users\\RY26099\\AppData\\Local\\Temp\\12\\tmpcxc7erfq\\source.py',
       'severity': 'error',
       'message': 'Operator "+" not supported for types "Literal[1]" and "str"\n\xa0\xa0Operator "+" not supported for types "Literal[1]" and "str"',
       'range': {'start': {'line': 1, 'character': 11},
        'end': {'line': 1, 'character': 16}},
       'rule': 'reportGeneralTypeIssues'}],
     'summary': {'filesAnalyzed': 20,
      'errorCount': 1,
      'warningCount': 0,
      'informationCount': 0,
      'timeInSec': 0.319}}

    Whereas this function scans "clean".

    >>> def g(x: int) -> int:
    ...     return 1 + x
    >>> pyright_analyze(g)
    {'version': '1.1.281',
     'time': '1669686578833',
     'generalDiagnostics': [],
     'summary': {'filesAnalyzed': 20,
      'errorCount': 0,
      'warningCount': 0,
      'informationCount': 0,
      'timeInSec': 0.29}}

    All imports must occur within the context of the scanned-object, or the imports can
    be specified in a preamble. For example, consider the following

    >>> import math  # import statement is not be in scope of `f`
    >>> def f():
    ...     math.acos(1)
    >>> pyright_analyze(f)["summary"]["errorCount"]
    1

    We can add a 'preamble' do that the `math` module is imported.

    >>> pyright_analyze(f, preamble="import math")["summary"]["errorCount"]
    0
    """
    TMP_CONFIG_HEADER = r"// temp config written by pyright_analyze" + "\n"

    if path_to_pyright is None:
        raise ModuleNotFoundError(
            "`pyright` was not found. It may need to be installed."
        )
    if not path_to_pyright.is_file():
        raise FileNotFoundError(
            f"`path_to_pyright – {path_to_pyright} – doesn't exist."
        )
    if not pyright_config:
        pyright_config = {}

    if python_version is not None:
        pyright_config["pythonVersion"] = python_version

    if report_unnecessary_type_ignore_comment is not None:
        pyright_config[
            "reportUnnecessaryTypeIgnoreComment"
        ] = report_unnecessary_type_ignore_comment

    if type_checking_mode is not None:
        pyright_config["typeCheckingMode"] = type_checking_mode

    if scan_docstring and (
        isinstance(code_or_path, (Path, str))
        or getattr(code_or_path, "__doc__") is None
    ):
        raise ValueError(
            "`scan_docstring=True` can only be specified when `code_or_path` is an "
            "object with a `__doc__` attribute that returns a string."
        )

    if not isinstance(code_or_path, (str, Path)):
        if preamble and not preamble.endswith("\n"):
            preamble = preamble + "\n"
        if not scan_docstring:
            source = preamble + textwrap.dedent((inspect.getsource(code_or_path)))
        else:
            source = preamble + get_docstring_examples(code_or_path.__doc__)
            print(source)
    else:
        source = None

    with chdir():
        cwd = Path.cwd()
        if source is not None:
            file_ = cwd / "source.py"
            file_.write_text(source)
        else:
            file_ = Path(code_or_path).absolute()
            assert (
                file_.exists()
            ), f"Specified path {file_} does not exist. Cannot be scanned by pyright."

        config_path = (
            file_.parent if file_.is_file() else file_
        ) / "pyrightconfig.json"

        if not overwrite_config_ok and config_path.exists() and pyright_config:
            raise ValueError(
                f"pyright config located at {config_path.absolute()} would be "
                "temporarily overwritten by this test. To permit this, specify "
                "`overwrite_config_ok=True`."
            )

        old_pyright_config = config_path.read_text() if config_path.exists() else None
        skip_config_delete = False

        if old_pyright_config and old_pyright_config.startswith(TMP_CONFIG_HEADER):
            # In the case where pyright_analyze is being run concurrently and
            # encountered a temp config that pyright_analyze wrote, we ought not
            # restore that temp config as this could inadvertently overwrite the
            # *correct* config that was restored during the execution of this function
            skip_config_delete = True
            old_pyright_config = None

        if pyright_config:
            config_path.write_text(TMP_CONFIG_HEADER + json.dumps(pyright_config))

        proc = subprocess.run(
            [str(path_to_pyright.absolute()), str(file_.absolute()), "--outputjson"],
            cwd=file_.parent,
            encoding="utf-8",
            text=True,
            capture_output=True,
        )
        try:
            return json.loads(proc.stdout)
        except Exception:
            print(proc.stdout)
            raise
        finally:
            if not skip_config_delete and config_path.is_file():
                os.remove(config_path)
            if old_pyright_config is not None:
                config_path.write_text(old_pyright_config)


def list_error_messages(results: PyrightOutput) -> List[str]:
    """A convenience function that returns a list of error messages reported by pyright."""
    return [
        e["message"] for e in results["generalDiagnostics"] if e["severity"] == "error"
    ]
