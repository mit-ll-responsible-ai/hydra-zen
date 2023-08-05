# Copyright (c) 2023 Massachusetts Institute of Technology
# SPDX-License-Identifier: MIT

import inspect
import json
import os
import re
import shutil
import subprocess
import tempfile
import textwrap
import time
from collections import Counter, defaultdict
from contextlib import contextmanager
from pathlib import Path
from typing import Any, DefaultDict, Dict, List, Optional, Union

from typing_extensions import Literal, NotRequired, TypedDict


def notebook_to_py_text(path_to_nb: Path) -> str:
    try:
        import jupytext
    except ImportError:
        raise ImportError(
            "`jupytext` must be installed in order to run pyright on jupyter notebooks."
        )
    ntbk = jupytext.read(path_to_nb, fmt="ipynb")
    return jupytext.writes(ntbk, fmt=".py")


class Summary(TypedDict):
    filesAnalyzed: int
    errorCount: int
    warningCount: int
    informationCount: int
    timeInSec: float


class LineInfo(TypedDict):
    line: int
    character: int


class Range(TypedDict):
    start: LineInfo
    end: LineInfo


class Diagnostic(TypedDict):
    file: str
    severity: Literal["error", "warning", "information"]
    message: str
    range: Range
    rule: NotRequired[str]


class PyrightOutput(TypedDict):
    """The schema for the JSON output of a pyright scan"""

    version: str
    time: str
    generalDiagnostics: List[Diagnostic]
    summary: Summary


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


def rst_to_code(src: str) -> str:
    """
    Consumes rst-formatted text like::

       lopsem est decorum

       .. code-block:: python
          :caption: blah
          :name: bark bark

          import math
          x = 1+1

       foorbarius isbarfooium

       .. code-block:: pycon
          :caption: blah

          >>> print("hi")
          hi
          >>> 2+1

    and returns the string::

       '''
       import math
       x = 1+1


       print("hi")

       2+1
       '''

    """
    block: Optional[List[str]] = None  # lines in code block
    indentation: Optional[str] = None  # leading whitespace before .. code-block
    preamble: Optional[str] = None  # python or pycon
    n = -float("inf")  # line no in code block
    blocks: List[str] = []  # respective code blocks, each ready for processing

    def add_block(
        block: Optional[List[str]],
        preamble: Optional[str],
        blocks: List[str],
    ):
        if block:
            block_str = "\n".join(block) + "\n"
            assert preamble
            if "pycon" in preamble:
                blocks.append(get_docstring_examples(block_str))
            else:
                blocks.append(textwrap.dedent(block_str))

    for line in src.splitlines():
        n += 1
        # 0 <= n: if within code block

        if line.strip().startswith(".. code-block:: py"):
            # Entering python/pycon code block
            add_block(block, preamble, blocks)
            n = -1
            block = []
            indentation = line.split("..")[0] + " " * 3
            preamble = line.split("::")[-1].strip()
            continue

        if n < 0:
            # outside of code block
            continue

        assert indentation is not None
        assert block is not None

        if not (line.startswith(indentation) or not line.strip()):
            # encountering non-empty line that isn't within
            # minimum indentation leaves the code block
            add_block(block, preamble, blocks)
            block = None
            n = -float("inf")
            continue

        if n == 0:
            # first line of code block is either empty or a directive
            stripped = line.strip()
            if not stripped:
                continue

            if line.startswith(indentation):
                if stripped.startswith(":"):
                    n = -1
                    # skip directive, act as if we are at top of code block
                    continue

        block.append(line)

    add_block(block, preamble, blocks)
    return "\n".join(blocks)


def pyright_analyze(
    *code_objs_and_or_paths: Any,
    pyright_config: Optional[Dict[str, Any]] = None,
    scan_docstring: bool = False,
    path_to_pyright: Union[Path, None] = PYRIGHT_PATH,
    preamble: str = "",
    python_version: Optional[str] = None,
    report_unnecessary_type_ignore_comment: Optional[bool] = None,
    type_checking_mode: Optional[Literal["basic", "strict"]] = None,
) -> List[PyrightOutput]:
    r"""
    Scan one or more Python objects, docstrings, or files with pyright.

    Note that it is much faster to call this function once for N objects than N
    times - once for each object.

    The following file formats are supported: `.py`, `.rst`, and `.ipynb`

    Some common pyright configuration options are exposed via this function for
    convenience; a full pyright JSON config can be specified to completely control
    the behavior of pyright.

    This function requires that pyright is installed and can be run from the command
    line [1]_.

    Parameters
    ----------
    *code_objs_and_or_paths : SourceObjectType | Path
        A function, module-object, class, or method to scan. Or, a path to a file
        to scan. Supported file formats are `.py`, `.rst`, and `.ipynb`.

        Specifying a directory is not permitted.

    pyright_config : None | dict[str, Any]
        A JSON configuration for pyright's settings [2]_.

    scan_docstring : bool, optional (default=False), keyword-only
        If `True` pyright will scan the docstring examples of any specified code
        objects, rather than the code object itself.

        Example code blocks are expected to have the doctest format [3]_.

    path_to_pyright : Path, optional, keyword-only
        Path to the pyright executable (see installation instructions: [4]_).
        Defaults to `shutil.where('pyright')` if the executable can be found.

    preamble : str, optional (default=''), keyword-only
        A "header" added to the source code that will be scanned. E.g., this can be
        useful for adding import statements.

    python_version : Optional[str], keyword-only
        The version of Python used for this execution environment as a string in the
        format "M.m". E.g., "3.9" or "3.7".

    report_unnecessary_type_ignore_comment : Optional[bool], keyword-only
        If `True` specifying `# type: ignore` for an expression that would otherwise
        not result in an error will cause pyright to report an error.

    type_checking_mode : Optional[Literal["basic", "strict"]], keyword-only
        Modifies pyright's default settings for what it marks as a warning verses an
        error. Defaults to 'basic'.

    Returns
    -------
    List[Dict[str, Any]]  (In one-to-one correspondence with `code_objs_and_or_paths`)
        A JSON-decoded result of the scan for each of the inputs [3]_.
            - version: str
            - time: str
            - generalDiagnostics: List[DiagnosticDict] (one entry per error/warning)
            - summary: SummaryDict

        See Notes for more details.

    Notes
    -----
    When supplying a single .rst file, code blocks demarcated by
    `.. code-block:: py[thon,con]` are parsed and used to populate a single temporary
    .py file that pyright will scan.

    `SummaryDict` consists of:
        - filesAnalyzed: int
        - errorCount: int
        - warningCount: int
        - informationCount: int
        - timeInSec: float

    `DiagnosticDict` consists of:
        - file: str
        - severity: Literal["error", "warning", "information"]
        - message: str
        - range: _Range
        - rule: NotRequired[str]

    References
    ----------
    .. [1] https://github.com/microsoft/pyright/blob/aad650ec373a9894c6f13490c2950398095829c6/README.md#command-line
    .. [2] https://github.com/microsoft/pyright/blob/main/docs/configuration.md
    .. [3] https://docs.python.org/3/library/doctest.html
    .. [4] https://github.com/microsoft/pyright/blob/main/docs/command-line.md#json-output

    Examples
    --------
    Here pyright will record an error when scan a function that attempts to add a
    string-annotated variable to an integer.

    >>> def f(x: str):
    ...     return 1 + x
    >>> pyright_analyze(f)
    {'version': '1.1.281',
     'time': '1669686515154',
     'generalDiagnostics': [{'file': 'source.py',
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

    Scanning a function's docstring.

    >>> def plus_1(x: int):
    ...     '''
    ...     Examples
    ...     --------
    ...     >>> from mylib import plus_1
    ...     >>> plus_1('2')  # <- pyright_analyze will catch typo (str instead of int)
    ...     3
    ...     '''
    ...     return x + 1
    >>> pyright_analyze(plus_1, scan_docstring=True)["summary"]["errorCount"]
    1

    Fixing the docstring issue

    >>> def plus_1(x: int):
    ...     '''
    ...     Examples
    ...     --------
    ...     >>> from mylib import plus_1
    ...     >>> plus_1(2)
    ...     3
    ...     '''
    ...     return x + 1
    >>> pyright_analyze(plus_1, scan_docstring=True)["summary"]["errorCount"]
    0
    """
    if path_to_pyright is None:  # pragma: no cover
        raise ModuleNotFoundError(
            "`pyright` was not found. It may need to be installed."
        )
    if not path_to_pyright.is_file():
        raise FileNotFoundError(
            f"`path_to_pyright` – {path_to_pyright} – doesn't exist."
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

    sources: List[Optional[str]] = []
    for code_or_path in code_objs_and_or_paths:
        if scan_docstring and (
            isinstance(code_or_path, (Path, str))
            or getattr(code_or_path, "__doc__") is None
        ):
            raise ValueError(
                "`scan_docstring=True` can only be specified when `code_or_path` is an "
                "object with a `__doc__` attribute that returns a string."
            )

        if isinstance(code_or_path, str):
            code_or_path = Path(code_or_path)

        if isinstance(code_or_path, Path):
            code_or_path = code_or_path.resolve()
            if code_or_path.is_dir():
                raise ValueError(
                    f"Paths must specify individual files, not directories. Got: {str(code_or_path)}"
                )
            if not code_or_path.exists():
                raise FileNotFoundError(
                    f"Specified path {code_or_path} does not exist. Cannot be scanned by pyright."
                )
            if code_or_path.suffix == ".rst":
                source = rst_to_code(code_or_path.read_text("utf-8"))
            elif code_or_path.suffix == ".ipynb":
                source = notebook_to_py_text(code_or_path)
            elif code_or_path.is_file() and code_or_path.suffix != ".py":
                raise ValueError(
                    f"{code_or_path}: File type {code_or_path.suffix} not supported by "
                    "`pyright_analyze`."
                )
            else:
                source = None
        else:
            if preamble and not preamble.endswith("\n"):
                preamble = preamble + "\n"
            if not scan_docstring:
                source = preamble + textwrap.dedent((inspect.getsource(code_or_path)))
            else:
                docstring = inspect.getdoc(code_or_path)
                assert docstring is not None
                source = preamble + get_docstring_examples(docstring)
        sources.append(source)

    with chdir():
        cwd = Path.cwd()

        for n, (source, code_or_path) in enumerate(
            zip(sources, code_objs_and_or_paths)
        ):
            target_dir = cwd / str(n)
            target_dir.mkdir()

            if source is not None:
                file_ = target_dir / f"{getattr(code_or_path, '__name__', 'source')}.py"
                file_.write_text(source, encoding="utf-8")
            else:
                file_ = Path(code_or_path).absolute()
                if file_ != cwd:
                    file_ = shutil.copy(file_, target_dir / file_.name)

        config_path = cwd / "pyrightconfig.json"

        if pyright_config:
            config_path.write_text(json.dumps(pyright_config))

        proc = subprocess.run(
            [str(path_to_pyright.absolute()), str(cwd.absolute()), "--outputjson"],
            cwd=cwd,
            encoding="utf-8",
            text=True,
            capture_output=True,
        )
        try:
            time.sleep(0.1)  # maybe fixes JSONDecoder errors
            scan: PyrightOutput = json.loads(proc.stdout)
        except Exception as e:  # pragma: no cover
            print(proc.stdout)
            raise e

    out = scan["generalDiagnostics"]

    diagnostics_by_file: DefaultDict[int, List[Diagnostic]] = defaultdict(list)

    for item in out:
        file_path = Path(item["file"])
        file_index = int(file_path.parent.name)
        diagnostic = item.copy()
        diagnostic["file"] = file_path.name
        diagnostics_by_file[file_index].append(diagnostic)

    results: List[PyrightOutput] = []

    for n in range(len(code_objs_and_or_paths)):
        severities = Counter(d["severity"] for d in diagnostics_by_file[n])
        summary = Summary(
            filesAnalyzed=1,
            errorCount=severities["error"],
            warningCount=severities["warning"],
            informationCount=severities["information"],
            timeInSec=scan["summary"]["timeInSec"],
        )
        results.append(
            PyrightOutput(
                version=scan["version"],
                time=scan["time"],
                generalDiagnostics=diagnostics_by_file[n],
                summary=summary,
            )
        )
    return results


def list_error_messages(results: PyrightOutput) -> List[str]:
    """A convenience function that returns a list of error messages reported by pyright."""
    return [
        f"(line start) {e['range']['start']['line']}: {e['message']}"
        for e in results["generalDiagnostics"]
        if e["severity"] == "error"
    ]
