# Copyright (c) 2022 Massachusetts Institute of Technology

# Usage:
#
# python project_tooling/add_header.py
#

import fileinput
import os
import os.path as path
from pathlib import Path

OLD_HEADER = "# Copyright (c) 2021 Massachusetts Institute of Technology"
NEW_HEADER = "# Copyright (c) 2021 Massachusetts Institute of Technology\n# SPDX-License-Identifier: MIT"
EXCLUDED = {"_version.py", "versioneer.py"}


def is_safe_dir(path: Path):
    path = path.resolve()
    if not any(p in {"hydra_utils", "hydra-zen", "hydra_zen"} for p in path.parts):
        raise ValueError(
            f"Dangerous! You are running a script that can overwrite files in an unexpected directory: {path}"
        )


def get_src_files(dirname):
    dirname = Path(dirname)
    is_safe_dir(dirname)

    if dirname.is_file():
        if dirname.name.endswith(".py") and dirname.name not in EXCLUDED:
            yield dirname

    else:
        for cur, _dirs, files in os.walk(dirname):
            cur = Path(cur)
            if any(p.startswith((".", "__")) for p in cur.parts):
                # exclude hidden/meta dirs
                continue

            for f in files:
                if f in EXCLUDED:
                    continue
                if f.endswith(".py"):
                    yield path.join(cur, f)


def add_headers(files):
    # this needs to be modified to be able to replace multi-line headers!
    for line in fileinput.input(files, inplace=True):
        if fileinput.isfirstline():
            if NEW_HEADER in line:
                print(line, end="")
            elif OLD_HEADER and OLD_HEADER in line:
                print(NEW_HEADER, end="\n")
            else:
                print(NEW_HEADER, end="\n\n")
                print(line, end="")
        else:
            print(line, end="")


if __name__ == "__main__":
    add_headers(get_src_files("./setup.py"))
    add_headers(get_src_files("./src/hydra_zen/"))
    add_headers(get_src_files("./tests/"))
