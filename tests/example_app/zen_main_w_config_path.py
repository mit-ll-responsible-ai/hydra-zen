# Copyright (c) 2025 Massachusetts Institute of Technology
# SPDX-License-Identifier: MIT

from pathlib import Path

from hydra_zen import zen

cwd = Path.cwd()


def main(zen_cfg):
    print(zen_cfg)


if __name__ == "__main__":
    zen(main).hydra_main(config_name="config", config_path=".")
