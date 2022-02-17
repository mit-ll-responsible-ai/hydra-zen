Thanks for your interest in contributing to hydra-zen! Please read 
through the following resources before you begin working on any contributions to this 
code base.


- [Installing hydra-zen for development](#installing-hydra-zen-for-development)
- [Pre-Commit Hooks (Required)](#pre-commit-hooks-required)
  - [What does this do?](#what-does-this-do)
- [Running Our Tests Manually](#running-our-tests-manually)
- [Running Tests Using `tox`](#running-tests-using-tox)
  - [Measuring Code Coverage](#measuring-code-coverage)
- [Validating Type Correctness](#validating-type-correctness)
  

## Installing hydra-zen for development

Install the toolkit along with its test dependencies; checkout the repo, navigate to its top level and run

```shell
pip install -e .
```

the `-e` option ensures that any changes that you make to the project's source code will be reflected in your local install – you need not reinstall the package in order for your modifications to take effect.

If your contributions involve changes to our support for NumPy, PyTorch, PyTorch-Lightning, JAX, pydantic, or beartype then you will need to install those dependencies as well, in order for our tests-against-third-parties to run locally in your environment.


## Pre-Commit Hooks (Required)

We provide contributors with pre-commit hooks, which will apply auto-formatters and 
linters to your code before your commit takes effect. You must install these in order to contribute to the repo.

First install pre-commit in your Python environment. Run:

```console
pip install pre-commit
```

Then, in the top-level of the `hydra_zen` repo, run:

```console
pre-commit install
pre-commit run
```

Great! You can read more about pre-commit hooks in general here: https://pre-commit.com/

### What does this do?

Our pre-commit hooks run the following auto-formatters on all commits:
- [black](https://black.readthedocs.io/en/stable/)
- [isort](https://pycqa.github.io/isort/)

It also runs [flake8](https://github.com/PyCQA/flake8) to enforce PEP8 standards.

## Running Our Tests Manually

Install the latest version of pytest and hypothesis:

```console
pip install pytest hypothesis
```

Navigate to the top-level of `hydra-zen` and run:

```console
pytest tests/
```

## Running Tests Using `tox`

`tox` is a tool that will create and manage a new Python environment where it can then run hydra-zen's
automated tests against various Python versions and dependencies.

Install `tox`:

```console
pip install tox
```

(if you like to use `conda` environments, you might also install `tox-conda`).

List the various tox-jobs that are defined for hydra-zen:

```console
tox -a
```

Then, run the job of choice using:

```console
tox -e [job-name]
```


### Measuring Code Coverage

Our CI requires that our tests achieve 100% code coverage. The easiest way to measure
code-coverage is by using `tox`:

```console
tox -e coverage
```

This will produce a coverage report that indicates any lines of code that were note covered by tests.

## Validating Type Correctness

Our CI runs the `pyright` type-checker in basic mode against hydra-zen's entire code base and against specific test files; this ensures that our type-annotations are complete and accurate.

If you use VSCode with Pylance, then make sure that `Type Checking Mode` is set to `basic` for your hydra-zen workspace. Your IDE will then mark any problematic code.Other IDEs can leverage the pyright language server to a similar effect. 

While this is helpful for getting immediate feedback about your code, it is no substitute for running `pyright` from the commandline. To do so, [install pyright](https://github.com/microsoft/pyright#command-line) and, from the top-level hydra-zen directory, run:

```console
pyright --lib tests/annotations/ src/
```
