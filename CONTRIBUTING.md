Thanks for your interest in contributing to hydra-zen! Please read 
through the following resources before you begin working on any contributions to this 
code base.

- [Installing hydra-zen for development](#installing-hydra-zen-for-development)
- [Configuring Your IDE](#configuring-your-ide)
- [Adding New Features to the Public API](#adding-new-features-to-the-public-api)
- [Running Our Tests Manually](#running-our-tests-manually)
- [Running Tests Using `tox`](#running-tests-using-tox)
  - [Measuring Code Coverage](#measuring-code-coverage)
  - [Running Static Type Checking Tests](#running-static-type-checking-tests)
- [Formatting](#formatting)
  - [Pre-Commit Hooks](#pre-commit-hooks)
    - [What does this do?](#what-does-this-do)
- [Documentation](#documentation)
  - [Building Our Documentation Locally](#building-our-documentation-locally)
  - [Publishing Documentation](#publishing-documentation)
- [Releasing a New Version of hydra-zen](#releasing-a-new-version-of-hydra-zen)
  

## Installing hydra-zen for development

Install the toolkit along with its test dependencies; checkout the repo, navigate to its top level and run

```shell
pip install -e .
```

the `-e` option ensures that any changes that you make to the project's source code will be reflected in your local install – you need not reinstall the package in order for your modifications to take effect.

If your contributions involve changes to our support for NumPy, PyTorch, PyTorch-Lightning, JAX, pydantic, or beartype then you will need to install those dependencies as well, in order for our tests-against-third-parties to run locally in your environment.

## Configuring Your IDE

hydra-zen utilizes pyright to validate its interfaces. Thus it is recommended that developers use [an IDE with pyright language server](https://github.com/microsoft/pyright#installation). 

VSCode's Pylance extension is built off of pyright, and thus is recommended. If you use VSCode with the Pylance, then make sure that `Type Checking Mode` is set to `basic` for your hydra-zen workspace. Your IDE will then mark any problematic code.


## Adding New Features to the Public API

All functions/classes that are part of the public API must have a docstring that adheres to the [numpy docstring style](https://numpydoc.readthedocs.io/en/latest/format.html), and the docstring must include and `Examples` section. The function's docstring must be scanned by pyright, by adding the function to [this test](https://github.com/mit-ll-responsible-ai/hydra-zen/blob/main/tests/test_docstrings.py).

All publicly-facing interfaces must be type-annotated and scan "clean" using the pyright type checker.

The CI for `hydra-zen` requires 100% code coverage, thus all new features will need to be tested appropriately. We use the [pytest framework](https://docs.pytest.org/en/7.2.x/) for collecting/running/reporting tests and are keen on using the [Hypothesis library](https://hypothesis.readthedocs.io/en/latest/) for writing property based tests where appropriate.

See the section on tox for details.


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

(if you like to use `conda` environments you can instead install `tox-conda`).

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


### Running Static Type Checking Tests

Our CI runs the `pyright` type-checker in basic mode against hydra-zen's entire code base and against specific test files. It also requires a [type completeness score](https://github.com/microsoft/pyright/blob/92b4028cd5fd483efcf3f1cdb8597b2d4edd8866/docs/typed-libraries.md#verifying-type-completeness) of 100%; this ensures that the type-annotations for our public API are complete and accurate. Lastly, we run some rudimentary tests to assess basic mypy compatibility. 

You can run these static type checking tests locally using tox via:

```console
tox -e typecheck
```

## Formatting

hydra-zen's CI requires that `black`, `isort`, and `flake8` can be run against `src/` and `tests/` without any diffs or errors. To run this test locally, use


```console
$ tox -e enforce-format
```

To locally format/fix code issues caught by this stage of our CI, run:


```console
$ tox -e format
```

That being said, it is recommended that you install pre-commit hooks that will apply these formatters to the diffs of each commit that you make.
This is substantially faster and more streamlined that using the tox solution.

### Pre-Commit Hooks

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

#### What does this do?

Our pre-commit hooks run the following auto-formatters on all commits:
- [black](https://black.readthedocs.io/en/stable/)
- [isort](https://pycqa.github.io/isort/)

It also runs [flake8](https://github.com/PyCQA/flake8) to enforce PEP8 standards.

## Documentation

### Building Our Documentation Locally

Running 

```console
tox -e docs
```

will build the docs as HTML locally, and store them in `hydra_zen/.tox/docs/html`. See the `docs/README.md` in this repo for details.

### Publishing Documentation

We use [GitHub Actions](https://github.com/mit-ll-responsible-ai/hydra-zen/blob/main/.github/workflows/publish_docs.yml) to handle building and publishing our docs. The job runs Sphinx and commits the resulting artifacts to the `gh-pages` branch, from which GitHub publishes the HTML pages.

The documentation is updated by each push to `main`.

## Releasing a New Version of hydra-zen

`hydra-zen` uses [semantic versioning](https://semver.org/) and the Python package extracts its version from the latest git tag (by leveraging [setuptools-scm](https://pypi.org/project/setuptools-scm/)). Suppose we want to update hydra-zen's version to `1.3.0`; this would amount to tagging a commit:

```console
$ git tag -a v1.3.0 -m "Release 1.3.0"
$ git push origin --tags
```

Nothing needs to be updated in the Python package itself.

We utilize GitHub Actions to publish to PyPI. Once the tag has been pushed to GitHub, [draft a new release on GitHub](https://github.com/mit-ll-responsible-ai/hydra-zen/releases) with the correct associated tag, and the new version will automatically be published to PyPI.

Before releasing a new version of hydra-zen, make sure to add a new section to [the changelog](https://github.com/mit-ll-responsible-ai/hydra-zen/blob/main/docs/source/changes.rst).
