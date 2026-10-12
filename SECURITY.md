# Security Policy

## Reporting a vulnerability

Please report security vulnerabilities privately, through GitHub's
[private vulnerability reporting](https://github.com/mit-ll-responsible-ai/hydra-zen/security/advisories/new),
rather than in a public issue. We will acknowledge your report and keep you
updated as we work on a fix.

Fixes are released in the latest version of hydra-zen.

## Threat model

hydra-zen treats configs as trusted code: instantiating a config imports and
calls whatever its targets name, much like loading a pickle file. Instantiating
a config from an untrusted source is therefore not a vulnerability in
hydra-zen. See [Configs Are Code: Threat Model](https://mit-ll-responsible-ai.github.io/hydra-zen/explanation/security.html)
for details, including hydra-zen's support for Hydra's execution whitelist.

Vulnerabilities in Hydra or OmegaConf themselves should be reported to those
projects.
