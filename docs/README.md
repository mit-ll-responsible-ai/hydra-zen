# Building the Docs

Install `hydra-zen` and install the docs-requirements:

```shell
/hydra_zen/docs> pip install -r requirements.txt
```

Then build the docs:
 
```shell script
/hydra_zen/docs> python -m sphinx source build
```

The resulting HTML files will be in `hydra-zen/docs/build`.