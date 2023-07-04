.. meta::
   :description: The changelog for hydra-zen, including what's new.

=========
Changelog
=========

This is a record of all past hydra-zen releases and what went into them, in reverse 
chronological order. All previous releases should still be available on pip.

.. _v0.11.0:

---------------------
0.11.0rc - 2023-04-21
---------------------

.. note:: This is documentation for an unreleased version of hydra-zen. You can try out this pre-release version using `pip install --pre hydra-zen`

This release drops support for Python 3.7 (which reached its end of life), hydra-core 
1.1 and for omegaconf 2.1; this enabled the removal of a lot of complex compatibility 
logic from hydra-zen's source code, and to improve the behavior of 
:func:`~hydra_zen.zen`.


Release Highlights
------------------
hydra-zen now uses the `trusted publishers <https://blog.pypi.org/posts/2023-04-20-introducing-trusted-publishers/>`_ method for publishing its build artifacts to PyPI via a protected GitHub Actions environment. In short, this improves security for users by further reducing the surface area by which malicious 3rd parties could attempt to upload tainted versions of hydra-zen. Note that hydra-zen has always abided by the most rigorous methods for secure publishing - we adopted the Trusted Publishers method one day after it became available ❤️.

:func:`~hydra_zen.builds` now has special behavior when it is passed a dataclass type that already possesses a `_target_` field: `_target_` is treated in a transitive way. E.g. `build(builds(int))` is equivalent to `builds(int)`. This enables a powerful "builder" pattern where configs can be populated in iterative, branching ways (with full type-checking support 😎). 

.. code-block:: python
   :caption: Basic 'builder' pattern

   from hydra_zen import make_custom_builds_fn, instantiate

   fbuilds = make_custom_builds_fn(populate_full_signature=True)

   def foo(x, y, z): return x, y, z

   base_cfg = fbuilds(foo, x=0, y=0, z=0)

   c1 = fbuilds(base_cfg, x=1)
   c2 = fbuilds(c1, y=2)
   c3 = fbuilds(c2, z=3)

.. code-block:: pycon

   >>> [instantiate(c) for c in [c1, c2, c3]]
   [(1, 0, 0), (1, 2, 0), (1, 2, 3)]


Improvements
------------
- :func:`~hydra_zen.builds` now has a transitive property that enables iterative build patterns. See :pull:`455`
- :func:`~hydra_zen.zen`'s instantiation phase has been improved so that dataclass objects and stdlib containers are returned instead of omegaconf objects. See :pull:`448`. 
- :func:`~hydra_zen.zen` can now be passed `resolve_pre_call=False` to defer the resolution of interpolated fields until after `pre_call` functions are called. See :pull:`460`.
- Added support for NumPy 1.25.0

Bug Fixes
---------
- Configs produced by `~hydra_zen.just` will no longer cause a `ReadonlyConfigError` during Hydra's config-composition process. See :pull:`459`
- :class:`~hydra_zen.ZenStore` now works with :func:`~hydra_zen.hydrated_dataclass`. See :issue:`453` and :pull:`455`.

Compatibility-Breaking Changes
------------------------------
Most of these changes will not have any impact on users, based on download statistics and the particular code patterns impacted by the following changes.

- Python 3.8 is now the earliest supported version of Python supported by hydra-zen
- hydra-core 1.2.0 and omegaconf 2.2.1 are now the minimum supported versions of hydra-zen's dependencies.
- The auto-instantiation behavior of :class:`~hydra_zen.wrapper.Zen` and :func:`~hydra_zen.zen` have been updated so that nested dataclasses (nested within lists, dicts, and other dataclasses) will no longer be returned as omegaconf configs (see :pull:`448`).
- :func:`~hydra_zen.just` not longer returns a frozen dataclass (see :pull:`459`).
- Users that relied on patterns like `builds(builds(...))` will find that :pull:`455` has changed their behaviors. This new behavior can be disabled via `builds(..., zen_convert={'flat_target': False})`
- :func:`~hydra_zen.zen`'s instantiation behavior was changed by :pull:`448`. See that PR for instructions on restoring the old behavior.
- The signature-inspection logic of :func:`~hydra_zen.builds` has been modified to adopt and backport a fix made to :py:func:`inspect.signature` in Python 3.11.4. See :pull:`497`.

--------------------------
Documentation - 2023-03-11
--------------------------

The following parts of the documentation underwent significant revisions:

- `The landing page <https://github.com/mit-ll-responsible-ai/hydra-zen>`_ now has a "hydra-zen at at glance" subsection.
- The docs for `~hydra_zen.ZenStore` were revamped.


.. _v0.10.2:

-------------------
0.10.2 - 2023-07-04
-------------------

This patch circumvents an upstream bug in pyright, which was causing pyright 1.1.305+ to report the return type of :func:`hydra_zen.make_custom_builds_fn` as "Unknown".


.. _v0.10.1:

-------------------
0.10.1 - 2023-05-23
-------------------

Bans ``typing-extension v4.6.0`` which has a breaking bug in it.

.. _v0.10.0:

-------------------
0.10.0 - 2023-03-05
-------------------

Release Highlights
------------------
:func:`hydra_zen.launch` now accepts non-string values for its overrides, and it 
accepts a dictionary for improved ergonomics. Previously, users had to form Hydra 
CLI-compatible strings when calling `launch`, now overrides can be passed to the 
`launch` API as their native types. 


.. tab-set::

   .. tab-item:: old launch

      .. code-block:: python
         :caption: Manually forming CLI-compatible overrides
      
         from hydra_zen import launch, instantiate, make_config

         values_for_experiment = [random.uniform(0, 1) for i in range(10)]

         jobs = launch(
            make_config(a=None, b=None),
            instantiate,
            overrides=[
                  "a=1",
                  "b=[1,2,3]",
                  "+param=" + ",".join([str(i) for i in values_for_experiment])
            ],
            multirun=True
         )

   .. tab-item:: improved launch

      .. code-block:: python
         :caption: Specifying native Python values in launch API
      
         from hydra_zen import launch, instantiate, make_config, multirun, hydra_list

         values_for_experiment = [random.uniform(0, 1) for i in range(10)]

         jobs = launch(
            make_config(a=None, b=None),
            instantiate,
            overrides={
                  "a": 1,
                  "b": hydra_list([1, 2, 3]),
                  "+param": multirun(values_for_experiment)
            },
            multirun=True
         )


Improvements
------------
- :func:`hydra_zen.launch` now supports dictionary overrides and will automatically convert basic Python types to CLI-compatible strings. See :pull:`313`.
- :class:`hydra_zen.ZenStore` now provides specialized support for storing instances/subclasses of `HydraConf`. See :issue:`395`.
- Adds auto-config support for jax 0.4.0. See :pull:`414`.
- Improved the type annotations of :class:`~hydra_zen.wrappers.ZenStore`. See :pull:`409`.
- :func:`hydra_zen.builds` now has type-conversion support for `dataclasses.InitVar`. See :pull:`418`.


--------------------------
Documentation - 2023-01-22
--------------------------

The following How-To guides were added:

- `How to configure multiple experiments <https://mit-ll-responsible-ai.github.io/hydra-zen/how_to/configuring_experiments.html>`_
- `How to customize Hydra's configuration <https://mit-ll-responsible-ai.github.io/hydra-zen/how_to/configure_hydra.html>`_
- `How to configure and run scikit-learn's "Classier Comparison" example <https://mit-ll-responsible-ai.github.io/hydra-zen/how_to/using_scikit_learn.html>`_



.. _v0.9.1:

------------------
0.9.1 - 2023-01-13
------------------


Improvements
------------
- :func:`hydra_zen.zen` now returns pickle-compatible wrapped functions. See :pull:`384`.

Bug Fixes
---------
- :func:`hydra_zen.zen`'s `hydra_main` method now handles string `config_path` entries properly (only for Hydra 1.3.0+). Previously Hydra could not find the path to the wrapped task function. hydra-zen will warn users that a string `config_path` is not supported via :func:`hydra_zen.zen` for Hydra 1.2 and earlier. See :pull:`384`.

.. _v0.9.0:

------------------
0.9.0 - 2022-12-30
------------------

Release Highlights
------------------
This release introduces :func:`~hydra_zen.zen` and :class:`~hydra_zen.ZenStore`, which enable hydra-zen users to eliminate Hydra-specific boilerplate code from their projects and to utilize new patterns and best practices for working with config stores.

The wrapper `~hydra_zen.zen` will automatically extract, resolve, and instantiate 
fields from a config in order to call the function that it has wrapped, thus saving the 
user from writing repetitive, hydra-specific boilerplate code in their function.
Thus this wrapper enables users to replace the following Hydra-specific task function:

.. code-block:: python
   :caption: The "old school" way of designing a task function for a Hydra app

   import hydra
   from hydra.utils import instantiate
   
   @hydra.main(config_name="my_app", config_path=None, version_base="1.2")
   def trainer_task_fn(cfg):
      model = instantiate(cfg.model)
      data = instantiate(cfg.data)
      partial_optim = instantiate(cfg.partial_optim)
      trainer = instantiate(cfg.trainer)
      
      optim = partial_optim(model.parameters())
      trainer(model, optim, data).fit(cfg.num_epochs)
   
   if __name__ == "__main__":
      trainer_task_fn()      

with a Hydra-agnostic task function that has an explicit signature:

.. code-block:: python
   :caption: Using `zen` to design a Hydra-agnostic task function


   # note: no Hydra or hydra-zen specific logic here
   def trainer_task_fn(model, data, partial_optim, trainer, num_epochs):
      optim = partial_optim(model.parameters())
      trainer(model, optim, data).fit(num_epochs)
   
   if __name__ == "__main__":
       from hydra_zen import zen
       
       # All config-field extraction & instantiation is automated/mediated by zen.
       # I.e. `zen` will extract & instantiate model, data, etc. from the input
       # config and pass it to `trainer_task_fn`
       zen(trainer_task_fn).hydra_main(config_name="my_app", config_path=None)


There are plenty more bells and whistles to :func:`~hydra_zen.zen`, refer to :pull:`310` and its reference documentation for more details.

:class:`~hydra_zen.ZenStore` is an abstraction over Hydra's config store.
It enables users to maintain multiple, isolated store instances before populating 
Hydra's global config store. It also protects users from accidentally overwriting  
entries in Hydra's global store. :class:`~hydra_zen.ZenStore` possesses auto-config 
capabilities: it will automatically apply :func:`~hyda_zen.builds` and 
:func:`~hyda_zen.just` in intuitive ways on inputs to generate the stored configs.

.. code-block:: python
   :caption: Using `hydra_zen.store` auto-generate and store configs

   from hydra_zen import ZenStore
   from torch.optim import Adam, AdamW, RMSprop

   torch_store = ZenStore("torch_store")

   # Specify defaults for storing entries (group=optim)
   # and for generating configs (_partial_=True and lr=1e-3)
   optim_store = torch_store(group="optim", zen_partial=True, lr=0.001)

   # Automatically applies `builds(<obj>, zen_partial=True, lr=0.001)` 
   # to create and then store configs under the "optim" group
   optim_store(Adam, name="adam", amsgrad=True)
   optim_store(AdamW, name="adamw", betas=(0.1, 0.999))
   optim_store(RMSprop, name="rmsprop")

   torch_store.add_to_hydra_store()  # populate Hydra's global store

The store can also be populated using a decorator pattern [1]_, e.g.

.. code-block:: python
   :caption: Using `hydra_zen.store` as a decorator to auto-configure and store objects.

   from dataclasses import dataclass
   from hydra_zen import store

   profile_store = store(group="profile")

   # Adds two store entries under the "profile" group of the store
   # with configured defaults for `has_root`
   @profile_store(name="admin", has_root=True)
   @profile_store(name="basic", has_root=False)
   @dataclass
   class Profile:
       username: str
       schema: str
       has_root: bool

   
   db_store = store(group="database")

   # calls `builds(profile_database, [...])` under the hood and
   # adds the config to the store under the "profile" group
   @db_store(name="database")
   @db_store(name="test_database", size=1)
   def profile_database(size):
       ...


New Features
------------
- hydra-zen now supports Python 3.11
- Adds the :func:`~hydra_zen.zen` decorator (see :pull:`310`)
- Adds the :class:`~hydra_zen.wrapper.Zen` decorator-class (see :pull:`310`)
- Adds the :class:`~hydra_zen.ZenStore` class (see :pull:`331`)
- Adds `hyda_zen.store`, which is a pre-initialized instance of :class:`~hydra_zen.ZenStore` (see :pull:`331`)
- The option `hydra_convert='object'` is now supported by all of hydra-zen's config-creation functions. So that an instantiated structured config can be converted to an instance of its backing dataclass. This feature was added by `Hydra 1.3.0 <https://github.com/facebookresearch/hydra/issues/1719>`_.
- Adds auto-config support for `torch.optim.optimizer.required` so that the common pattern `builds(<torch_optimizer_type>, populate_full_signature=True, zen_partial=True)` works and exposes `lr` as a required configurable parameter. Thanks to @addisonklinke for requesting this in :issue:`257`.
- :ref:`builds([...], zen_wrapper=...) <zen-wrapper>` can now accept a partial'd function as a wrapper.

Improvements
------------
- Updated the :ref:`Tutorials <tutorials>` and :ref:`How-To Guides <hydra-zen-how-to>` to reflect idiomatic usage of :class:`~hydra_zen.ZenStore` and :func:`~hydra_zen.zen`.
- :func:`~hydra_zen.hydrated_dataclass` will now produce a pickle-compatible dataclass type. See :pull:`338`.
- hydra-zen's :ref:`auto-config support <additional-types>` has been enhanced so that it produces pickle-compatible configs. This excludes auto-config support for :py:func:`functools.partial` and :py:func:`dataclasses.dataclass`. See :pull:`360`.
- All options available to :func:`dataclasses.dataclass` are now exposed by :func:`~hydra_zen.hydrated_dataclass`, :func:`~hydra_zen.builds`, :func:`~hydra_zen.make_custom_builds_fn`, :func:`~hydra_zen.make_config`, and :func:`~hydra_zen.just` via the :class:`hydra_zen.typing.DataclassOptions` API. See :pull:`360`.
- All documentation code blocks are scanned by pyright as part of our CI process. Several errors in the documentation were fixed. See :pull:`343` and :pull:`344`.
- Updated the `hydra_zen.typing.Partial` protocol to match against the output of `functools.partial` more reliably in the eyes of pyright (:pull:`354`).

Bug Fixes
---------
- :pull:`355` fixes an issue where the parameterized generic `hydra_zen.typing.Partial[<...>]` would return `None` for Python versions 3.9+. This prevented this annotation from being used by runtime type checkers.

Deprecations
------------
- Specifying `frozen=True` via :func:`~hydra_zen.builds` or :func:`~hydra_zen.make_config` is deprecated in favor of `zen_dataclass={'frozen': True}`. See :pull:`360`.
- Specifying `config_name=<str>` via :func:`~hydra_zen.builds` or :func:`~hydra_zen.make_config` is deprecated in favor of `zen_dataclass={'cls_name': True}`. See :pull:`360`.

Compatibility-Breaking Changes
------------------------------
- Calling :func:`~hydra_zen.just` on a class-object or function will now return a frozen instance of a statically-defined dataclass. Previously it returned a dynamically-defined dataclass type. This change was made to improve pickle-compatibility and hashability of configs that are automatically generated by hydra-zen. This is unlikely to cause any issues for users.
- Previously, any class decorated by :func:`~hydra_zen.hydrated_dataclass` would have a `__module__` attribute set to `typing`. Now the class's `__module__` will reflect the module where its static definition resides. This enables pickle-compatibility  (:pull:`338`). This is unlikely to cause any issues for users.

Mutable Default Values for Dataclasses
======================================
Beginning in Python 3.11 :func:`dataclasses.dataclass` `checks for mutable default values <https://docs.python.org/3/library/dataclasses.html#mutable-default-values>`_ by assessing if an object possesses a `__hash__` attribute. Previously it only considered `set`, `dict`, and `list` types to be mutable. Accordingly, dataclass instances are now considered to be mutable unless they are frozen or if `unsafe_hash=True` was specified.

.. code-block:: python
   :caption: Demonstrating change in mutability rules for dataclasses starting in Python 3.11

   from dataclasses import dataclass, field
   
   @dataclass
   class A:
      ...
   
   @dataclass
   class NoLongerValid:
      number: int = 1
      nested: A = A()  # will raise at runtime due to mutable default

   @dataclass
   class IsOK:
      number: int = 1
      nested: A = field(default_factory=lambda: A())

A ramification of the use of a default-factory in this example is that the field `nested` can only be accessed from an *instance* of ``IsOK``, whereas non-factory defaults can be accessed from the dataclass type itself.

.. code-block:: pycon
   :caption: Default factories require access from dataclass instances; they cannot be accessed from the dataclass type.

   >>> hasattr(IsOK, "number")
   True
   >>> hasattr(IsOK, "nested")
   False
   >>> hasattr(IsOK(), "nested")
   True

Because hydra-zen users frequently nest dataclasses, hydra-zen's dataclass-creation functions (`builds` et al.) now specify `unsafe_hash=True` by default. Thus the following pattern is still valid:

.. code-block:: python
   :caption: The dataclasses produced by hydra-zen 0.9.0 are hashable by default so that existing patterns do not break in Python 3.11.

   from dataclasses import dataclass, field
   from typing import Any
   
   from hydra_zen import builds
   from hydra_zen.typing import Builds

   @dataclass
   class Config:
       # This is still OK
       builds_dict: Builds[type[dict[Any, Any]]] = builds(dict)()

That being said, hydra-zen will now treat dataclass instances whose `__hash__` attribute is `None` as mutable – *regardless of the Python version* – in order to ensure consistent behaviors across all supported Python versions. Thus the following pattern will now break

.. code-block:: python
   
   @dataclass
   class A:
       ...

   Conf = builds(dict, y=A(), zen_convert={'dataclass': False})
   
   Conf.y  # this will raise in hydra_zen 0.9.0+
   Conf().y  # this is OK

In general it is recommended that config fields be accessed from dataclass instances, not types. This will avoid all such default value/factory issues.


.. _v0.8.0:

------------------
0.8.0 - 2022-09-13
------------------


Release Highlights
------------------
This release adds auto-config support for dataclass types and instances, **including pydantic datclasses**. Thus one can now include in 
a structured config type-annotations and default values that *are not natively 
supported by Hydra*, and then use :func:`~hydra_zen.builds` and/or 
:func:`~hydra_zen.just` to create a Hydra-compatible intermediate .

Consider the following dataclass; neither the type-annotation for ``reduction_fn`` nor its default values are supported by Hydra/omegaconf, and thus it cannot be serialized to a yaml file nor used in a Hydra config.

.. code-block:: python
   :caption: A dataclass that cannot be used natively within a Hydra app as a structured config.

   from typing import Callable, Sequence
   from dataclasses import dataclass
   
   @dataclass
   class Bar:
      reduce_fn: Callable[[Sequence[float]], float] = sum  # <- not compat w/ Hydra


With the release of hydra-zen 0.8.0, we can now use :func:`~hydra_zen.just` to 
automatically create a Hydra-compatible config that, when instantiated, returns ``Bar()``:

.. code-block:: pycon
   :caption: Using :func:`~hydra_zen.just` to create a Hydra-compatible structured config

   >>> from hydra_zen import builds, just, instantiate, to_yaml
   >>> just_bar = just(Bar())
   
   >>> print(to_yaml(just_bar))
   _target_: __main__.Bar
   reduce_fn:
     _target_: hydra_zen.funcs.get_obj
     path: builtins.sum
   
   >>> instantiate(just_bar)  # returns Bar()
   Bar(reduce_fn=<built-in function sum>)

This auto-conversion process works recursively as well

.. code-block:: pycon
   :caption: Demonstrating recursive auto-conversion of dataclasses.

   >>> from statistics import mean
   >>> @dataclass
   ... class Foo:
   ...     bar: Bar

   >>> foobar = Foo(Bar(reduce_fn=mean))
   >>> instantiate(just(foobar))
   Foo(bar=Bar(reduce_fn=<function mean at 0x000001F224640310>))
   >>> instantiate(builds(Foo, bar=Bar(sum)))
   Foo(bar=Bar(reduce_fn=<built-in function sum>))

Thus we can include these Hydra-compatible intermediates in our Hydra config or config store, and then use :func:`~hydra_zen.instantiate` to create the desired dataclass instances of ``Bar()`` and ``Foo(Bar(mean))`` within our app's task function.
Note that this functionality works with `pydantic dataclasses <https://pydantic-docs.helpmanual.io/usage/dataclasses/>`_ as well, which enables us to leverage enhanced runtime value and type-checking.

Big thanks to `Jasha10 <https://github.com/Jasha10>`_ for proposing and prototyping the crux of this new capability.

Compatibility-Breaking Changes
------------------------------
This release drops support for Python 3.6. If you require Python 3.6, please restrict your hydra-zen installation dependency as `hydra-zen<0.8.0`.

Specifying `make_custom_builds_fn([...], builds_bases=<...>)` was deprecated in 
hydra-zen 0.7.0 (:pull:`263`). Accordingly, this option has now been removed from
:func:`hydra_zen.make_custom_builds_fn`.

The addition of auto-config support for dataclasses (:pull:`301`) changes the default 
behaviors of :func:`~hydra_zen.just` and :func:`~hydra_zen.builds`. Previously, all 
dataclass types and instances lacking a `_target_` field would be left unprocessed by 
these functions, and omegaconf would convert dataclass types and instances alike to 
DictConfigs

.. code-block:: python
   :caption: hydra-zen < 0.8.0

   from hydra_zen import just, builds, to_yaml
   from dataclasses import dataclass
   from omegaconf import DictConfig
   
   @dataclass
   class A:
       x: int = 1
   
   assert to_yaml(just(A)) == "x: 1\n"
   assert to_yaml(just(A())) == "x: 1\n"
   assert to_yaml(builds(dict, x=A)().x) == "x: 1\n"
   assert to_yaml(builds(dict, x=A())().x) == "x: 1\n"

Now these objects will automatically be converted to corresponding targeted configs 
with the desired behavior under Hydra-instantiation:

.. code-block:: python
   :caption: hydra-zen >= 0.8.0

   from hydra_zen import just, builds, instantiate
   from dataclasses import dataclass

   @dataclass
   class A:
       x: int = 1

   assert instantiate(just(A)) is A
   assert instantiate(builds(dict, x=A)().x) is A
   
   assert str(just(A())()) == "Builds_A(_target_='__main__.A', x=1)"
   assert str(builds(dict, x=A(), hydra_convert="all")()) == "Builds_dict(_target_='builtins.dict', _convert_='all', x=<class 'types.Builds_A'>)"

If you depended on the previous default behavior, you can recreate it by using the new 
:ref:`zen-convert settings <zen-convert>` as so:

.. code-block:: python
   :caption: Restoring old default behavior
   
   from hydra_zen import just, make_custom_builds_fn
   from functools import partial
   
   just = partial(just, zen_convert={"dataclass": False})
   builds = make_custom_builds_fn(zen_convert={"dataclass": False})

Improvements
------------
- Adds auto-config support for `dataclasses.dataclass` (as highlighted above). (See :pull:`301`)
- :func:`~hydra_zen.builds` no longer has restrictions on inheritance patterns involving `PartialBuilds`-type configs. (See :pull:`290`)
- We now verify that basic use cases of our config-creation and instantiation functions type-check correctly via mypy. Previously, we had only assured type-checking behavior via pyright
- Added :class:`~hydra_zen.typing.ZenConvert` typed dictionary to document new zen-convert options for :func:`~hydra_zen.builds`, :func:`~hydra_zen.just`, and :func:`~hydra_zen.make_config`. (See :pull:`301`)
- Adds support for using `builds(<target>, populate_full_signature=True)` where `<target>` is a dataclass type that has a field with a default factory. (See :pull:`299`)
- Adds auto-config support for `pydantic.Field`, improving hydra-zen's ability to automatically construct configs that describe pydantic models and dataclasses. (See :pull:`303`) 
- Two new utility functions were added to the public API: :func:`~hydra_zen.is_partial_builds` and :func:`~hydra_zen.uses_zen_processing`
- The :ref:`automatic type refinement <type-support>` performed by :func:`~hydra_zen.builds` now has enhanced support for ``typing.Annotated``, ``typing.NewType``, and ``typing.TypeVarTuple``. (See :pull:`283`)
- Docs: Upgraded sphinx theme: dark mode is now available!
- Docs: Re-enabled sphinx code auto-link

**Support for New Hydra/OmegaConf Features**

- OmegaConf ``v2.2.1`` added native support for :py:class:`pathlib.Path`. hydra-zen :ref:`already provides support for these <additional-types>`, but will now defer to OmegaConf's native support when possible. (See :pull:`276`)
- Improved :ref:`automatic type refinement <type-support>` for bare sequence types, and adds conditional support for `dict`, `list`, and `tuple` as type annotations when omegaconf 2.2.3+ is installed. (See :pull:`297`)


Bug Fixes
---------
- :func:`~hydra_zen.builds` would raise a ``TypeError`` if it encountered a target whose signature contained the annotations ``ParamSpecArgs`` or  ``ParamSpecKwargs``. It can now sanitize these annotations properly. (See :pull:`283`)


.. _v0.7.1:

------------------
0.7.1 - 2022-06-22
------------------

Bug Fixes
---------

The validation that hydra-zen performs on ``hydra_defaults`` was overly restrictive. E.g. it would flag ``[{"some_group": None}]`` as invalid, even though null is permitted in `Hydra's default list syntax <https://hydra.cc/docs/advanced/defaults_list/>`_.
This patch fixes this validation and updates the docs & annotations for ``hydra_defaults`` in :func:`~hydra_zen.builds` and :func:`~hydra_zen.make_config`.
See :pull:`287` for more details. Thanks to ``@mgrinshpon-doxel`` for the bug report.


.. _v0.7.0:

------------------
0.7.0 - 2022-05-10
------------------

New Features
------------

**Support for defaults lists**

Hydra's `defaults list <https://hydra.cc/docs/advanced/defaults_list/>`_ field can be passed to :func:`~hydra_zen.builds` and :func:`~hydra_zen.make_config` via the new ``hydra_defaults`` argument. Basic runtime and static type-checking are performed on this field. See :pull:`264` for more details and examples.


**Improved functionality for types with Specialized hydra-zen support**

:func:`~hydra_zen.just`, :func:`~hydra_zen.to_yaml`, and :func:`~hydra_zen.save_as_yaml` can directly 
operate on values of :ref:`types with specialized support from hydra-zen <additional-types>`; these 
values will automatically be converted to structured configs. 

.. code-block:: pycon

   >>> from functools import partial
   >>> from hydra_zen import to_yaml, just

   >>> def f(x): return x**2
   >>> partiald_f = partial(f, x=2)

   >>> just(partiald_f)  # convert to structured config
   PartialBuilds_f(_target_='__main__.f', _partial_=True, x=2)

   >>> print(to_yaml(partiald_f))  # convert to yaml
   _target_: __main__.f
   _partial_: true
   x: 2

See :pull:`250` and :pull:`259` for more details and examples.

Support for Upcoming Hydra/OmegaConf Features
---------------------------------------------
OmegaConf ``v2.2.0`` is adding native support for the following types:

- :py:class:`bytes`

hydra-zen :ref:`already provides support for these <additional-types>`, but this version will defer to OmegaConf's native support when possible. (See :pull:`262`)

OmegaConf ``v2.2.0`` improves its type-checking, with added support for nested 
containers. Accordingly, hydra-zen's :ref:`automatic type refinement <type-support>` 
will no longer auto-broaden nested container types when ``OmegaConf v2.2.0+`` is 
installed. (See :pull:`261`)


Hydra ``v1.2.0`` is introducing a ``version_base`` parameter that can control default behaviors in ``hydra.run`` and ``hydra.initialize``.
Correspondingly, ``version_base`` is now exposed via `~hydra_zen.launch`. See :pull:`273` for more details.


.. _0p7p0-deprecations:

Deprecations
------------
:pull:`263` deprecates the ``builds_bases`` argument in :func:`~hydra_zen.make_custom_builds`. It will 
be removed in hydra-zen v0.8.0. Users will need to specify ``builds_bases`` on a 
per-config basis via ``builds``.


Bug Fixes
---------
- ``hydra_zen.builds(<Child.class-method>)`` would create a config with the wrong target if ``<class-method>`` was defined on a parent of ``Child``. See :issue:`265`.

Improvements
------------
- Fixed internal protocol of ``partial`` to be compatible with latest type-shed annotations.
- Add missing annotation overloads for :func:`~hydra_zen.builds` and :func:`~hydra_zen.make_custom_builds`
- Substantial source code reorganization
- Improved pyright tests

.. _v0.6.0:

------------------
0.6.0 - 2022-03-09
------------------

This release focuses on improving hydra-zen's type-annotations; it increases the 
degree to which IDEs and static-analysis tools can infer information about common
hydra-zen code patterns.

It should be noted that hydra-zen leverages advanced typing features (e.g. recursive 
types) and that some type-checkers do not support these features yet. hydra-zen's type 
annotations are validated by `pyright <https://github.com/microsoft/pyright>`_. Thus we recommend that users leverage pyright and pyright-based language servers in their 
IDEs (e.g. using Pylance in VSCode) for the best experience.

(A note to VSCode users: make sure to set `Type Checking Mode` to `basic` in your IDE -- it is disabled by default!)

Bug Fixes
---------

``builds(<target>, builds_bases=(...))`` now properly supports the case where a parent config introduces zen-processing features via inheritance. See :pull:`236` for more details.


Improvements
------------
- ``builds(<target>, populate_full_signature=True)`` now carries accurate type information about the target's signature. Thus IDEs can now auto-complete the signature of the resulting structured config. See :pull:`224` for examples and details.
- Type-information is now dispatched by :func:`~hydra_zen.make_custom_builds_fn` for the common use-cases of ``populate_full_signature=True`` and ``zen_partial=True``, respectively. See :pull:`224` for examples and details.
- ``hydra_zen.typing.ZenWrappers`` is now a publicly-available annotation. It reflects valid types for ``builds(..., zen_wrappers=<...>)``.
- hydra-zen now has a pyright-verified `type completeness score <https://github.com/microsoft/pyright/blob/92b4028cd5fd483efcf3f1cdb8597b2d4edd8866/docs/typed-libraries.md#verifying-type-completeness>`_ of 100%. Our CI now requires that this score does not drop below 100%. See :pull:`226` for more details.
- Improved compatibility with mypy (:pull:`243`)
 

Support for Upcoming Hydra Features
-----------------------------------

Hydra 1.1.2 will introduce `support for partial instantiation of targeted configs <https://hydra.cc/docs/next/advanced/instantiate_objects/overview/#partial-instantiation>`_ via the ``_partial_`` field. ``builds(<target>, zen_partial=True)`` will now set the ``_partial_`` field on the structured config
rather than using ``hydra_zen.funcs.zen_processing`` to facilitate partial instantiation.


+---------------------------------------------------+---------------------------------------------------+
| .. code-block:: pycon                             | .. code-block:: pycon                             |
|    :caption: Hydra < 1.1.2                        |    :caption: 1.1.2 <= Hydra                       |
|                                                   |                                                   |
|    >>> Conf = builds(dict, a=1, zen_partial=True) |    >>> Conf = builds(dict, a=1, zen_partial=True) |
|                                                   |                                                   |
|    >>> print(to_yaml(Conf))                       |    >>> print(to_yaml(Conf))                       |
|    _target_: hydra_zen.funcs.zen_processing       |    _target_: builtins.dict                        |
|    _zen_target: builtins.dict                     |    _partial_: true                                |
|    _zen_partial: true                             |    a: 1                                           |
|    a: 1                                           |                                                   |
|                                                   |    >>> instantiate(Conf)                          |
|    >>> instantiate(Conf)                          |    functools.partial(<class 'dict'>, a=1)         |
|    functools.partial(<class 'dict'>, a=1)         |                                                   |
+---------------------------------------------------+---------------------------------------------------+


This change will only occur when one's locally-installed version of ``hydra-core`` is 1.1.2 or higher. Structured configs and yamls that configure partial'd objects via ``hydra_zen.funcs.zen_processing`` are still valid and will instantiate in the same way as before. I.e. this is only a compatibility-breaking change for code that relied on the specific implementation details of the structured config produced by ``builds(<target>, zen_partial=True)``.

In accordance with this change, the definition of ``hydra_zen.typing.PartialBuilds`` has been changed; it now reflects a union of protocols: ``ZenPartialBuilds[T] | HydraPartialBuilds[T]``, both are which are now part of the public API of ``hydra_zen.typing``.

(See :pull:`186` and :pull:`230` for additional details)

Compatibility-Breaking Changes
------------------------------

``hydra_zen.typing.PartialBuilds`` is no longer a runtime-checkable protocol.
Code that used ``PartialBuilds`` in this way can be updated as follows:


+---------------------------------------------------+--------------------------------------------------------------------------+
|                                                   |                                                                          |
| .. code-block:: pycon                             | .. code-block:: pycon                                                    |
|    :caption: hydra-zen < 0.6.0                    |    :caption: 0.6.0 <= hydra-zen                                          |
|                                                   |                                                                          |
|    >>> from hydra_zen.typing import PartialBuilds |    >>> from hydra_zen.typing import HydraPartialBuilds, ZenPartialBuilds |
|                                                   |                                                                          |
|    >>> Conf = builds(int, zen_partial=True)       |    >>> Conf = builds(int, zen_partial=True)                              |
|    >>> isinstance(Conf, PartialBuilds)            |    >>> isinstance(Conf, (HydraPartialBuilds, ZenPartialBuilds))          |
|    True                                           |    True                                                                  |
+---------------------------------------------------+--------------------------------------------------------------------------+

.. _v0.5.0:

------------------
0.5.0 - 2022-01-27
------------------

This release primarily improves the ability of :func:`~hydra_zen.builds` to inspect and
the signatures of its targets; thus its ability to both auto-generate and validate 
configs is improved. This includes automatic support for specifying "partial'd" objects 
-- objects produced by :py:func:`functools.partial` -- as configured values, and even as
the target of :func:`~hydra_zen.builds`.

New Features
------------
- Objects produced by :py:func:`functools.partial` can now be specified directly as configured values in :func:`~hydra_zen.make_config` and :func:`~hydra_zen.builds`. See :pull:`198` for examples.
- An object produced by :py:func:`functools.partial` can now be specified as the target of :func:`~hydra_zen.builds`; ``builds`` will automatically "unpack" this partial'd object and incorporate its arguments into the config. See :pull:`199` for examples.

Improvements
------------
- Fixed an edge case `caused by an upstream bug in inspect.signature <https://bugs.python.org/issue40897>`_, which prevented :func:`~hydra_zen.builds` from accessing the appropriate signature for some target classes. This affected a couple of popular PyTorch classes, such as ``torch.utils.data.DataLoader`` and ``torch.utils.data.Dataset``. See :pull:`189` for examples. 
- When appropriate, ``builds(<target>, ...)`` will now consult ``<target>.__new__`` to acquire the type-hints of the target's signature. See :pull:`189` for examples. 
- Fixed an edge case in the :ref:`type-widening behavior <type-support>` in both :func:`~hydra_zen.builds` and :func:`~hydra_zen.make_config` where a ``Builds``-like annotation would be widened to ``Any``; this widening was too aggressive. See :pull:`185` for examples.
- :ref:`Type widening <type-support>` will now be applied to configured fields where an interpolated variable -- a string of form ``"${<var-name>}"`` -- is specified. See :issue:`206` for rationale and examples.
- Fixed incomplete annotations for ``builds(..., zen_wrappers=<..>)``. See :pull:`180`

Compatibility-Breaking Changes
------------------------------

The deprecations :ref:`introduced in v0.3.0 <0p3p0-deprecations>` are now errors. Refer to those notes for details and for solutions for fixing stale code.


Notes
-----
It should be noted that the aforementioned improvements to :func:`~hydra_zen.builds` 
can change the interface to your app.

For instance, if you were configuring ``torch.utils.data.DataLoader``, note the 
following difference in behavior:

.. code-block:: python

   import torch as tr
   from hydra_zen import builds, to_yaml

   # DataLoader was affected by a bug in `inspect.signature`
   ConfLoader = builds(tr.utils.data.DataLoader, populate_full_signature=True)

Before 0.5.0:

.. code-block:: pycon

   >>> print(to_yaml(ConfLoader))  # builds could not access signature
   _target_: torch.utils.data.dataloader.DataLoader

After:

.. code-block:: pycon

   >>> print(to_yaml(ConfLoader))
   _target_: torch.utils.data.dataloader.DataLoader
   dataset: ???
   batch_size: 1
   shuffle: false
   sampler: null
   batch_sampler: null
   num_workers: 0
   collate_fn: null
   pin_memory: false
   drop_last: false
   timeout: 0.0
   worker_init_fn: null
   multiprocessing_context: null
   generator: null
   prefetch_factor: 2
   persistent_workers: false


.. _v0.4.1:

------------------
0.4.1 - 2021-12-06
------------------

:ref:`v0.4.0` introduced an undocumented, compatibility-breaking change to how hydra-zen treats :py:class:`enum.Enum` values. This patch reverts that change.

.. _v0.4.0:

------------------
0.4.0 - 2021-12-05
------------------

This release makes improvements to the validation performed by hydra-zen's 
:ref:`config-creation functions <create-config>`. It also adds specialized support for 
types that are not natively supported by Hydra.

Also included is an important compatibility-breaking change and a downstream 
fix for an upstream bug in 
`omegaconf <https://omegaconf.readthedocs.io/en/2.1_branch/>`_ (a library on which 
Hydra intimately depends). Thus it is highly recommended that users prioritize 
upgrading to hydra-zen v0.4.0.

New Features
------------

- Strict runtime *and* static validation of configuration types. See :pull:`163` for detailed descriptions and examples.
  
    hydra-zen's :ref:`config-creation functions <create-config>` now provide both strict runtime and static validation of the configured values that they are fed. Thus users will have a much easier time identifying and diagnosing bad configs, before launching a Hydra job.
- Specialized support for additional configuration-value types. See :pull:`163` for detailed descriptions and examples.

   Now values of types like :py:class:`complex` and :py:class:`pathlib.Path` can be specified directly in hydra-zen's configuration functions, and hydra-zen will automatically construct nested configs for those values. Consult :ref:`valid-types` for a complete list of the additional types that are supported.

Compatibility-Breaking Changes
------------------------------
We changed the behavior of :func:`~hydra_zen.builds` when 
`populate_full_signature=True` and one or more base-classes are specified for 
inheritance. 

Previously, fields specified by the parent class would take priority over those that 
would be auto-populated. However, this behavior is unintuitive as 
`populate_full_signature=True` should behave identically as the case where one 
manually-specifies the arguments from a target's signature. Thus we have changed the 
behavior accordingly. Please read more about it in :pull:`174`.

Bug Fixes
---------
The following bug was discovered in ``omegaconf <= 2.1.1``: a config that specifies a 
mutable default value for a field, but inherits from a parent that provides a 
non-mutable value for that field, will instantiate with the parent's field. Please read more about this issue, and our downstream fix for it, at :pull:`172`. 

It is recommended that users upgrade to the latest version of omegaconf once it is 
released, which will likely include a proper upstream fix of the bug.

Other improvements
------------------
hydra-zen will never be the first to import third-party libraries for which it provides 
specialized support (e.g., NumPy).

.. _v0.3.1:

------------------
0.3.1 - 2021-11-13
------------------

This release fixes a bug that was reported in :issue:`161`. Prior to this patch,
there was a bug in :func:`~hydra_zen.builds` where specifying ``populate_full_sig=True``
for a target that did not have ``**kwargs`` caused all user-specified zen-meta fields
to be excluded from the resulting config.

.. _v0.3.0:

------------------
0.3.0 - 2021-10-27
------------------

This release adds many new features to hydra-zen, and is a big step towards ``v1.0.0``. It also introduces some significant API changes, meaning that there are notable deprecations of expressions that were valid in ``v0.2.0``.

.. note::

   📚 We have completely rewritten our docs! The docs now follow the `Diátaxis Framework for technical documentation authoring <https://diataxis.fr/>`_.

.. admonition:: Join the Discussion 💬

   The hydra-zen project `now has a discussion board <https://github.com/mit-ll-responsible-ai/hydra-zen/discussions>`_. Stop by and say "hi"! 


New Features
------------
- The introduction of ``builds(..., zen_wrappers=<>)``. 
  
    This is an extremely powerful feature that enables one to modify the instantiation of a builds-config, by including wrappers in a target's configuration. `Read more about it here <https://github.com/mit-ll-responsible-ai/hydra-zen/pull/122>`_.
- Rich support for runtime type-checking of configurations. 

   Piggybacking off of the introduction of the ``zen_wrappers`` feature, **hydra-zen now offers support for customized runtime type-checking**. Presently, either of two type-checking libraries can be used: pydantic and beartype.

   - `Read about hydra-zen compatibility with pydantic <https://github.com/mit-ll-responsible-ai/hydra-zen/pull/126>`_
   - `Read about hydra-zen compatibility with beartype <https://github.com/mit-ll-responsible-ai/hydra-zen/pull/128>`_
   
  The type-checking capabilities offered by :func:`~hydra_zen.third_party.pydantic.validates_with_pydantic` and :func:`~hydra_zen.third_party.beartype.validates_with_beartype`, respectively, are both far more robust than those `offered by Hydra <https://hydra.cc/docs/tutorials/structured_config/intro/#structured-configs-supports>`_.
- A new, simplified method for creating a structured config, via :func:`~hydra_zen.make_config`.
  
   This serves as a much more succinct way to create a dataclass, where specifying type-annotations is optional. Additionally, provided type-annotations and default values are automatically adapted to be made compatible with Hydra. `Read more here <https://github.com/mit-ll-responsible-ai/hydra-zen/pull/130>`_.
- :func:`~hydra_zen.make_custom_builds_fn`, which enables us to produce new "copies" of the :func:`~hydra_zen.builds` function, but with customized default-values.
- :func:`~hydra_zen.get_target`, which is used to retrieve target-objects from structured configs. See :pull:`94`
- ``builds(..., zen_meta=<dict>)`` users to attach "meta" fields to a targeted config, which will *not* be used by instantiate when building the target. 

   A meta-field can be referenced via relative interpolation; this
   interpolation will be valid no matter where the configuration is
   utilized. See :pull:`112`.

.. _0p3p0-deprecations:

Deprecations
------------
- The use of both ``hydra_zen.experimental.hydra_run`` and ``hydra_zen.experimental.hydra_multirun`` are deprecated in favor of the the function :func:`~hydra_zen.launch`.
- Creating partial configurations with ``builds(..., hydra_partial=True)`` is now deprecated in favor of ``builds(..., zen_partial=True)``.
- The first argument of :func:`~hydra_zen.builds` is now a positional-only argument. Code that specifies ``builds(target=<target>, ...)`` will now raise a deprecation warning; use ``builds(<target>, ...)`` instead. Previously, it was impossible to specify ``target`` as a keyword argument for the object being configured; now, e.g., ``builds(dict, target=1)`` will work. (See: `#104 <https://github.com/mit-ll-responsible-ai/hydra-zen/pull/104>`_).
- All keyword arguments of the form ``zen_xx``, ``hydra_xx``, and ``_zen_xx`` are reserved by both :func:`~hydra_zen.builds` and :func:`~hydra_zen.make_config`, to ensure that future features introduced by Hydra and hydra-zen will not cause compatibility conflicts for users.


Additional Items
----------------

- Improves type-annotations on :func:`~hydra_zen.builds`. Now, e.g., ``builds("hi")`` will be marked as invalid by static checkers (the target of :func:`~hydra_zen.builds` must be callable). See :pull:`104`.
- Migrates zen-specific fields to a new naming-scheme, and zen-specific processing to a universal mechanism. See :pull:`110` for more details.
- Ensures that hydra-zen's source code is "pyright-clean", under `pyright's basic type-checking mode <https://github.com/microsoft/pyright/blob/main/docs/configuration.md#diagnostic-rule-defaults>`_. `#101 <https://github.com/mit-ll-responsible-ai/hydra-zen/pull/101>`_
- Adds to all public modules/packages an ``__all__`` field. See :pull:`99`.
- Adds PEP 561 compliance (e.g. hydra-zen is now compatible with mypy). See :pull:`97`.
- Refactors hydra-zen's internals using `shed <https://pypi.org/project/shed/>`_. See :pull:`95`.
- Makes improvements to hydra-zen's test suite. See :pull:`90` and :pull:`91`.

.. _v0.2.0:

------------------
0.2.0 - 2021-08-12
------------------

This release:

- Improves hydra-zen's `automatic type refinement <https://mit-ll-responsible-ai.github.io/hydra-zen/structured_configs.html#automatic-type-refinement>`_. See :pull:`84` for details
- Cleans up the namespace of ```hydra_zen.typing``. See :pull:`85` for details.

**Compatibility-Breaking Changes**

- The protocol ``hydra_zen.typing.DataClass`` is no longer available in the public namespace, as it is not intended for public use. To continue using this protocol, you can import it from ``hydra_zen.typing._implementations``, but note that it is potentially subject to future changes or removal.


.. _v0.1.0:

------------------
0.1.0 - 2021-08-04
------------------

This is hydra-zen's first stable release on PyPI!
Although we have not yet released version `v1.0.0`, it should be noted that hydra-zen's codebase is thoroughly tested.
Its test suite makes keen use of the property-based testing library `Hypothesis <https://hypothesis.readthedocs.io/en/latest/>`_.
Furthermore, 100% code coverage is enforced on all commits into `main`.

We plan to have an aggressive release schedule for compatibility-preserving patches of bug-fixes and quality-of-life improvements (e.g. improved type annotations).
hydra-zen will maintain a wide window of compatibility with Hydra versions; we test against pre-releases of Hydra and will maintain compatibility with its future releases.


---------
Footnotes
---------
.. [1] The config creation process associated with the decorator is deferred until 
   the config is actually accessed by the store. Thus this decorator pattern does not add substantial runtime overhead to library code until Hydra capabilities are actually utilized.
