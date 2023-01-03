.. meta::
   :description: hydra-zen's builds function can be used to create partial configurations for Hydra applications.

.. admonition:: The Answer
   
   Simply use ``builds(<target>, zen_partial=True, [...])`` or ``just(functools.partial(<target>, [...]))`` to create a partial config for ``<target>``.

.. _partial-config:

=============================
Partially Configure an Object
=============================

It is often the case that our configuration of a target-object ought to only partially 
describe its input parameters; while some of the target's parameters can be specified 
via our app's interface, other parameters may need to be specified only after our app 
has launched.


In this How-To we will write a config that only partially configures a target object.
We will

1. Define toy examples of a class and a function, respectively.
2. Use ``builds(<target>, zen_partial=True)`` and ``just(functools.partial(<target>))`` to create their partial configurations.
3. Check that :func:`~hydra_zen.instantiate` only partially-instantiates the targets of these two configs.
4. Examine the YAML-config for one of these partial configs.


Let's define toy examples of a class and a function; we'll create partial configs for 
both of these.

.. code-block:: python
   :caption: 1: Defining toy examples.
   
   class Optimizer:
       def __init__(self, model_weights, learning_rate):
           self.weights = model_weights
           self.lr = learning_rate

   def logger(message: str, format_spec: str) -> str:
       return format_spec.format(message)

Suppose that the ``model_weights`` parameter in :class:`Optimizer`, and the ``message`` parameter in :func:`logger` ought not be configured by our app's interface. 
:func:`~hydra_zen.builds` provides the ``zen_partial`` feature, which makes it trivial 
to partially-configure a target. We can also apply :func:`~hydra_zen.just` to a instance of :py:class:`functools.partial` Let's create configs for our toy class and 
toy function.

.. code-block:: python
   :caption: 2: Creating partial configs.
   
   from functools import partial
   from hydra_zen import builds, instantiate, just
   
   OptimConf = builds(Optimizer, learning_rate=0.1, zen_partial=True)
   
   partial_logger = partial(logger, format_spec='{0:>8s}')
   LogConf = just(partial_logger)

Instantiating these configs will apply :func:`functools.partial` to the config's target.

.. code-block:: pycon
   :caption: 3: Instantiating the configs
   
   >>> partiald_optim = instantiate(OptimConf)
   >>> partiald_optim
   functools.partial(<class '__main__.Optimizer'>, learning_rate=0.1)
   
   >>> partiald_optim(model_weights=[1, 2, 3])
   <__main__.Optimizer at 0x189991fbb50>

   >>> partiald_logger = instantiate(LogConf)
   >>> partiald_logger
   functools.partial(<function logger at 0x0000018998D1E040>, format_spec='{0:>8s}')
   
   >>> partiald_logger("hello")
   '   hello'
   >>> partiald_logger("goodbye")
   ' goodbye'

Lastly, let's inspect the YAML-serialized config for :class:`OptimConf` and :class:`LongConf`.

.. code-block:: pycon
   :caption: 4: Examining a YAML-serialized partial config.

   >>> from hydra_zen import to_yaml

   >>> print(to_yaml(OptimConf))
   _target_: __main__.Optimizer
   _partial_: true
   learning_rate: 0.1

   >>> print(to_yaml(LogConf))
   _target_: __main__.logger
   _partial_: true
   format_spec: '{0:>8s}'