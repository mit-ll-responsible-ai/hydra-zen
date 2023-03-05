.. meta::
   :description: Configuring Hydra.


===============================
Customize Hydra's Configuration
===============================

Hydra is highly configurable. The way that Hydra runs jobs, logs information, manages job directories are among its many configurable components [1]_. This How-To guide demonstrates how to configure Hydra from within a Python module [2]_.


To configure Hydra using hydra-zen we must update the default instance of :class:`hydra.conf.HydraConf` (which is a :py:func:`dataclasses.dataclass`).
Thus we can simply create our own :class:`hydra.conf.HydraConf` instance and overwrite store's entry; `ZenStore` will automatically place this under the appropriate `group="hydra", name="config"` entry in Hydra's config store.

In this example, we configure Hydra to automatically change the runtime working directory to that of the job's output directory. I.e., we set `hydra.job.chdir=True`:


.. code-block:: python
   :caption: Contents of `my_app.py`

   from hydra_zen import zen, store
   from hydra.conf import HydraConf, JobConf

   def task():  # just an example task function - no important details
       import os
       print(f"working dir: {os.getcwd()}")

   if __name__ == "__main__":
       store(HydraConf(job=JobConf(chdir=True)))  # store `hydra.job.chdir=True`
       store(task)  # create & store config for our task fn
       store.add_to_hydra_store()

       zen(task).hydra_main(config_path=None, config_name="task", version_base="1.2")


Let's confirm that we have successfully changed Hydra's configuration. By default,
our app's working dir should be job's the time-stamped output directory.

.. code-block:: console
   :caption: Running our application using the default config.

   $ pwd
   foo/

   $ python my_app.py
   working dir: foo/outputs/2023-01-20/17-48-19

   $ python my_app.py hydra.job.chdir=False
   working dir: foo


Footnotes
=========
.. [1] See `this documentation <https://hydra.cc/docs/configure_hydra/intro/>`_ to learn about all of the ways that Hydra can be configured.
.. [2] Hydra's documentation for configuring Hydra focuses on yaml-based and CLI-based approaches. In contrast, hydra-zen promotes a pure-Python approach to Hydra and thus this guide follows suite.