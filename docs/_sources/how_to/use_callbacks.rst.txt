.. meta::
   :description: Using callbacks with hydra-zen.

.. _callbacks:

=======================================================
Use Hydra's Callbacks to Run Code Before and After Jobs
=======================================================

Hydra's `callback system <https://hydra.cc/docs/experimental/callbacks/>`_ lets us run 
custom code that is triggered by events, such as a job starting and a job completing. 
This enables us to do things like upload a job's results to cloud storage or 
turn on performance profiling in a configurable and modular way. These callbacks can be 
used across applications - independent of our task function and its config.

In this How-To, we will write toy versions of two such callbacks and will incorporate 
them in our hydra-zen code. First, we will hardcode our application to use these 
callbacks, and then we will rewrite things so that the callbacks can be enabled from the CLI.


Adding basic callback support to an application
===============================================

Here we define two callbacks - `TimeIt` and `UploadResultsCallback` [1]_ - and manually 
add them to :ref:`Hydra's config <HydraConf>`.

.. code-block:: python
   :caption: Contents of `my_app.py`- two callbacks are defined and added to Hydra's config.

   import time
   from dataclasses import dataclass
   from hydra.experimental.callback import Callback
   
   # The config for our task function
   @dataclass
   class Config:
       x: int
   
   # Our task function
   def task(x: int):  # just an example task function - no important details
      print(f".. running task({x=})")
      import random
      time.sleep(random.random())

   # Defining our callbacks
   class TimeIt(Callback):
       def on_job_start(self, **kw) -> None:  # type: ignore
           self._start = time.time()
   
       def on_job_end(self, **kw) -> None:  # type: ignore
           print(f"TimeIt: Took {round(time.time() - self._start, 2)} seconds")
   
   
   class UploadResultsCallback(Callback):
       def __init__(self, *, bucket: str = "s3:/") -> None:
           self.bucket = bucket
   
       def on_job_end(self, config: Config, **kwargs) -> None:  # type: ignore
           # Leverage access to the job's config to create a distinct file path.
           path = f"file_{config.x}.txt"
           print(f"UploadResultsCallback: Job ended, uploading results to {self.bucket}/{path}")

   if __name__ == "__main__":
       from hydra.conf import HydraConf
       from hydra_zen import make_custom_builds_fn, zen, ZenStore

       fbuilds = make_custom_builds_fn(populate_full_signature=True)

       store = ZenStore()
       # Add our callbacks directly to Hydra's config and add it to our 
       # config store.
       store(
           HydraConf(
               callbacks={
                   "upload": fbuilds(UploadResultsCallback),
                   "timeit": fbuilds(TimeIt),
               },
           )
       )
       # Add our task function's config to the store
       store(Config, name="task")  
       store.add_to_hydra_store()
       
       # Expose CLI for running `task`
       zen(task).hydra_main(
           config_path=None,
           config_name="task",
           version_base="1.3",
       )

When we run `my_app` we should see that both of our callbacks are running.
Let's do a multirun over two values of `x`.

.. code-block:: console
   :caption: Running our application using the default config.

   $ python my_app.py x=1,2 -m
   [2023-11-19 13:54:22,232][HYDRA] Launching 2 jobs locally
   [2023-11-19 13:54:22,232][HYDRA]        #0 : x=1
   .. running task(x=1)
   TimeIt: Took 0.13 seconds
   UploadResultsCallback: Job ended, uploading results to s3://file_1.txt

   [2023-11-19 13:54:22,481][HYDRA]        #1 : x=2
   .. running task(x=2)
   TimeIt: Took 0.72 seconds
   UploadResultsCallback: Job ended, uploading results to s3://file_2.txt


We can override the default bucket for `UploadResultsCallback`.

.. code-block:: console
   :caption: Running with `UploadResultsCallback(bucket='gcp:/')`.

   $ python my_app.py x=1,2 hydra.callbacks.upload.bucket='gcp:/' -m
   [2023-11-19 14:00:46,350][HYDRA] Launching 2 jobs locally
   [2023-11-19 14:00:46,350][HYDRA]        #0 : x=1
   .. running task(x=1)
   TimeIt: Took 0.49 seconds
   UploadResultsCallback: Job ended, uploading results to gcp://file_1.txt

   [2023-11-19 14:00:46,981][HYDRA]        #1 : x=2
   .. running task(x=2)
   TimeIt: Took 0.9 seconds
   UploadResultsCallback: Job ended, uploading results to gcp://file_2.txt


We can disable the `TimeIt` callback.

.. code-block:: console
   :caption: Disabling `TimeIt` from the CLI.

   $ python my_app.py x=1,2 ~hydra.callbacks.timeit -m
   [2023-11-19 14:01:42,093][HYDRA] Launching 2 jobs locally
   [2023-11-19 14:01:42,093][HYDRA]        #0 : x=1
   .. running task(x=1)
   UploadResultsCallback: Job ended, uploading results to s3://file_1.txt

   [2023-11-19 14:01:42,256][HYDRA]        #1 : x=2
   .. running task(x=2)
   UploadResultsCallback: Job ended, uploading results to s3://file_2.txt


Enabling callbacks from the CLI
===============================

Suppose that we do not want our callbacks to be enabled by default, and that we would 
prefer to turn callbacks on from the CLI. To do this, we can add our callbacks to a 
'callbacks' group in our :class:`~hydra_zen.ZenStore`, and then leverage Hydra's 
`group@pkg` `override <https://hydra.cc/docs/advanced/override_grammar/basic/>`_.


.. code-block:: python
   :caption: Modifying `__main__` in `my_app.py`

   # Config, TimeIt, UploadResultsCallback, and task are unchanged

   if __name__ == "__main__":
       from hydra_zen import zen, ZenStore
   
       store = ZenStore()
       
       # Create configs for our callbacks and store them under the 'callbacks' group
       store(UploadResultsCallback, name="upload", group="callbacks")
       store(TimeIt, name="timeit", group="callbacks")
       
       store(Config, name="task")
       store.add_to_hydra_store()
   
       zen(task).hydra_main(
           config_path=None,
           config_name="task",
           version_base="1.3",
       )


By default, running our app no longer includes any callbacks.

.. code-block:: console
   :caption: Running my_app without any callbacks.

   $ python my_app.py x=1,2 -m
   [2023-11-19 14:01:42,093][HYDRA] Launching 2 jobs locally
   [2023-11-19 14:01:42,093][HYDRA]        #0 : x=1
   .. running task(x=1)

   [2023-11-19 14:01:42,256][HYDRA]        #1 : x=2
   .. running task(x=2)

Let's enable both callbacks from the CLI *and* configure `UploadResultsCallback(bucket='gcp:/')`.

.. code-block:: console
   :caption: Running my_app with both callbacks enabled and `UploadResultsCallback(bucket='gcp:/')`.

   $ python my_app.py x=1,2 \
            +callbacks@hydra.callbacks.timeit=timeit \
            +callbacks@hydra.callbacks.upload=upload \
            hydra.callbacks.upload.bucket=gcp:/ \
            -m
   [2023-11-19 14:15:41,282][HYDRA] Launching 2 jobs locally
   [2023-11-19 14:15:41,282][HYDRA]        #0 : x=1 +callbacks@hydra.callbacks.timeit=timeit +callbacks@hydra.callbacks.upload=upload
   .. running task(x=1)
   UploadResultsCallback: Job ended, uploading results to gcp://file_1.txt
   TimeIt: Took 0.21 seconds

   [2023-11-19 14:15:41,617][HYDRA]        #1 : x=2 +callbacks@hydra.callbacks.timeit=timeit +callbacks@hydra.callbacks.upload=upload
   .. running task(x=2)
   UploadResultsCallback: Job ended, uploading results to gcp://file_2.txt
   TimeIt: Took 0.39 seconds

While the input here isn't all that concise it is nonetheless important to see that 
callbacks can be enabled and configured without having to modify one's code.


Footnotes
=========
.. [1] See `this code <https://github.com/facebookresearch/hydra/blob/809718cdcd64f9cd930d26dea69f2660a6ffa833/hydra/experimental/callback.py#L13-L65>`_ for the full `Callback` API.
