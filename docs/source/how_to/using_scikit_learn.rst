.. meta::
   :description: Experimenting with SciKit Learn.


===================
Using SciKit Learn
===================

This guid will demonstrate how to use hydra-zen with `SciKit-Learn <https://scikit-learn.org/stable/index.html>`_.  We will
demonstrate how to configure multiple SciKit-Learn datasets and classifiers [1]_ using hydra-zen and launching multiple
experiments using Hydra's CLI.  In this How-To we will:

1. Configure multiple datasets and classifiers.
2. Build a task function to load data, fit a classifier, and plot the result.
3. Save the figure in the local experiment directory.
4. Gather the saved figures after executing the experiments and plot the final result.


Configuring and Building an Experiment
======================================

Below we create configs for three datasets and ten classifiers and add them to :class:`hydra-zen's config store <hydra_zen.ZenStore>`.
After building the configs we define the task function for loading 

.. code-block:: python
   :caption: Application: my_app.py

   import numpy as np
   import matplotlib.pyplot as plt
   from matplotlib.colors import ListedColormap
   from sklearn.model_selection import train_test_split
   from sklearn.preprocessing import StandardScaler
   from sklearn.pipeline import make_pipeline
   from sklearn.datasets import make_moons, make_circles, make_classification
   from sklearn.neural_network import MLPClassifier
   from sklearn.neighbors import KNeighborsClassifier
   from sklearn.svm import SVC
   from sklearn.gaussian_process import GaussianProcessClassifier
   from sklearn.gaussian_process.kernels import RBF
   from sklearn.tree import DecisionTreeClassifier
   from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
   from sklearn.naive_bayes import GaussianNB
   from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
   from sklearn.inspection import DecisionBoundaryDisplay

   from hydra_zen import builds, make_config, store, zen, load_from_yaml

   # 1. Configuring multiple datasets and classifiers

   #
   # Stores for classifier and dataset groups
   #
   sklearn_store = store(group="classifier")
   sklearn_dataset = store(group="dataset")

   #
   # Build config and store classifiers
   #
   sklearn_store(KNeighborsClassifier, n_neighbors=3, name="knn")
   sklearn_store(SVC, kernel="linear", C=0.025, name="svc_linear")
   sklearn_store(SVC, gamma=2, C=1, name="svc_rbf")
   sklearn_store(
       GaussianProcessClassifier,
       kernel=builds(RBF, length_scale=1.0),
       name="gp",
   )
   sklearn_store(DecisionTreeClassifier, max_depth=5, name="decision_tree")
   sklearn_store(
       RandomForestClassifier,
       max_depth=5,
       n_estimators=10,
       max_features=1,
       name="random_forest",
   )
   sklearn_store(MLPClassifier, alpha=1, max_iter=1000, name="mlp")
   sklearn_store(AdaBoostClassifier, name="ada_boost")
   sklearn_store(GaussianNB, name="naive_bayes")
   sklearn_store(QuadraticDiscriminantAnalysis, name="qda")

   # Build config and store datasets
   def linearly_separable_dataset():
       X, y = make_classification(
           n_features=2,
           n_redundant=0,
           n_informative=2,
           random_state=1,
           n_clusters_per_class=1,
       )
       rng = np.random.RandomState(2)
       X += 2 * rng.uniform(size=X.shape)
       return X, y


   sklearn_dataset(
       builds(linearly_separable_dataset, zen_partial=True),
       name="linear",
   )
   sklearn_dataset(
       builds(make_moons, noise=0.3, random_state=0, zen_partial=True), name="moons"
   )
   sklearn_dataset(
       builds(make_circles, noise=0.2, factor=0.5, random_state=1, zen_partial=True),
       name="circles",
   )

   # Task configuration
   store(
       make_config(
           hydra_defaults=["_self_", {"dataset": "moons"}, {"classifier": "knn"}],
           dataset=None,
           classifier=None,
       ),
       name="config",
   )

   # 2. Build a task function to load data, fit a classifier, and plot the result.

   def task(dataset, classifier):
       fig, ax = plt.subplots()

       # split data for train and test
       X, y = dataset()
       X_train, X_test, y_train, y_test = train_test_split(
           X, y, test_size=0.4, random_state=42
       )

       # plot the data
       x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
       y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5

       # just plot the dataset first
       cm = plt.cm.RdBu
       cm_bright = ListedColormap(["#FF0000", "#0000FF"])

       # Plot the training points
       ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright, edgecolors="k")

       # Plot the testing points
       ax.scatter(
           X_test[:, 0],
           X_test[:, 1],
           c=y_test,
           cmap=cm_bright,
           alpha=0.6,
           edgecolors="k",
       )

       clf = make_pipeline(StandardScaler(), classifier)
       clf.fit(X_train, y_train)
       score = clf.score(X_test, y_test)
       DecisionBoundaryDisplay.from_estimator(clf, X, cmap=cm, alpha=0.8, ax=ax, eps=0.5)

       ax.set_xlim(x_min, x_max)
       ax.set_ylim(y_min, y_max)
       ax.set_axis_off()
       ax.text(
           x_max - 0.3,
           y_min + 0.3,
           ("%.2f" % score).lstrip("0"),
           size=25,
           horizontalalignment="right",
       )

       # load overrides to set plot title   
       overrides = load_from_yaml(".hydra/overrides.yaml")


       # 3. Save the figure in the local experiment directory.
       if len(overrides) == 2:
           dname = overrides[0].split("=")[1]
           cname = overrides[1].split("=")[1]
           fig.savefig(f"{dname}_{cname}.png", pad_inches=0.0, bbox_inches = 'tight')
       else:
           fig.savefig("result.png", pad_inches=0.0, bbox_inches = 'tight')

   # For hydra multirun figures will stay open until all runs are completed
   # if we do not close the figure
   plt.close()


   if __name__ == "__main__":
       store.add_to_hydra_store()
       zen(task).hydra_main(config_path=None, config_name="config")

We can run the default experiment with:

.. code-block:: bash

   $ python my_app.py

Hydra will execute the experiment and the resulting figure will be saved in the experiment
directory.  Below is the directory structure of saved results.

.. code-block:: text

   output
     |
     --<date>
         |
         result.png
         .hydra
           |
           overrides.yaml
           config.yaml
           hydra.yaml

To run over all configured datasets and models:

.. code-block:: bash

   $ python my_app.py dataset=glob("*") classifier=glob(*) --multirun

A total of 30 jobs will execute for this multirun where each experiment
is stored in the following directory structure:

.. code-block:: text

   multirun
     |
     --<date>
         |
         --<job number: e.g., 0>
               |
               <dataset_name>_<classifier_name>.png
               .hydra
                 |
                 overrides.yaml
                 config.yaml
                 hydra.yaml

Gathering and Visualizing the Results
=====================================

To load images and visualize the results simply load in all `png` files
stored in job directories and plot the results.

.. code-block:: python
   :caption: 4. Gathering and Plotting Results

   import matplotlib.pyplot as plt
   import matplotlib.image as mpimg


   from pathlib import Path

   images = sorted(
       Path("multirun/2023-01-20/10-26-06").glob("**/*.png"),
       # sort by dataset name
       key=lambda x: str(x.name).split(".png")[0].split("_")[0],
   )

   fig, ax = plt.subplots(
       ncols=10,
       nrows=3,
       figsize=(18, 4),
       tight_layout=True,
       subplot_kw=dict(xticks=[], yticks=[]),
   )


   for i, image in enumerate(images):
       dname, cname = image.name.split(".png")[0].split("_", 1)
       image = str(image)

       img = mpimg.imread(image)

       row = i // 10
       col = i % 10
       # ax[row, col].set_axis_off()
       ax[row, col].imshow(img)

       if row == 0:
           ax[row, col].set_title(cname)

       if col == 0:
           ax[row, col].set_ylabel(dname)

The resulting figure should be:

.. image:: scikit_learn.png 


Footnotes
=========
.. [1] This closely mirrors SciKit-Learn's (`Classifier Comparison <https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html#sphx-glr-auto-examples-classification-plot-classifier-comparison-py>`_ ) example.  We emphasize the ability to configure multiple datasets and classifiers using hydra-zen and launching multiple experiments using Hydra CLI.
