# Crack Deflection Quantification

Welcome to the documentation of the crack deflection quantification code as described in "Unraveling the effect of collagen damage on bone fracture using in situ synchrotron microtomography with deep learning." We will describe how the code works step-by-step using an example crack segmentation through an interactive Jupyter notebook. We recommend follow the code using this notebook after getting set up.

## Getting started
---
We recommend starting with an installation of conda or Anaconda. This will make building the environment for running the code easier. It's free and you can find it [here](https://www.anaconda.com/).

You can type the following to build the environment in conda using the included environment.yml file.

```
conda env create -f environment.yml
```

Following this, activate the environment and launch jupyter lab using these two commands:
```
conda activate crack_deflection
jupyter lab
```

From here, follow along with the notebook.

## Usage
---
Included in the repo are is the main example notebook as well as two .py files that include helpful functions for formatting crack image data and calculating crack deflections. These python files may also be of use for implementations of calculating crack deflections by others or if something is not straightforward in the notebook.
