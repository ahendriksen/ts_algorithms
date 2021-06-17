# Tomosipo algorithms

A collection of common tomographic reconstruction algorithms
implemented using the tomosipo package.

The following algorithms are implemented:

- FBP
- FDK
- SIRT
- Total-variation minimization


* Free software: GNU General Public License v3
* Documentation: [https://ahendriksen.github.io/ts_algorithms]

## Getting Started

It takes a few steps to setup Tomosipo algorithms on your
machine. We recommend installing
[Anaconda package manager](https://www.anaconda.com/download/) for
Python 3.

### Installing with conda

Install with:
```
# Pytorch, CUDA and ASTRA
conda install -n tomosipo pytorch=1.8 cudatoolkit=10.2 astra-toolbox -c pytorch -c astra-toolbox/label/dev
source activate tomosipo
# Latest Tomosipo develop branch
pip install git+https://github.com/ahendriksen/tomosipo.git@develop
# Tomosipo algorithms
pip install git+https://github.com/ahendriksen/ts_algorithms.git
```

### Running

``` python
import torch
import tomosipo as ts
from ts_algorithms import fbp, sirt, tv_min2d, fdk

# Setup up volume and parallel projection geometry
vg = ts.volume(shape=(1, 256, 256))
pg = ts.parallel(angles=384, shape=(1, 384))
A = ts.operator(vg, pg)

# Create hollow cube phantom
x = torch.zeros(A.domain_shape)
x[:, 10:-10, 10:-10] = 1.0
x[:, 20:-20, 20:-20] = 0.0

# Forward project
y = A(x)

rec_fbp = fbp(A, y)
rec_sirt = sirt(A, y)
```

## Authors and contributors

* **Allard Hendriksen** - *Initial work*
* **Dirk Schut** - *FDK implementation*

See also the list of [contributors](https://github.com/ahendriksen/ts_algorithms/contributors) who participated in this project.

## How to contribute

Contributions are always welcome. Please submit pull requests against the `master` branch.

If you have any issues, questions, or remarks, then please open an issue on GitHub.

## License

This project is licensed under the GNU General Public License v3 - see the [LICENSE.md](LICENSE.md) file for details.
