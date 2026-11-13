# MULAR
MULti-layer and ARbitrary-layer repair


## Table of Contents
- [Installation Guide](#installation-guide)
- [Example](#example)
- [Small Experiment](#small-experiment)
- [Reproducing Materials](#reproducing-materials)
- [Executing Experiments](#executing-experiments)
- [Project Structure](#project-structure)



## Installation Guide
### 1. Clone the repository

[repository](https://anonymous.4open.science/r/Provable-Repair-2682) is in here. Clone it to your local machine.


### 2. Install Gurobi (solver)

Reproducing experiments requires a Gurobi license. Please install Gurobi from the official website: [gurobi installation](https://www.gurobi.com/). Free academic licenses for students and researchers [Gurobi academic license](https://www.gurobi.com/academia/academic-program-and-licenses) are provided if needed.

Aside from the official instructions, the following steps might be helpful.

- Login to the Gurobi user portal.
- Go to the ["Licenses - Request" tab](https://portal.gurobi.com/iam/licenses/request), genearte a "WLS Academic" license if you don't have one. If you already have a "WLS Academic" license, you might get an "[LICENSES_ACADEMIC_EXISTS] Cannot create academic license as other academic licenses already exists" error.
- Go to the "Home" tab, click "Licenses - Open the WLS manager" to open the WLS manager.
- In the WLS manager, you should see a license under the "Licenses" tab. Click "extend" if it has expired (it might take some time to take effect).
- Go to the "API Keys" tab, click the "CREATE API KEY" button to create a new license, download the generated `gurobi.lic` file by following the instructions and place it at the proper location.

Before running the experiments, ensure that your Gurobi license is properly installed and gurobipy works in Python.


### 3. Install uv (python environment manager)
Please install by following guide: [uv installation](https://github.com/astral-sh/uv?tab=readme-ov-file#installation)

The following command might be helpful for installation:

- For macOS/Linux:
```
curl -LsSf https://astral.sh/uv/install.sh | sh
```
- For Windows (PowerShell):
```
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

After installation, ensure that `uv` command is available in your terminal. You might need to run the command shown in the output of the installation script.


### 4. Setup python version
The project requires Python 3.12. Please install and pin the version using uv:
```
uv python install 3.12
cd [repository folder]
uv python pin 3.12
```

### 5. Create uv environment and install dependencies
```
uv sync
```
This command:
- creates a virtual environment (.venv)
- installs all dependencies from `pyproject.toml`
- ensure the environment uses Python 3.12

### 6. Install Datasets
In our experiments, we use the following datasets:
- CIFAR10
- [MNIST-C](https://zenodo.org/records/3239543)
- [CIFAR10-C](https://zenodo.org/records/2535967)
- [GTSRB](https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign)

CIFAR10 is already included in the `torchvision` package, so you don't need to download it separately.

For MNIST-C and CIFAR10-C, you can download the datasets by running the following commands:
```
mkdir data
cd data

# mnist-c
wget https://zenodo.org/record/3239543/files/mnist_c.zip
unzip mnist_c.zip

# cifar10-c
wget -O cifar10_c.tar https://zenodo.org/records/2535967/files/CIFAR-10-C.tar
tar -xf cifar10_c.tar
```
For GTSRB, you can download the dataset from Kaggle. You need to create a Kaggle account and generate an API token. After downloading, make sure to rename the folder to `gtsrb` and place it in the `data` folder.



## Example (smoke test)

The self-contained example used in Section 3 illustrates why modifying only the last layer can fail and how MULAR repairs the same network by modifying multiple layers. It uses a network for two inputs and one output with one hidden layer and ReLU activation, with weights:

$$
W_1=\begin{bmatrix}2&-1\\-2&1\end{bmatrix},\qquad
W_2=\begin{bmatrix}3&-3\end{bmatrix},
$$

over the integer-bounded input region $[3,8]\times[0,8]$, with the desired
property $o\geq3$ and a per-weight modification limit of $0.5$.

Run the example from the repository root:

```bash
uv run python example/track_repair.py
```

The script reports every bound-propagation, safe-subspace search, and repair
step. The expected progression is:

| Method or stage | Certified output bounds | Result |
|---|---:|---|
| Original network | $[-6,48]$ | Failure |
| Last-layer-only repair | $[-5,48]$ | Failure |
| After the first-layer subrepair | $[21/8,39]$ | Failure |
| Completed multilayer repair | $[49/16,91/2]$ | Success |

To save the immediately flushed execution trace, run:

```bash
uv run python example/track_repair.py \
  --output example/repair_trace.txt
```


## Small Experiment

After installing the dependencies, run the lightweight end-to-end smoke test
from the repository root:

```bash
uv run python smoke_test.py
```

The test runs two region-wise repair instances on MNIST-F1 for all approaches
(APRNN, ProRepair, PREPARED, LLR, and MULAR), using one repair region with five
perturbed input dimensions. It stops immediately if an approach exits with an
error and saves the normal approach-specific logs under `result/`.

The experiment size and approaches can also be changed, for example:

```bash
uv run python smoke_test.py \
  --num-runs 1 \
  --ndims 5 \
  --approaches LLR MULAR
```



## Executing Experiments
### APRNN
```
uv run test_aprnn.py --net-id 0
```
You can switch the network by changing the `--net-id` argument. We currently support the following networks:
- net-id 0: mnist f1
- net-id 1: mnist f2

\# perturbed dimensions is already set to (5, 6, 7, 8, 9, 10, 12, 14, 16, 18, 20) in the code. You can change it by modifying the `ndims_list` variable in `test_prorepair.py`.

### ProRepair
```
uv run test_prorepair.py --exe-id 0
```
You can switch the experiment by changing the `--exe-id` argument. We currently support the following experiments:
- exe-id 0: (mnist f1, nonzero)
- exe-id 1: (mnist f1, all)
- exe-id 2: (mnist f2, nonzero)
- exe-id 3: (mnist f2, all)

where (model_name, pick) respectively corresponds to the model and the type of perturbation. As described in the paper, we execute the experiments with different numbers of perturbed dimensions (\#DIM) and different numbers of perturbed boxes (\#P). `nonzero` and `all` respectively correspond to the two types of perturbations. The perturbed dimensions are already set to (5, 6, 7, 8, 9, 10, 12, 14, 16, 18, 20) in the code. You can change it by modifying the `ndims_list` variable in `test_prorepair.py`. The number of perturbed boxes is already set to (1, 2, 3, 4, 5, 6, 8, 10, 12, 14) in the code. You can change it by modifying the `N_list` variable in `test_prorepair.py`.

### PREPARED, LLR (last layer repair)
For PREPARED:
```
uv run test_prepared.py --exe-id 0
```
For LLR (last layer repair):
```
uv run test_lastlayer.py --exe-id 0
```
You can switch the experiment by changing the `--exe-id` argument. We currently support the following experiments:
- exe-id 0: (mnist-f1, nonzero)
- exe-id 1: (mnist-f1, all)
- exe-id 2: (mnist-f2, nonzero)
- exe-id 3: (mnist-f2, all)
- exe-id 4: (cifar-c, all)
- exe-id 5: (gtsrb-c, all)

where (model_name, pick) respectively corresponds to the model and the type of perturbation. As "ProRepair" experiments, we already set the perturbed dimensions to (5, 6, 7, 8, 9, 10, 12, 14, 16, 18, 20) in the code. You can change it by modifying the `ndims_list` variable in `test_prepared.py` (`test_lastlayer.py`). The number of perturbed boxes is already set to (1, 2, 3, 4, 5, 6, 8, 10, 12) for "mnist-f1" and "mnist-f2", (12, 14, 16, 18, 20) for "cifar-c", and (4, 6, 8, 10, 12) for "gtsrb-c" in the code. You can change it by modifying the `N_list` variable in `test_prepared.py` (`test_lastlayer.py`).

### MULAR (our method)
```
uv run test.py --exe-id 0
```
You can switch the experiment by changing the `--exe-id` argument. We currently support the following experiments:
- exe-id 0: (mnist-f1, nonzero)
- exe-id 1: (mnist-f1, all)
- exe-id 2: (mnist-f2, nonzero)
- exe-id 3: (mnist-f2, all)
- exe-id 4: (cifar-c, all)
- exe-id 5: (gtsrb-c, all)

where (model_name, pick) respectively corresponds to the model and the type of perturbation. As experiments so far, we already set the perturbed dimensions to (5, 6, 7, 8, 9, 10, 12, 14, 16, 18, 20) in the code. You can change it by modifying the `ndims_list` variable in `test.py`. The number of perturbed boxes is already set to (1, 2, 3, 4, 5, 6, 8, 10, 12) for "mnist-f1" and "mnist-f2", (12, 14, 16, 18, 20) for "cifar-c", and (4, 6, 8, 10, 12) for "gtsrb-c" in the code. You can change it by modifying the `N_list` variable in `test.py`.


#### Specifing target layers to repair
You can specify the target layers to repair by using the `--repair-layer-config` argument.
For example, if you want to repair the first and second layers, you can run:
```
uv run test.py --exe-id 0 --repair-layer-config last_2
```
or you can manually specify the layers to repair
```
uv run test.py --exe-id 0 --repair-layer-indices 5 7
```




## Project Structure
```
sabre/
 ├─ test.py    # Entry point for MULAR experiments
 ├─ test_lastlayer.py    # Entry point for LLR experiments
 ├─ test_prepared.py    # Entry point for PREPARED experiments
 ├─ test_prorepair.py    # Entry point for ProRepair experiments
 ├─ test_aprnn.py    # Entry point for APRNN experiments
 ├─ repair/    # Repair modules of MULAR
 ├─ LastLayerRepair/    # Repair modules of LLR
 ├─ PREPARED/    # Repair modules of PREPARED
 ├─ ProRepair/    # Repair modules of ProRepair
 ├─ network_bound/    # Approximated bounding modules
 ├─ dual_network/    # Helper modules for bounding
 ├─ LPsolver/    # LP solver modules for repair optimization
 ├─ input_space/    # Construct input properties for repair tasks
 ├─ pyproject.toml    # Project dependencies
 ├─ README.md
 └─ (others)/  # Other modules and files, except main components above
```
