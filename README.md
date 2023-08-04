# vm-technical-interview-solution

## Results

The following table shows the performance evaluation results on the default MNIST test set. The models denoted with (D) are trained with the default hyperparameters, while the ones denoted with (O) are trained with the optimized hyperparameters. The hyperparameters are optimized using the validation set. The validation set is obtained by splitting the training set into two parts, where the validation set is $16.666%$ of the training set.

**Performance evaluation on MNIST's test split**

| **Model**                  | **Accuracy** | **Precision** | **Recall** |  **F1**  |
| :------------------------- | :----------: | :-----------: | :--------: | :------: |
| **LogisticRegression** (D) |   $0.8563$   |   $0.8546$    |  $0.8544$  | $0.8543$ |
| **LogisticRegression** (O) |   $0.9034$   |   $0.9035$    |  $0.9018$  | $0.9020$ |
| **KNN** (D)                |   $0.9477$   |   $0.9489$    |  $0.9470$  | $0.9474$ |
| **KNN** (O)                |   $0.9623$   |   $0.9629$    |  $0.9619$  | $0.9621$ |
| **Nadaraya-Watson** (D)    |   $0.9498$   |   $0.9505$    |  $0.9492$  | $0.9494$ |
| **Nadaraya-Watson** (O)    |   $0.9629$   |   $0.9632$    |  $0.9624$  | $0.9627$ |

**Hyperparameters for the experiments**

Random seed for all experiments and models is set to $42$.

| **Model**                  |                    **Hyperparameters**                    |
| :------------------------- | :-------------------------------------------------------: |
| **LogisticRegression** (D) | lr=0.05, wd=0.001, bs=64, epochs=10, vs=0.16666, ns=30000 |
| **LogisticRegression** (O) |  lr=0.1, wd=0.01, bs=64, epochs=10, vs=0.16666, ns=30000  |
| **KNN** (D)                |                       k=3,ns=10000                        |
| **KNN** (O)                |                       k=5,ns=30000                        |
| **Nadaraya-Watson** (D)    |                       k=3,ns=10000                        |
| **Nadaraya-Watson** (O)    |                       k=10,ns=30000                       |

The abbreviations in the table above are as follows: `lr` - learning rate, `wd` - weight_decay, `bs` - batch size, `vs` - percentage for validation split, `ns` - number of training samples.

## Installation

Position your terminal to the root of the project and run the following command:

```bash
pip install .
```

The command will install the package in the current environment as well as all the dependencies.

Download the MNIST dataset from [here](http://yann.lecun.com/exdb/mnist/) and unzip the data
`gzip *ubyte.gz -d`. The data will be unzipped in the same directory and use that directory as the `dataset` argument for the scripts.

## How to run

See `scripts` for the available bash scripts for running the experiments. The scripts are named after the model they are running. For example, to run the optimized Logistic Regression model, run the following command (from the root of the project):

```bash
bash scripts/train_eval_lr_o.sh
```

## Project structure

The project is structured as follows:

- `src/miniml` - a package containing the implementation of the models
- `experiments` - Python scripts for running the experiments (both train-eval and hyperparameter optimization)
- `scripts` - bash scripts for running the experiments
- `notebooks` - Jupyter notebooks containing simple examples of the models
