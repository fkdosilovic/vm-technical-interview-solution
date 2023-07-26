# vm-technical-interview-solution

## Results

The following table shows the performance evaluation results on the default MNIST test set. The models denoted with (D) are trained with the default hyperparameters, while the ones denoted with (O) are trained with the optimized hyperparameters. The hyperparameters are optimized using the validation set. The validation set is obtained by splitting the training set into two parts, where the validation set is $16.666%$ of the training set.

**Performance evaluation on MNIST's test split**

| **Model**                  | **Accuracy** | **Precision** | **Recall** |  **F1**  |
| :------------------------- | :----------: | :-----------: | :--------: | :------: |
| **LogisticRegression** (D) |   $0.8563$   |   $0.8546$    |  $0.8544$  | $0.8543$ |
| **LogisticRegression** (O) |   $0.9034$   |   $0.9035$    |  $0.9018$  | $0.9020$ |
| **KNN** (D)                |   $0.9477$   |   $0.9489$    |  $0.9470$  | $0.9474$ |
| **KNN** (O)                |              |               |            |          |
| **Nadaraya-Watson** (D)    |   $0.9498$   |   $0.9505$    |  $0.9492$  | $0.9494$ |
| **Nadaraya-Watson** (O)    |              |               |            |          |

**Hyperparameters for the experiments**

Random seed for all experiments and models is set to $42$.

| **Model**                  |                    **Hyperparameters**                    |
| :------------------------- | :-------------------------------------------------------: |
| **LogisticRegression** (D) | lr=0.05, wd=0.001, bs=64, epochs=10, vs=0.16666, ns=30000 |
| **LogisticRegression** (O) |  lr=0.1, wd=0.01, bs=64, epochs=10, vs=0.16666, ns=30000  |
| **KNN** (D)                |                       k=3,ns=10000                        |
| **KNN** (O)                |                                                           |
| **Nadaraya-Watson** (D)    |                       k=3,ns=10000                        |
| **Nadaraya-Watson** (O)    |                                                           |

The abbreviations in the table above are as follows: `lr` - learning rate, `wd` - weight_decay, `bs` - batch size, `vs` - percentage for validation split, `ns` - number of training samples.

## Installation

Position your terminal to the root of the project and run the following commands:

```bash
pip install -r requirements.txt
pip install .
```

The first command installs the necessary dependencies, while the second command installs the package build specifically for the project.

## How to run

See `scripts` and run the chosen model.
