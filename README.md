# vm-technical-interview-solution

## Results

The following table shows the performance evaluation results on the default MNIST test
set. The models denoted with (D) are trained with the default hyperparameters, while the
ones denoted with (O) are trained with the optimized hyperparameters. The hyperparameters
are optimized using the validation set. The validation set is obtained by splitting the
training set into two parts, where the validation set is 16.666% of the training set.
You can see the hyperparameters in the table below.

| **Model**                  | **Accuracy** | **Precision** | **Recall** |  **F1**  |
| :------------------------- | :----------: | :-----------: | :--------: | :------: |
| **LogisticRegression** (D) |   $0.8844$   |   $0.8915$    |  $0.8828$  | $0.8829$ |
| **LogisticRegression** (O) |              |               |            |          |
| **KNN** (D)                |   $0.7631$   |   $0.8223$    |  $0.7577$  | $0.7643$ |
| **KNN** (O)                |              |               |            |          |
| **Nadaraya-Watson** (D)    |              |               |            |          |
| **Nadaraya-Watson** (O)    |              |               |            |          |

**Hyperparameters**:

Random seed for all experiments and models is set to $42$.

| **Model**                  |                    **Hyperparameters**                    |
| :------------------------- | :-------------------------------------------------------: |
| **LogisticRegression** (D) | lr=0.01, wd=0.001, bs=64, epochs=10, vs=0.16666, ns=30000 |
| **LogisticRegression** (O) |                                                           |
| **KNN** (D)                |                       k=3,ns=10000                        |
| **KNN** (O)                |                                                           |
| **Nadaraya-Watson** (D)    |                                                           |
| **Nadaraya-Watson** (O)    |                                                           |

## How to run

See `scripts` and run the chosen model.
