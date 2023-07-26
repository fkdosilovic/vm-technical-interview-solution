# vm-technical-interview-solution

## Results

| **Model**                  | **Accuracy** | **Precision** | **Recall** |  **F1**  |
| :------------------------- | :----------: | :-----------: | :--------: | :------: |
| **LogisticRegression** (D) |   $0.8844$   |   $0.8915$    |  $0.8828$  | $0.8829$ |
| **LogisticRegression** (O) |              |               |            |          |
| **KNN** (D)                |              |               |            |          |
| **KNN** (O)                |              |               |            |          |
| **Nadaraya-Watson** (D)    |              |               |            |          |
| **Nadaraya-Watson** (O)    |              |               |            |          |

- **D**: Default hyperparameters
- **O**: Optimized hyperparameters

**Hyperparameters**:

| **Model**                  |                         **Hyperparameters**                         |
| :------------------------- | :-----------------------------------------------------------------: |
| **LogisticRegression** (D) | lr=$0.01$, wd=$0.001$, bs=$64$, epochs=$10$, vs=0.16666, ns=$30000$ |
| **LogisticRegression** (O) |                                                                     |
| **KNN** (D)                |                                                                     |
| **KNN** (O)                |                                                                     |
| **Nadaraya-Watson** (D)    |                                                                     |
| **Nadaraya-Watson** (O)    |                                                                     |

- **D**: Default hyperparameters
- **O**: Optimized hyperparameters

## How to run

See `scripts` and run the chosen model.
