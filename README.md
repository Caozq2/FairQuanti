# FairQuanti: Enhancing Fairness in Deep Neural Network Quantization via Neuron Role Contribution

This repository contains the implementation of the paper **"FairQuanti: Enhancing Fairness in Deep Neural Network Quantization via Neuron Role Contribution"**. The project focuses on improving fairness in deep neural network quantization by considering the role and contribution of individual neurons.

---

## 📁 Project Structure

- `train.py`: Script for training the neural network models on fairness-sensitive datasets.
- `main.py`: Script for implementing the FairQuanti method and evaluating its performance.
- `evaluate.py`: Utilities for model evaluation and metrics computation.
- `models.py`: Definitions of neural network architectures (e.g., `Net`, `Net_CENSUS`, `NetPlus_`).
- `utils/`: Helper functions for dataset transformation and preprocessing.
- `data/`: Directory for storing datasets (e.g., COMPAS, Census, Credit).
- `results/`: Directory for saving trained models and outputs.

---

## 🛠 Requirements

Make sure you have Python 3.8+ installed. Install dependencies with:

```bash
pip install -r requirements.txt
```

## Datasets

The code supports three datasets:

- **COMPAS**: Recidivism prediction dataset.
- **Census**: Adult income prediction dataset.
- **Credit**: German credit risk dataset.

Place the dataset files in the `data/` directory with the following structure:

```plaintext
data/
├── COMPAS/compas_recidive_two_years_sanitize_age_category_jail_time_decile_score.csv
├── Census/adult
└── Credit/german_credit
```

## Usage

### 1. Training the Model (`train.py`)

Use `train.py` to train a neural network on a specified dataset. Below are example commands:

- **Train on COMPAS dataset:**
  ```bash
  python train.py --dataset compas --batch-size 128 --bit 8 --input-path ./data/COMPAS/compas_recidive_two_years_sanitize_age_category_jail_time_decile_score.csv --save-dir ./results
  ```

- **Train on Census dataset:**
  ```bash
  python train.py --dataset census --batch-size 128 --bit 8 --input-path ./data/Census/adult --save-dir ./results
  ```

- **Train on Credit dataset:**
  ```bash
  python train.py --dataset credit --batch-size 128 --bit 8 --input-path ./data/Credit/german_credit --save-dir ./results
  ```

### 2. Implementing FairQuanti (`main.py`)

Use `main.py` to implement the FairQuanti method and evaluate its performance. Below are example commands:

- **Run FairQuanti on COMPAS dataset:**

    ```bash
    python main.py --dataset compas --batch-size 128 --bits 8 --input-path ./data/COMPAS/compas_recidive_two_years_sanitize_age_category_jail_time_decile_score.csv --save-dir ./results
    ```

- **Run FairQuanti on Census dataset:**

    ```bash
    python main.py --dataset census --batch-size 128 --bits 8 --input-path ./data/Census/adult --save-dir ./results
    ```

- **Run FairQuanti on Credit dataset:**

    ```bash
    python main.py --dataset credit --batch-size 128 --bits 8 --input-path ./data/Credit/german_credit --save-dir ./results
    ```

## Results

The results are saved in the `results/` directory. The results include the trained model, the quantized model, and the evaluation metrics.

## Citation

If you find this work useful, please cite it as follows:

```bibtex
@article{fairquanti,
  title={FairQuanti: Enhancing Fairness in Deep Neural Network Quantization via Neuron Role Contribution},
  author={Zhiqi Cao and Jingjing Li and Jingwei Li and Jingren Zhou},
  year={2025},
}   
```     
