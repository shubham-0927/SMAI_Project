
# Chess Analysis Using AI Methodology

## SMAI Project – CS7.403b | Team 32

# Model Training and Testing Notebook

This repository contains a Jupyter Notebook (`model-training-testing.ipynb`) that demonstrates a complete pipeline for training and testing a machine learning model. The notebook includes data preprocessing, model training, evaluation, and visualization of results.

---

## 🛠️ Requirements

Make sure the following are installed:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

---

## 🚀 How to Run the Notebook

1. Clone this repository or download the `.ipynb` file.

2. Launch Jupyter Notebook:

   ```bash
   jupyter notebook
   ```

3. Open `model-training-testing.ipynb`.

4. **Adjust the dataset path** in the notebook if your dataset file is not in the same directory.

   ### 📌 Modify the file path:

   Locate the line where the dataset is loaded, for example:

   ```python
   df = pd.read_csv("path/to/your/dataset.csv")
   ```

   Update `"path/to/your/dataset.csv"` to reflect the correct path to your dataset file.

5. Run the notebook cells sequentially using **Shift + Enter** or from the menu: **Cell > Run All**.

---

## ⚙️ Model Training Options

* If you want to train the **Random Forest model from scratch**, you will need the dataset files. The required datasets are located in the `dataset` folder. It contains two different CSV files, each with a distinct set of features.
  Simply run the notebook `model-training-testing.ipynb` from start to finish. The full training process takes approximately **3 minutes**.

* If you want to **skip model training from scratch**, you can directly load the pre-trained model by providing the path to the target model file. A pre-trained model is available in the `model_weights` folder.

---

## 🧠 Understanding the Notebook

### 1. **Data Loading**

* Loads a dataset (CSV format) using pandas.
* Displays the dataset’s first few rows and basic info.

### 2. **Data Preprocessing**

* Missing values handled.
* Features may be normalized or encoded.
* Dataset is split into training and test sets.

### 3. **Model Training**

* A machine learning model is trained using training data (`X_train`, `y_train`).

### 4. **Evaluation**

* Predictions are made on the test set (`X_test`).
* Evaluation metrics are calculated: accuracy, precision, recall, F1-score.

### 5. **Visualization**

* Confusion matrix and performance metrics plotted using `matplotlib` or `seaborn`.

---

## 📊 Interpreting Results

* **Accuracy**: Overall prediction correctness.
* **Precision**: Correctness of positive predictions.
* **Recall**: Coverage of actual positives.
* **F1 Score**: Balance between precision and recall.
* **Confusion Matrix**: Visual layout of classification performance.

---

## 📁 Output

* Printed classification metrics.
* Confusion matrix and other plots.
* Optionally, saved model using `joblib` or `pickle`  dump function.

---
