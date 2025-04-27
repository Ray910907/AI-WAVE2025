# 服務器繁忙，請稍後再試

## Fraud Detection Pipeline

Train a LightGBM model to detect fraudulent accounts and fill a template CSV with predictions.

## 🚀 Quickstart

1. **Clone the repo**  
   ```bash
   git clone https://github.com/Ray910907/AI-WAVE2025.git
   cd AI-WAVE2025
    ```

2. **Install dependencies**

    To install dependencies, please run the following command:
    ```bash
    pip install -r requirements.txt
    ```

3. **Run full pipeline**
    ```bash
    python gbm.py
    ```

4. **Fill test CSV**
    ```bash
    python simple-csv-filler.py \
    --template path/to/test_result.csv \
    --output path/to/filled_test_result.csv \
    --threshold 0.3
    ```

### Project Structure
    ```
    .
    ├─ gbm.py   # core class + main()
    ├─ simple-csv-filler.py          # fills test CSV with predictions
    ├─ train_file_reader.py          # train-data loader
    ├─ test_file_reader.py           # test-data loader
    ├─ utils.py                      # helpers: balance, split, metrics
    ├─ comp_data/
    │  ├─ Train/                     # training CSVs
    │  └─ Test/                      # test CSVs
    ├─ models/                       # saved model file
    └─ plots/                        # evaluation plots
    ```

### ⚙️ Configuration
- DATA_PATH (in `gbm.py`): default ./comp_data/Train/

### 📦 Outputs
- Model: `models/fraud_detection_model.txt`
- Plots (in the `plots` folder)

### 📝 Notes
- GPU auto-enabled if CUDA + PyTorch detected.

- Install shap for SHAP plots.
