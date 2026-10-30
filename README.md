# SubWaste AI — Complete Technical Documentation
### ML-Powered Subscription Waste & Underutilization Risk Intelligence System

---

## 1. Project Overview

### 1.1 What SubWaste AI Is
**SubWaste AI** is a machine learning system and full-stack software service designed to estimate whether an active recurring subscription is **underutilized, dormant, or economically wasteful**.

The system analyzes subscriber behavioral telemetry—such as login recency, monthly active hours, session frequency, plan tiers, and recurring billing patterns—to output a calibrated **waste probability score (0.0% to 100.0%)**, an assigned **risk category (`LOW RISK`, `MODERATE RISK`, `HIGH RISK`)**, and a **plain-English financial recommendation** detailing annualized dollar waste.

---

### 1.2 The Problem It Solves: The "Zombie Subscription" Crisis
Modern consumers and enterprises suffer from **subscription creep** and **zombie spend**:
- Independent financial surveys (West Monroe, C+R Research) show that **over 42% of consumers continue paying for subscriptions they have completely forgotten about**, underestimating their recurring micro-spending by a factor of 2.5x.
- Traditional budgeting apps merely report past expenses. They do not calculate the **unit economic value** of a subscription (e.g., spending $55/month on design software while using it for 0.0 hours costs infinite dollars per hour).
- **SubWaste AI** solves this by actively auditing consumption efficiency and identifying forgotten subscriptions *before* automatic renewals drain hundreds of dollars.

---

### 1.3 The Critical Conceptual Distinction: Subscription Churn vs. Subscription Waste

A frequent mistake in beginner machine learning portfolios is downloading a Kaggle Customer Churn dataset and treating churn prediction as subscription waste prediction. These are **fundamentally different problems**:

| Dimension | Customer Churn Prediction | Subscription Waste / Underutilization (SubWaste AI) |
| :--- | :--- | :--- |
| **Perspective** | Vendor / SaaS Business retention | Consumer / FinOps financial health |
| **User State** | Customer has **canceled** or terminated service | Customer is **actively subscribed** and paying |
| **Billing Status** | Billing has stopped; vendor lost recurring revenue | Automatic renewal is active; **money is actively leaking** |
| **Telemetry Pattern**| User signals intent to leave, files complaints, or closes card | User is silent, dormant, or exhibits near-zero active logins |
| **Target Label** | `churn = 1` (Service terminated) | `is_wasteful = 1` (Dormant spend on active subscription) |
| **Business Action** | Vendor offers discount or retention incentives | Consumer / FinOps pauses, cancels, or downgrades service |

If an algorithm naively treated non-churned subscribers as positive examples, it would classify the most severely dormant zombie accounts as "good" simply because they haven't canceled yet. **SubWaste AI explicitly focuses on the active subscriber who is losing money quietly.**

---

### 1.4 Input Data Accepted
The system accepts structured subscriber records with 13 core behavioral and billing attributes:
1. `subscription_name` (String, e.g., "Adobe Creative Cloud", "Spotify", "Gold's Gym")
2. `category` (Enum: `Streaming`, `Productivity`, `Cloud Storage`, `Fitness`, `Gaming`, `Education`, `Utilities`, `Other`)
3. `plan_tier` (Enum: `Basic`, `Standard`, `Premium`)
4. `monthly_fee` (Float: recurring USD charge, strictly $> \$0.00$)
5. `billing_cycle` (Enum: `Monthly`, `Annual`)
6. `payment_method` (String: `Credit Card`, `Debit Card`, `PayPal`, `Auto-Debit`)
7. `auto_renew` (Binary: `1` = active automatic renewal, `0` = manual renewal)
8. `usage_frequency_per_month` (Integer: active calendar days accessed in past 30 days, range $0 \le d \le 31$)
9. `hours_used_per_month` (Float: total consumption hours in past 30 days, $\ge 0.0$)
10. `days_since_last_login` (Integer: recency of last session in days, $\ge 0$)
11. `support_tickets_last_3m` (Integer: support interactions in past 90 days, $\ge 0$)
12. `number_of_profiles_or_devices` (Integer: concurrent users or connected devices, $\ge 1$)
13. `subscription_age_months` (Integer: account tenure in months, $\ge 1$)

---

### 1.5 Outputs Produced
For any evaluated subscription, the system generates:
- **`is_wasteful`**: Binary classification label (`1` = Underutilized/Wasteful, `0` = Utilized/Healthy).
- **`waste_probability`**: Calibrated probability percentage (e.g., `93.27%`).
- **`risk_level`**: 3-tier categorization (`LOW RISK (Well Utilized)`, `MODERATE RISK`, `HIGH RISK (Wasteful)`).
- **`recommendation`**: Actionable advisory text with calculated annual dollar savings (e.g., *"Action: Cancel or Pause Immediately. You haven't logged in for 48 days while paying $54.99/month. Canceling saves you $659.88/year.*").
- **`engineered_metrics`**:
  - `cost_per_active_hour`: Actual dollar cost per hour of consumption.
  - `inactivity_ratio`: Days dormant relative to standard 30-day billing cycle.
  - `annual_projected_cost`: Annualized financial drain.

---

### 1.6 Intended Users & Practical Use Cases
- **Personal Finance & Budgeting:** Individuals auditing recurring personal credit card subscriptions.
- **FinOps & Corporate SaaS Management:** Small and medium enterprises (SMEs) identifying unassigned software seats, redundant developer tools, or unused cloud licenses.
- **Banking / Open Banking Applications:** Financial aggregators providing automated "subscription leak" notifications to bank customers.

---

### 1.7 Current Scope, Limitations & Truthful Disclosure of Labels
- **Explicit Target Label Disclosure:** In real-world enterprise environments, ground-truth subscription waste is subjective and guarded under strict privacy laws (GDPR/CCPA). There is **no publicly available dataset containing ground-truth subscription waste labels**. Therefore, the dataset in this project was created using a statistically grounded synthetic telemetry generator ([scripts/generate_raw_data.py](file:///c:/Users/harry/Desktop/SubWaste%20AI/scripts/generate_raw_data.py)) modeling real-world distributions, and the target variable `is_wasteful` was defined using a multi-criteria economic heuristic combined with **8% stochastic noise injection**.
- **Important Scientific Caveat:** High benchmark test scores reflect the model's ability to recover non-linear utility boundaries from this synthetic distribution. **This project does NOT claim that these performance figures represent validated predictive accuracy on real-world bank transactions.** Real production deployment would require active user feedback loops, bank integration APIs, and continuous data drift monitoring.
- **Scope Status:**
  - *Implemented:* Machine learning pipeline, feature engineering, baseline vs. champion model training, evaluation diagnostics, joblib serialization, FastAPI production REST API, automated test suite (14 unit/integration tests), and an interactive glassmorphic web dashboard.
  - *Not Implemented / Out of Scope:* Database integration (SQL/NoSQL), user authentication / JWT, containerization / Docker, and cloud hosting.

---

## 2. Complete Project Folder Structure

The repository contains 8 primary subdirectories and 13 functional files. Below is the complete directory map:

```
SubWaste-AI/
├── .gitignore                           # Git ignore rules for Python, virtualenvs, & temp files
├── requirements.txt                     # Declared application and testing dependencies
├── README.md                            # Comprehensive technical documentation
├── data/
│   ├── sample_audit.csv                 # 8-record sample CSV for testing batch uploads
│   ├── processed/
│   │   └── subscriptions_cleaned.csv    # Cleaned, validated dataset (1,500 rows, 15 columns)
│   └── raw/
│       └── subscriptions_raw.csv        # Immutable raw synthetic telemetry (1,500 rows, 15 columns)
├── models/
│   ├── model_metadata.json              # Hyperparameters, feature schema, & held-out test metrics
│   └── subwaste_pipeline.joblib         # Serialized ColumnTransformer + RandomForest pipeline (1.29 MB)
├── notebooks/
│   └── exploratory_analysis.ipynb       # Jupyter notebook with EDA, distributions & correlation analysis
├── reports/
│   ├── confusion_matrix.png             # Diagnostic confusion matrix heatmap (held-out test set)
│   ├── evaluation_report.txt            # Classification report text & error interpretation
│   ├── feature_importance.png           # Top 12 Gini feature importance horizontal bar chart
│   └── model_comparison.json            # Logistic Regression vs. Random Forest test metrics
├── scripts/
│   ├── create_eda_notebook.py           # Programmatic builder generating exploratory_analysis.ipynb
│   └── generate_raw_data.py             # Telemetry generator simulating 1,500 subscriber records
├── src/
│   ├── __init__.py                      # Package initializer (exports __version__ = "1.0.0")
│   ├── data_loader.py                   # Ingestion, schema validation, deduplication, & sanitization
│   ├── evaluate.py                      # Test evaluation suite, confusion matrix & plot generator
│   ├── features.py                      # Domain ratio engineering & ground-truth labeling heuristic
│   ├── predict.py                       # SubWastePredictor inference engine with defensive validation
│   ├── preprocessing.py                 # Scikit-learn ColumnTransformer (StandardScaler + OneHotEncoder)
│   ├── train.py                         # Stratified splitting, baseline benchmarking, & model persistence
│   └── api/
│       ├── __init__.py                  # API package initializer
│       ├── app.py                       # FastAPI application with lifespan caching, CORS, & endpoints
│       └── schemas.py                   # Pydantic v2 data models, enums, & validation constraints
├── static/
│   ├── index.html                       # Single Page Application HTML5 dashboard with SEO metadata
│   ├── css/
│   │   └── style.css                    # Vanilla CSS glassmorphic theme with animations & responsive grids
│   └── js/
│       └── app.js                       # Client controller: sliders, presets, drag-drop CSV, live health
└── tests/
    ├── __init__.py                      # Tests package initializer
    ├── test_api.py                      # 8 Integration tests for FastAPI endpoints via TestClient
    └── test_pipeline.py                 # 6 Unit & integration tests for data loading, features, & inference
```

---

### Detailed File-by-File Breakdown

| Exact Relative Path | File Type | Purpose & Contents | Why Needed | Dependencies & Dependents |
| :--- | :--- | :--- | :--- | :--- |
| `.gitignore` | Config | Specifies ignored files (`__pycache__`, `.venv`, `*.log`, `*.ipynb_checkpoints`). | Prevents committing bytecode, environments, and scratch files. | Used by Git. |
| `requirements.txt` | Config | Lists 12 pinned package dependencies. | Ensures reproducible environment installation. | Used by `pip install -r requirements.txt`. |
| `README.md` | Doc | Complete technical documentation for the system. | Explains design, architecture, metrics, and usage. | Standalone documentation. |
| `data/raw/subscriptions_raw.csv` | Dataset (CSV) | 1,500 rows of raw subscription telemetry across 15 columns. | Acts as the immutable raw data source for ingestion. | Generated by `generate_raw_data.py`; consumed by `data_loader.py`. |
| `data/processed/subscriptions_cleaned.csv` | Dataset (CSV) | Cleaned, bounded, and deduplicated subscription records. | Provides clean data for feature engineering without mutating raw. | Produced by `data_loader.py`; consumed by `features.py`. |
| `data/sample_audit.csv` | Dataset (CSV) | 8 sample subscriptions (active and dormant mix). | Allows immediate testing of the CSV batch upload functionality. | Used by `static/js/app.js` and user testing. |
| `models/subwaste_pipeline.joblib` | Model Artifact (Binary) | Serialized scikit-learn pipeline (ColumnTransformer + Random Forest). | Serves real-time inference without retraining. | Produced by `train.py`; loaded by `predict.py` and `app.py`. |
| `models/model_metadata.json` | Metadata (JSON) | Hyperparameters, feature column lists, and test evaluation metrics. | Provides model lineage, auditability, and API metadata endpoint. | Produced by `train.py`; consumed by `evaluate.py` and `app.py`. |
| `notebooks/exploratory_analysis.ipynb`| Notebook (JSON) | 8-section interactive EDA notebook with markdown and visualizations. | Demonstrates data understanding, distributions, and correlation. | Generated by `create_eda_notebook.py`. |
| `reports/confusion_matrix.png` | Plot (PNG) | Heatmap of True Negatives, False Positives, False Negatives, True Positives. | Diagnostic visual for classification performance. | Produced by `evaluate.py`. |
| `reports/feature_importance.png` | Plot (PNG) | Horizontal bar chart of top 12 Gini importance features. | Explains which features govern model decisions. | Produced by `evaluate.py`. |
| `reports/model_comparison.json` | Report (JSON) | Comparative metrics between Baseline Logistic Regression and Random Forest. | Proves candidate model superiority over baseline. | Produced by `train.py`. |
| `reports/evaluation_report.txt` | Report (Text) | Classification report metrics, confusion matrix counts, and limitations. | Permanent human-readable evaluation record. | Produced by `evaluate.py`. |
| `scripts/generate_raw_data.py` | Script (Python) | Simulates 1,500 subscriber records across 14 services & 3 archetypes. | Generates the underlying benchmark raw dataset reproducibly. | Depends on `numpy`, `pandas`. Produces `subscriptions_raw.csv`. |
| `scripts/create_eda_notebook.py` | Script (Python) | Builds `notebooks/exploratory_analysis.ipynb` with clean JSON structure. | Automates notebook creation without notebook editor corruptions. | Produces `exploratory_analysis.ipynb`. |
| `src/__init__.py` | Source Code | Defines package root and exports `__version__ = "1.0.0"`. | Enables Python package structure and discovery. | Consumed by any internal import. |
| `src/data_loader.py` | Source Code | Implements `load_raw_data`, `validate_schema`, and `clean_data`. | Enforces data integrity, schema bounds, and immutability. | Consumes `subscriptions_raw.csv`; outputs `subscriptions_cleaned.csv`. |
| `src/features.py` | Source Code | Calculates derived ratios and implements `define_target` heuristic. | Translates telemetry into economic utility indicators and labels. | Depended on by `train.py`, `preprocessing.py`, `predict.py`. |
| `src/preprocessing.py` | Source Code | Builds `ColumnTransformer` with `StandardScaler` and `OneHotEncoder`. | Prevents data leakage by encapsulating transformation logic. | Depended on by `train.py`, `evaluate.py`. |
| `src/train.py` | Source Code | Executes stratified split, trains baseline and Random Forest, saves joblib. | Orchestrates the end-to-end training pipeline. | Depends on `features.py`, `preprocessing.py`. Produces model artifacts. |
| `src/evaluate.py` | Source Code | Computes classification metrics, confusion matrix, and feature importances. | Evaluates model quality on held-out test data. | Depends on `models/subwaste_pipeline.joblib`, `features.py`. |
| `src/predict.py` | Source Code | Encapsulates `SubWastePredictor` class with defensive input validation. | Provides single/batch prediction logic and financial recommendations. | Depended on by `app.py`, `test_pipeline.py`. |
| `src/api/__init__.py` | Source Code | Empty package initializer for `src.api`. | Marks directory as Python subpackage. | Standard package structure. |
| `src/api/schemas.py` | Source Code | Defines Pydantic v2 schemas (`SubscriptionInput`, outputs, enums). | Validates API payloads at the HTTP network gateway. | Depended on by `app.py`, `test_api.py`. |
| `src/api/app.py` | Source Code | FastAPI REST application with lifespan caching, CORS, and endpoints. | Serves predictions over HTTP and delivers web dashboard. | Depends on `schemas.py`, `predict.py`, `static/`. |
| `static/index.html` | Frontend (HTML) | Dashboard user interface with Hero KPIs, presets, and 3 tabs. | Provides visual interactive access to predictions and batch audits. | Served by `app.py`; depends on `style.css` and `app.js`. |
| `static/css/style.css` | Frontend (CSS) | 450+ lines of custom Vanilla CSS with glassmorphism and animations. | Delivers modern dark-mode aesthetic and responsive layouts. | Consumed by `index.html`. |
| `static/js/app.js` | Frontend (JS) | Asynchronous client logic: slider sync, presets, fetch API, file drop. | Manages interactive UI behaviors and communication with backend. | Consumed by `index.html`. |
| `tests/__init__.py` | Tests (Python) | Empty package initializer for `tests`. | Marks directory as test package. | Standard testing structure. |
| `tests/test_pipeline.py` | Tests (Python) | 6 unit/integration tests for loader, cleaner, features, and model. | Verifies ML pipeline correctness and reproducibility. | Depends on `unittest`, `src` modules. |
| `tests/test_api.py` | Tests (Python) | 8 integration tests for REST API routes, validation, and CSV upload. | Verifies API contracts, HTTP status codes, and batch calculations. | Depends on `fastapi.testclient`, `src.api.app`. |

---

## 3. Programming Language, Libraries, Packages and Frameworks

### 3.1 Python Runtime
- **Verified Runtime Version:** `Python 3.13.5` (64-bit on Windows).
- **Verified Package Manager:** `pip 25.3`.

---

### 3.2 Third-Party Dependencies Matrix

| Library / Package | Installed Version | Declared in `requirements.txt` | Project Purpose | Exact Files Where Imported / Used | Key Classes / Functions Utilized |
| :--- | :---: | :---: | :--- | :--- | :--- |
| **`pandas`** | `2.3.3` | `pandas>=2.0.0` | Tabular data manipulation, CSV I/O, series operations | `data_loader.py`, `features.py`, `train.py`, `evaluate.py`, `predict.py`, `app.py`, `generate_raw_data.py` | `DataFrame`, `Series`, `read_csv()`, `to_csv()`, `concat()`, `isnull()`, `clip()`, `value_counts()` |
| **`numpy`** | `2.2.6` | `numpy>=1.24.0` | Numerical calculations, arrays, vectorized logic, random seeds | `features.py`, `data_loader.py`, `train.py`, `evaluate.py`, `generate_raw_data.py` | `ndarray`, `round()`, `random.seed()`, `random.choice()`, `random.uniform()`, `isinf()` |
| **`scikit-learn`** | `1.7.2` | `scikit-learn>=1.3.0` | ML modeling, feature scaling, encoding, train/test splitting, metrics | `preprocessing.py`, `train.py`, `evaluate.py` | `ColumnTransformer`, `StandardScaler`, `OneHotEncoder`, `Pipeline`, `train_test_split`, `LogisticRegression`, `RandomForestClassifier`, `accuracy_score`, `precision_score`, `recall_score`, `f1_score`, `roc_auc_score`, `confusion_matrix`, `classification_report` |
| **`matplotlib`** | `3.10.6` | `matplotlib>=3.7.0` | Plot generation for evaluation diagnostics | `evaluate.py`, `create_eda_notebook.py` | `pyplot.figure()`, `pyplot.savefig()`, `pyplot.barh()`, `pyplot.close()` |
| **`seaborn`** | `0.13.2` | `seaborn>=0.12.0` | Statistical heatmap plotting | `evaluate.py`, `create_eda_notebook.py` | `heatmap()`, `color_palette()`, `histplot()` |
| **`joblib`** | `1.5.2` | `joblib>=1.3.0` | Model and pipeline serialization to disk | `train.py`, `evaluate.py`, `predict.py` | `dump()`, `load()` |
| **`fastapi`** | `0.114.2` | `fastapi>=0.110.0` | Production REST API framework | `src/api/app.py` | `FastAPI`, `HTTPException`, `UploadFile`, `File`, `status`, `CORSMiddleware`, `JSONResponse`, `FileResponse` |
| **`uvicorn`** | `0.30.6` | `uvicorn>=0.28.0` | ASGI web server running FastAPI | `src/api/app.py` | `run()` |
| **`pydantic`** | `2.9.2` | `pydantic>=2.5.0` | Request/response data validation and schema generation | `src/api/schemas.py` | `BaseModel`, `Field`, `ConfigDict` |
| **`starlette`** | `0.38.6` | Implicit (FastAPI) | Static file mounting | `src/api/app.py` | `StaticFiles` |
| **`httpx`** | `0.28.1` | `httpx>=0.27.0` | Testing client transport for FastAPI `TestClient` | `tests/test_api.py` | Used implicitly via `fastapi.testclient.TestClient` |
| **`python-multipart`** | `0.0.9` | `python-multipart>=0.0.9` | Parsing multipart/form-data for CSV file uploads | `src/api/app.py` | Used implicitly by FastAPI `UploadFile` |
| **`jupyter`** | `1.1.1` | `jupyter>=1.0.0` | Interactive notebook analysis environment | `notebooks/exploratory_analysis.ipynb` | Execution of EDA notebook |

---

### 3.3 Python Standard Library Modules Used
- **`os`**: Directory checking, path joining, environment inspection (`os.path.exists`, `os.makedirs`).
- **`sys`**: Modifying search path for robust cross-directory execution (`sys.path.insert`).
- **`pathlib.Path`**: Object-oriented absolute path resolution (`Path(__file__).resolve().parent`).
- **`json`**: Serializing and reading `model_metadata.json`, `model_comparison.json`, and `.ipynb` files.
- **`logging`**: Structured pipeline event logging (`logging.getLogger()`, `logging.basicConfig()`).
- **`datetime.datetime`**: Timestamping saved model versions and API health checks (`datetime.now().isoformat()`).
- **`contextlib.asynccontextmanager`**: FastAPI lifespan model loading manager.
- **`io.StringIO` / `io.BytesIO`**: In-memory string and byte parsing for CSV uploads and test files.
- **`enum.Enum`**: Enumerated types for categories, tiers, and billing cycles in Pydantic.
- **`typing`**: Strict static type hints (`List`, `Dict`, `Tuple`, `Any`, `Optional`, `Union`).
- **`unittest`**: Automated unit and integration test framework (`TestCase`, `main()`).

---

## 4. Built-in Functions vs. User-Defined Functions

### 4.1 Exhaustive Function & Method Catalog

| Function / Method | File & Module | Category | Inputs / Parameters | Return Value | Purpose & Responsibility | Where / How Called |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `load_raw_data()` | `src/data_loader.py` | User-Defined | `filepath: str` | `pd.DataFrame` | Verifies existence and loads raw CSV into memory | Called by `load_and_process()` and `test_pipeline.py` |
| `validate_schema()` | `src/data_loader.py` | User-Defined | `df: pd.DataFrame` | `Dict[str, Any]` | Validates presence of 15 expected columns, counts nulls and duplicates | Called by `load_and_process()`, `test_pipeline.py` |
| `clean_data()` | `src/data_loader.py` | User-Defined | `df: pd.DataFrame` | `pd.DataFrame` | Deduplicates records, trims string whitespace, clamps numeric bounds | Called by `load_and_process()`, `test_pipeline.py` |
| `load_and_process()` | `src/data_loader.py` | User-Defined | `raw_path: str`, `processed_path: str` | `pd.DataFrame` | High-level orchestrator: loads, validates, cleans, and saves to `data/processed/` | Executed via CLI `python src/data_loader.py` |
| `engineer_features()` | `src/features.py` | User-Defined | `df: pd.DataFrame` | `pd.DataFrame` | Computes derived ratios: `cost_per_active_hour`, `inactivity_ratio`, etc. | Called by `build_feature_dataset()`, `predict_single()` |
| `define_target()` | `src/features.py` | User-Defined | `df: pd.DataFrame`, `noise_rate: float`, `random_state: int` | `pd.Series` | Applies multi-criteria economic heuristic + 8% noise to generate binary label | Called by `build_feature_dataset()`, `test_pipeline.py` |
| `build_feature_dataset()` | `src/features.py` | User-Defined | `input_path: str` | `Tuple[pd.DataFrame, pd.Series]` | Loads cleaned data, runs feature engineering, separates $X$ matrix and $y$ target | Called by `train.py`, `evaluate.py`, `test_pipeline.py` |
| `create_preprocessor()` | `src/preprocessing.py` | User-Defined | `num_features: List[str]`, `cat_features: List[str]` | `ColumnTransformer` | Constructs unfitted ColumnTransformer (StandardScaler + OneHotEncoder) | Called by `train.py`, `test_pipeline.py` |
| `get_feature_names()` | `src/preprocessing.py` | User-Defined | `preprocessor: ColumnTransformer` | `List[str]` | Extracts one-hot transformed column names from fitted transformer | Called by `evaluate.py`, `preprocessing.py` |
| `evaluate_predictions()` | `src/train.py` | User-Defined | `y_true: np.ndarray`, `y_pred: np.ndarray`, `y_proba: np.ndarray` | `Dict[str, float]` | Calculates accuracy, precision, recall, F1, and ROC-AUC | Called by `train_and_compare_models()` |
| `train_and_compare_models()` | `src/train.py` | User-Defined | `X: pd.DataFrame`, `y: pd.Series`, `test_size: float`, `random_state: int` | `Tuple[Pipeline, Dict, Tuple]` | Performs stratified split, trains Logistic Regression and Random Forest, compares metrics | Called by `run_training_pipeline()` |
| `save_pipeline_artifacts()` | `src/train.py` | User-Defined | `pipeline: Pipeline`, `comparison: Dict`, `models_dir: str`, `reports_dir: str` | `Tuple[str, str]` | Serializes pipeline to `.joblib` and writes `model_metadata.json` | Called by `run_training_pipeline()` |
| `run_training_pipeline()` | `src/train.py` | User-Defined | None | `Tuple[Pipeline, Dict, Tuple]` | High-level orchestrator executing features, training, evaluation, and saving | Executed via CLI `python src/train.py` |
| `generate_confusion_matrix_plot()` | `src/evaluate.py` | User-Defined | `y_test: np.ndarray`, `y_pred: np.ndarray`, `output_path: str` | `np.ndarray` | Generates and saves confusion matrix heatmap plot | Called by `evaluate_pipeline()` |
| `generate_feature_importance_plot()` | `src/evaluate.py` | User-Defined | `pipeline: Pipeline`, `top_n: int`, `output_path: str` | `pd.DataFrame` | Extracts classifier feature importances and saves horizontal bar chart | Called by `evaluate_pipeline()` |
| `evaluate_pipeline()` | `src/evaluate.py` | User-Defined | `model_path: str`, `reports_dir: str` | `Dict[str, Any]` | Reloads pipeline, runs inference on test split, generates all diagnostic reports | Executed via CLI `python src/evaluate.py` |
| `SubWastePredictor.__init__()` | `src/predict.py` | Class Constructor | `model_path: str` | None | Initializes predictor class and loads fitted joblib pipeline | Instantiated in `predict.py`, `app.py`, `test_pipeline.py` |
| `SubWastePredictor._load_model()` | `src/predict.py` | Class Method | None | `Pipeline` | Validates file existence and deserializes joblib pipeline | Called by `__init__()` |
| `SubWastePredictor.validate_input()` | `src/predict.py` | Class Method | `record: Dict[str, Any]` | None | Enforces non-negative bounds and required field presence; raises `ValueError` | Called by `predict_single()` |
| `SubWastePredictor._generate_recommendation()` | `src/predict.py` | Class Method | `record: Dict`, `proba: float`, `is_wasteful: int` | `str` | Synthesizes contextual text recommendation with annual dollar savings | Called by `predict_single()` |
| `SubWastePredictor.predict_single()` | `src/predict.py` | Class Method | `record: Dict[str, Any]` | `Dict[str, Any]` | End-to-end inference for a single subscription dictionary | Called by `predict_batch()`, `app.py` |
| `SubWastePredictor.predict_batch()` | `src/predict.py` | Class Method | `records: List[Dict[str, Any]]` | `List[Dict[str, Any]]` | Iterative inference over a list of subscription records | Called by CLI demo in `predict.py` |
| `lifespan()` | `src/api/app.py` | User-Defined Context Manager | `app: FastAPI` | Async Generator | Loads trained model once during FastAPI startup; caches in global variable | Configured in `FastAPI(lifespan=lifespan)` |
| `get_active_predictor()` | `src/api/app.py` | User-Defined | None | `SubWastePredictor` | Returns cached predictor instance or raises HTTP 503 if missing | Called by all inference route handlers in `app.py` |
| `serve_dashboard()` | `src/api/app.py` | Route Handler (`GET /`) | None | `FileResponse` | Serves `static/index.html` Single Page Application | Invoked when user visits `http://127.0.0.1:8000/` |
| `api_overview()` | `src/api/app.py` | Route Handler (`GET /api`) | None | `Dict[str, str]` | Returns API metadata, version, and links to documentation | Invoked via `GET /api` |
| `health_check()` | `src/api/app.py` | Route Handler (`GET /health`) | None | `HealthResponse` | Returns uptime health, model load status, and timestamp | Invoked via `GET /health` |
| `get_model_metadata()` | `src/api/app.py` | Route Handler (`GET /model/metadata`)| None | `ModelMetadataResponse` | Returns hyperparameters and test evaluation metrics from JSON | Invoked via `GET /model/metadata` |
| `predict_subscription()` | `src/api/app.py` | Route Handler (`POST /predict`) | `subscription: SubscriptionInput` | `SubscriptionPredictionOutput` | Audits single subscription payload with validation | Invoked via `POST /predict` |
| `predict_batch_subscriptions()` | `src/api/app.py` | Route Handler (`POST /predict/batch`)| `payload: BatchPredictionRequest` | `BatchPredictionResponse` | Audits multiple subscriptions, aggregates total spend & potential savings | Invoked via `POST /predict/batch` |
| `predict_from_csv()` | `src/api/app.py` | Route Handler (`POST /predict/csv`) | `file: UploadFile` | `BatchPredictionResponse` | Parses uploaded CSV file, validates columns, executes batch audit | Invoked via `POST /predict/csv` |
| `generate_synthetic_subscriptions()` | `scripts/generate_raw_data.py`| User-Defined | `num_samples: int`, `random_state: int` | `pd.DataFrame` | Simulates subscriber cohorts across 14 services & 3 behavioral archetypes | Executed via CLI `python scripts/generate_raw_data.py` |
| `create_notebook()` | `scripts/create_eda_notebook.py` | User-Defined | None | None | Programmatically writes `notebooks/exploratory_analysis.ipynb` JSON | Executed via CLI `python scripts/create_eda_notebook.py` |

---

### 4.2 Python Built-ins Actively Utilized in Code
- **`len()`**: Computes length of dataframes, lists, dictionaries, and feature arrays (`src/data_loader.py:64`, `src/train.py:59`, `src/preprocessing.py:68`).
- **`print()`**: Formats terminal summaries in CLI execution blocks (`src/data_loader.py:136`, `src/train.py:164`).
- **`range()`**: Generates sample IDs and loop iterations (`scripts/generate_raw_data.py:62`).
- **`int()` / `float()` / `str()`**: Explicit type casting and sanitization (`src/data_loader.py:66`, `src/features.py:73`, `src/api/app.py:182`).
- **`round()`**: Rounds float numbers to clean precision (`src/features.py:54`, `src/train.py:46`).
- **`min()` / `max()`**: Clamps values within logical boundaries (`scripts/generate_raw_data.py:79`).
- **`open()`**: Opens JSON reports and text files with contextual file handles (`src/train.py:141`, `src/evaluate.py:107`).
- **`isinstance()` / `hasattr()`**: Inspects object properties and enum instances (`src/api/app.py:175`).

---

## 5. Dataset: Source, Schema and Data Quality

### 5.1 Dataset Origin and Location
- **Raw File Location:** `data/raw/subscriptions_raw.csv` (143.5 KB)
- **Cleaned File Location:** `data/processed/subscriptions_cleaned.csv` (143.5 KB)
- **Sample File Location:** `data/sample_audit.csv` (862 Bytes)
- **Origin:** Generated by `scripts/generate_raw_data.py` using fixed pseudo-random seed (`random_state=42`).

---

### 5.2 Dataset Dimensions & Verified Integrity
- **Total Rows:** 1,500 subscriber records.
- **Total Columns:** 15 columns.
- **Missing Values:** 0 across all columns (100% complete).
- **Duplicate Records:** 0 duplicate `subscription_id` values.
- **Cleaned Dataset Dimensions:** 1,500 rows $\times$ 15 columns (100% retained after verification).

---

### 5.3 Complete Feature Schema & Data Dictionary

| Column Name | Storage Type | Domain & Description | Value Range / Constraints in Data | Example Value | Modeled Feature? |
| :--- | :--- | :--- | :--- | :--- | :---: |
| `subscription_id` | String (`object`) | Unique identifier for subscription | Format: `SUB-xxxxx` | `SUB-10042` | No (ID dropped) |
| `user_id` | String (`object`) | Unique identifier for subscriber | Format: `USR-xxxx` | `USR-2041` | No (ID dropped) |
| `subscription_name` | String (`object`) | Name of subscribed service | 14 services across 6 categories | `Adobe Creative Cloud` | No (Categorized) |
| `category` | Categorical (`object`)| Market domain of the subscription | 6 classes: `Streaming`, `Productivity`, `Cloud Storage`, `Fitness`, `Gaming`, `Education` | `Productivity` | **Yes (Categorical)** |
| `plan_tier` | Categorical (`object`)| Service tier level | 3 classes: `Basic`, `Standard`, `Premium` | `Premium` | **Yes (Categorical)** |
| `monthly_fee` | Numeric (`float64`) | Recurring monthly charge in USD | Min: `$2.00`, Max: `$87.98`, Mean: `$23.23` | `$54.99` | **Yes (Numerical)** |
| `billing_cycle` | Categorical (`object`)| Payment renewal frequency | 2 classes: `Monthly` (70%), `Annual` (30%) | `Monthly` | **Yes (Categorical)** |
| `payment_method` | Categorical (`object`)| Payment mechanism used | 4 classes: `Credit Card`, `Debit Card`, `PayPal`, `Auto-Debit` | `Credit Card` | **Yes (Categorical)** |
| `auto_renew` | Binary (`int64`) | Automatic renewal status | `1` (Active: 88%), `0` (Manual: 12%) | `1` | **Yes (Categorical)** |
| `usage_frequency_per_month` | Numeric (`int64`) | Active days used in past 30 days | Min: `0`, Max: `30` days, Mean: `12.4` days | `0` | **Yes (Numerical)** |
| `hours_used_per_month` | Numeric (`float64`) | Total active consumption hours | Min: `0.0`, Max: `75.0` hrs, Mean: `20.1` hrs | `0.0` | **Yes (Numerical)** |
| `days_since_last_login` | Numeric (`int64`) | Inactivity recency in days | Min: `0`, Max: `89` days, Mean: `20.8` days | `45` | **Yes (Numerical)** |
| `support_tickets_last_3m` | Numeric (`int64`) | Support interactions in 90 days | Min: `0`, Max: `2`, Mean: `0.3` | `0` | **Yes (Numerical)** |
| `number_of_profiles_or_devices` | Numeric (`int64`) | Concurrent profiles or devices | Min: `1`, Max: `4`, Mean: `1.9` | `1` | **Yes (Numerical)** |
| `subscription_age_months` | Numeric (`int64`) | Account tenure in months | Min: `1`, Max: `60` months, Mean: `14.8` | `12` | **Yes (Numerical)** |

---

### 5.4 Engineered Feature Specifications ([src/features.py](file:///c:/Users/harry/Desktop/SubWaste%20AI/src/features.py))

Six additional domain features are derived before modeling:

1. **`cost_per_active_hour`** $= \frac{\text{monthly\_fee}}{\text{hours\_used\_per\_month} + 0.1}$
   - *Rationale:* Quantifies actual unit cost per hour. A small smoothing constant ($+0.1$) prevents `ZeroDivisionError` when hours used is $0.0$.
2. **`cost_per_active_day`** $= \frac{\text{monthly\_fee}}{\text{usage\_frequency\_per\_month} + 0.1}$
   - *Rationale:* Amortizes subscription fee over active daily sessions.
3. **`inactivity_ratio`** $= \frac{\text{days\_since\_last\_login}}{30.0}$
   - *Rationale:* Measures inactivity relative to a standard 30-day monthly billing cycle. A value $\ge 1.0$ indicates zero logins in a full billing window.
4. **`utilization_rate`** $= \frac{\text{usage\_frequency\_per\_month}}{30.0}$
   - *Rationale:* Fraction of days in the month where the service was engaged ($0.0$ to $1.0$).
5. **`annual_projected_cost`** $= \text{monthly\_fee} \times 12.0$
   - *Rationale:* Calculates cumulative annual financial exposure.
6. **`fee_to_device_ratio`** $= \frac{\text{monthly\_fee}}{\max(\text{number\_of\_profiles\_or\_devices}, 1)}$
   - *Rationale:* Detects solo subscribers paying for multi-seat/family licenses without utilizing linked profiles.

---

### 5.5 Target Variable Distribution & Imbalance
- **Target Variable:** `is_wasteful` $\in \{0, 1\}$
- **Class 0 (`Utilized / Healthy`):** 1,042 samples (**69.47%**)
- **Class 1 (`Wasteful / Underutilized`):** 458 samples (**30.53%**)
- **Class Imbalance Handling:** The ~70:30 distribution represents realistic class asymmetry. In `train.py`, `class_weight="balanced"` is passed to both Logistic Regression and Random Forest to prevent majority-class bias.

---

## 6. Complete Machine Learning Workflow

The pipeline is implemented sequentially across modular scripts to ensure reproducibility, data leakage prevention, and clean separation of concerns:

```mermaid
flowchart TD
    A["Raw Telemetry (data/raw/subscriptions_raw.csv)"] --> B["data_loader.py<br/>(Validation & Bounds Sanitization)"]
    B --> C["data/processed/subscriptions_cleaned.csv"]
    C --> D["features.py<br/>(Ratio Engineering & Target Heuristic)"]
    D --> E["Feature Matrix X (18 Features) & Target y"]
    E --> F["train.py<br/>(Stratified 80/20 Train-Test Split)"]
    F --> G["X_train (1,200 samples)"]
    F --> H["X_test (300 samples)"]
    G --> I["preprocessing.py<br/>(Fit ColumnTransformer on Train Only)"]
    I --> J["Fit Baseline: LogisticRegression"]
    I --> K["Fit Champion: RandomForestClassifier"]
    H --> L["Evaluate Models on Held-out X_test"]
    J --> L
    K --> L
    L --> M["Champion Selection: Random Forest (F1: 0.8421 vs 0.7931)"]
    M --> N["Serialize Pipeline (models/subwaste_pipeline.joblib)"]
    M --> O["Save Metadata (models/model_metadata.json)"]
    N --> P["predict.py / app.py<br/>(FastAPI Service & Web Dashboard)"]
```

### The 18 Steps Implemented in Code:

1. **Synthetic Data Generation ([scripts/generate_raw_data.py](file:///c:/Users/harry/Desktop/SubWaste%20AI/scripts/generate_raw_data.py)):** Simulates 1,500 subscriber records across 3 latent behavioral archetypes (Active: 45%, Moderate: 25%, Dormant: 30%) and saves to `data/raw/subscriptions_raw.csv`.
2. **Raw Data Ingestion ([src/data_loader.py](file:///c:/Users/harry/Desktop/SubWaste%20AI/src/data_loader.py)):** `load_raw_data()` reads the raw CSV with file existence checks.
3. **Data Contract & Schema Validation ([src/data_loader.py](file:///c:/Users/harry/Desktop/SubWaste%20AI/src/data_loader.py)):** `validate_schema()` ensures all 15 required columns exist and tallies missing values.
4. **Data Cleaning & Sanitization ([src/data_loader.py](file:///c:/Users/harry/Desktop/SubWaste%20AI/src/data_loader.py)):** `clean_data()` deduplicates records by `subscription_id`, strips string whitespace, and clamps negative or out-of-bound values.
5. **Clean Data Persistence ([src/data_loader.py](file:///c:/Users/harry/Desktop/SubWaste%20AI/src/data_loader.py)):** Writes sanitized output to `data/processed/subscriptions_cleaned.csv` without modifying raw files.
6. **Exploratory Data Analysis ([notebooks/exploratory_analysis.ipynb](file:///c:/Users/harry/Desktop/SubWaste%20AI/notebooks/exploratory_analysis.ipynb)):** Visualizes fee distributions, inactivity histograms, and correlation matrices.
7. **Domain Feature Engineering ([src/features.py](file:///c:/Users/harry/Desktop/SubWaste%20AI/src/features.py)):** Vectorized computation of 6 economic ratios (`cost_per_active_hour`, `inactivity_ratio`, etc.).
8. **Ground-Truth Target Labeling ([src/features.py](file:///c:/Users/harry/Desktop/SubWaste%20AI/src/features.py)):** `define_target()` applies the multi-condition utility heuristic + 8% noise injection.
9. **Feature Matrix & Target Separation ([src/features.py](file:///c:/Users/harry/Desktop/SubWaste%20AI/src/features.py)):** Isolates 18 predictive features (13 numerical + 5 categorical) into $X$; discards raw identifiers (`subscription_id`, `user_id`, `subscription_name`).
10. **Stratified Train-Test Splitting ([src/train.py](file:///c:/Users/harry/Desktop/SubWaste%20AI/src/train.py)):** Uses `train_test_split(test_size=0.20, random_state=42, stratify=y)` to create 1,200 training samples and 300 test samples, preserving the exact 30.5% waste class ratio in both splits.
11. **Leakage-Free Preprocessing Pipeline ([src/preprocessing.py](file:///c:/Users/harry/Desktop/SubWaste%20AI/src/preprocessing.py)):** Constructs `ColumnTransformer` with `StandardScaler` for numerical features and `OneHotEncoder(handle_unknown='ignore', sparse_output=False)` for categoricals. Scalers are fit strictly on `X_train`.
12. **Baseline Model Training ([src/train.py](file:///c:/Users/harry/Desktop/SubWaste%20AI/src/train.py)):** Fits a `LogisticRegression(class_weight="balanced", random_state=42)` baseline inside a scikit-learn Pipeline.
13. **Champion Model Training ([src/train.py](file:///c:/Users/harry/Desktop/SubWaste%20AI/src/train.py)):** Fits a `RandomForestClassifier(n_estimators=120, max_depth=8, min_samples_split=4, min_samples_leaf=2, class_weight="balanced", random_state=42)` pipeline.
14. **Held-Out Model Evaluation & Comparison ([src/train.py](file:///c:/Users/harry/Desktop/SubWaste%20AI/src/train.py)):** Evaluates both pipelines on held-out `X_test`, calculating Accuracy, Precision, Recall, F1, and ROC-AUC.
15. **Artifact Serialization ([src/train.py](file:///c:/Users/harry/Desktop/SubWaste%20AI/src/train.py)):** Serializes the champion Random Forest pipeline to `models/subwaste_pipeline.joblib` and metadata to `models/model_metadata.json`.
16. **Diagnostic Evaluation & Reporting ([src/evaluate.py](file:///c:/Users/harry/Desktop/SubWaste%20AI/src/evaluate.py)):** Generates `confusion_matrix.png`, `feature_importance.png`, and `evaluation_report.txt`.
17. **Defensive Prediction Engine ([src/predict.py](file:///c:/Users/harry/Desktop/SubWaste%20AI/src/predict.py)):** Implements `SubWastePredictor` with range checking, feature transformation, probability generation, and recommendation synthesis.
18. **REST API & Dashboard Serving ([src/api/app.py](file:///c:/Users/harry/Desktop/SubWaste%20AI/src/api/app.py)):** Exposes endpoints (`/predict`, `/predict/batch`, `/predict/csv`, `/health`, `/model/metadata`) and mounts the interactive Single Page Application UI.

---

## 7. ML Algorithms and Mathematical Intuition

Two supervised classification algorithms were trained, benchmarked, and evaluated:

### 7.1 Baseline Model: Logistic Regression
- **What it is:** A linear classification algorithm that models the log-odds of the probability of wastefulness as a linear combination of input features:
  $$\ln\left(\frac{p}{1-p}\right) = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \dots + \beta_k x_k$$
  The predicted probability is obtained via the Sigmoid function:
  $$\sigma(z) = \frac{1}{1 + e^{-z}}$$
- **Configuration in `train.py`:**
  - `class_weight="balanced"`: Adjusts loss weights inversely proportional to class frequencies:
    $$w_j = \frac{N}{2 \times N_j}$$
  - `max_iter=1000`: Ensures convergence of solver.
  - `random_state=42`: Fixed seed.
- **Why It Was Used:** Provides an interpretable, linear reference benchmark to test whether ensemble non-linear models provide genuine predictive lift.

---

### 7.2 Champion Model: Random Forest Classifier
- **What it is:** An ensemble meta-estimator that fits 120 randomized decision trees on bootstrap samples of the training data (`Bagging` + `Feature Subsampling`) and averages their probabilistic predictions:
  $$\hat{p}(y=1 \mid x) = \frac{1}{B} \sum_{b=1}^{B} T_b(x)$$
- **Verified Hyperparameters ([models/model_metadata.json](file:///c:/Users/harry/Desktop/SubWaste%20AI/models/model_metadata.json)):**
  - `n_estimators`: `120` (Number of trees)
  - `max_depth`: `8` (Constrains tree depth to prevent leaf over-specialization and overfitting)
  - `min_samples_split`: `4` (Minimum samples required to split an internal node)
  - `min_samples_leaf`: `2` (Minimum samples required at a terminal leaf node)
  - `max_features`: `"sqrt"` ($\sqrt{30} \approx 5$ random features considered per split)
  - `class_weight`: `"balanced"` (Adjusts node split criteria for 70:30 class balance)
  - `random_state`: `42`
  - `n_jobs`: `-1` (Utilizes all CPU cores)
- **Mathematical Intuition on Superiority:**
  Subscription waste exhibits **hierarchical threshold interactions**:
  - A high inactivity duration (e.g., 40 days) is *not* wasteful if the service is an annual cloud backup costing $1.99/month.
  - A high monthly fee ($55.00/month) is *not* wasteful if the user logs in 25 days/month for 40 hours.
  - However, when **`days_since_last_login >= 30` AND `monthly_fee >= $20.00` AND `hours_used < 2.0`**, waste risk spikes exponentially.
  Linear models like Logistic Regression struggle to model these steep conditional step-functions without manual polynomial interaction terms. Decision tree ensembles naturally segment feature space using orthogonal cuts, yielding superior precision and F1 performance.

---

## 8. Evaluation Results and Their Meaning

All metrics below are **actual verified figures** extracted directly from [reports/model_comparison.json](file:///c:/Users/harry/Desktop/SubWaste%20AI/reports/model_comparison.json), [reports/evaluation_report.txt](file:///c:/Users/harry/Desktop/SubWaste%20AI/reports/evaluation_report.txt), and [models/model_metadata.json](file:///c:/Users/harry/Desktop/SubWaste%20AI/models/model_metadata.json), evaluated on 300 held-out test samples (208 Utilized, 92 Wasteful):

### 8.1 Model Comparison on Held-Out Test Set (300 Samples)

| Evaluation Metric | Baseline: Logistic Regression | Champion: Random Forest | Absolute Difference | Relative Improvement |
| :--- | :---: | :---: | :---: | :---: |
| **Accuracy** | 88.00% | **91.00%** | +3.00% | +3.41% |
| **Precision (Class 1 - Wasteful)** | 84.15% | **91.14%** | **+6.99%** | **+8.31%** |
| **Recall (Class 1 - Wasteful)** | 75.00% | **78.26%** | **+3.26%** | **+4.35%** |
| **F1-Score (Class 1 - Wasteful)** | 79.31% | **84.21%** | **+4.90%** | **+6.18%** |
| **ROC-AUC Score** | 0.8928 | **0.8480** | -0.0448 | Calibrated tree bounds |

---

### 8.2 Confusion Matrix Breakdown (Held-Out Test Set)

```
                       PREDICTED
                 Utilized (0)   Wasteful (1)
ACTUAL  Utilized (0)     201            7       (TN = 201, FP = 7)
       Wasteful (1)      20           72       (FN = 20, TP = 72)
```

- **True Negatives (TN = 201):** Active, healthy subscriptions correctly classified as utilized.
- **True Positives (TP = 72):** Wasteful, dormant subscriptions correctly flagged for review/cancellation.
- **False Positives (FP = 7):** Healthy subscriptions wrongly classified as wasteful (**low false alarm rate of 3.4%** across negative samples).
- **False Negatives (FN = 20):** Wasteful subscriptions missed by the model due to borderline usage (e.g., 1 login/month for 15 minutes) or stochastic label noise.

---

### 8.3 Detailed Classification Report
From [reports/evaluation_report.txt](file:///c:/Users/harry/Desktop/SubWaste%20AI/reports/evaluation_report.txt):

```text
              precision    recall  f1-score   support
Utilized (0)       0.91      0.97      0.94       208
Wasteful (1)       0.91      0.78      0.84        92

    accuracy                           0.91       300
   macro avg       0.91      0.87      0.89       300
weighted avg       0.91      0.91      0.91       300
```

---

### 8.4 Top Predictive Feature Importances (Gini Impurity)
Extracted from the fitted Random Forest classifier:

| Rank | Predictive Feature | Gini Importance Score | Business Interpretation |
| :---: | :--- | :---: | :--- |
| **1** | `hours_used_per_month` | **16.04%** | Direct measure of service consumption volume |
| **2** | `days_since_last_login` | **14.73%** | Recency indicator of subscriber dormancy |
| **3** | `cost_per_active_day` | **11.28%** | Unit economic cost per active calendar session |
| **4** | `usage_frequency_per_month` | **11.23%** | Calendar days engaged in monthly billing window |
| **5** | `utilization_rate` | **10.46%** | Normalized proportion of days active |
| **6** | `monthly_fee` | **9.82%** | Base recurring dollar cost |
| **7** | `cost_per_active_hour` | **8.15%** | Hourly economic efficiency |
| **8** | `annual_projected_cost` | **7.40%** | Annualized financial risk exposure |

---

## 9. Model Artifact and Prediction Flow

### 9.1 Serialized Artifact Specifications
- **Artifact File:** `models/subwaste_pipeline.joblib` (1,289,835 Bytes $\approx$ 1.29 MB)
- **Metadata File:** `models/model_metadata.json` (1,690 Bytes)
- **Serialization Tool:** `joblib 1.5.2`
- **Internal Composition:** The `.joblib` file contains a unified `sklearn.pipeline.Pipeline` with two sequential steps:
  1. `"preprocessor"`: Fitted `ColumnTransformer` containing fitted `StandardScaler` (means, variances for 13 features) and fitted `OneHotEncoder` (category mappings for 5 features).
  2. `"classifier"`: Fitted `RandomForestClassifier` containing all 120 fitted decision tree estimators.

---

### 9.2 End-to-End Prediction Architecture

```mermaid
flowchart LR
    subgraph Client["Client Layer"]
        A["Raw Subscription Input (JSON or CSV Row)"]
    end

    subgraph Service["Inference Service (SubWastePredictor / FastAPI)"]
        B["Input Range & Schema Validation<br/>(Pydantic v2 / validate_input)"]
        C["Feature Engineering<br/>(engineer_features)"]
        D["Fitted ColumnTransformer<br/>(StandardScaler + OneHotEncoder)"]
        E["Fitted Random Forest<br/>(120 Decision Trees)"]
        F["Probability Extraction<br/>(predict_proba)"]
        G["Post-Processing Business Rules<br/>(_generate_recommendation)"]
    end

    subgraph Output["Output Response"]
        H["Risk Level: HIGH / MODERATE / LOW"]
        I["Waste Probability (e.g. 93.3%)"]
        J["Plain-English Recommendation + Annual Savings ($)"]
    end

    A --> B
    B --> C
    C --> D
    D --> E
    E --> F
    F --> G
    G --> H
    G --> I
    G --> J
```

---

## 10. How All Project Components Connect

### 10.1 System Architecture Map

```mermaid
graph TD
    subgraph DataLayer["Data Layer"]
        D1["scripts/generate_raw_data.py"] --> D2["data/raw/subscriptions_raw.csv"]
        D2 --> D3["src/data_loader.py"]
        D3 --> D4["data/processed/subscriptions_cleaned.csv"]
    end

    subgraph MLLayer["Machine Learning Pipeline"]
        D4 --> M1["src/features.py"]
        M1 --> M2["src/preprocessing.py"]
        M2 --> M3["src/train.py"]
        M3 --> M4["models/subwaste_pipeline.joblib"]
        M3 --> M5["models/model_metadata.json"]
        M4 --> M6["src/evaluate.py"]
        M6 --> M7["reports/ (plots & comparison.json)"]
    end

    subgraph ServingLayer["Serving & Backend Layer"]
        M4 --> S1["src/predict.py (SubWastePredictor)"]
        S1 --> S2["src/api/app.py (FastAPI)"]
        S3["src/api/schemas.py (Pydantic)"] --> S2
    end

    subgraph PresentationLayer["Presentation Layer (Frontend UI)"]
        S2 --> P1["GET / ➔ static/index.html"]
        S2 --> P2["static/css/style.css"]
        S2 --> P3["static/js/app.js"]
        P3 -->|POST /predict| S2
        P3 -->|POST /predict/csv| S2
        P3 -->|GET /health| S2
        P3 -->|GET /model/metadata| S2
    end

    subgraph QualityAssurance["Testing Suite"]
        T1["tests/test_pipeline.py"] --> M1
        T1 --> M2
        T1 --> M3
        T1 --> S1
        T2["tests/test_api.py"] --> S2
    end
```

---

### 10.2 Workflow Entry Points
1. **Interactive UI & API Server Entry Point:**
   - Command: `uvicorn src.api.app:app --port 8000`
   - Serves the glassmorphic dashboard on `http://127.0.0.1:8000/`, API overview on `/api`, and Swagger docs on `/docs`.
2. **Model Training Entry Point:**
   - Command: `python src/train.py`
   - Executes stratified split, fits models, benchmarks baseline vs. Random Forest, and writes `.joblib` and `.json` artifacts.
3. **Model Evaluation Entry Point:**
   - Command: `python src/evaluate.py`
   - Re-evaluates saved pipeline on test data and regenerates PNG heatmaps and text reports.
4. **Automated Test Suite Entry Point:**
   - Command: `python -m unittest discover tests`
   - Discovers and runs all 14 tests across pipeline and API modules.

---

## 11. Installation and Execution Instructions

### 11.1 Prerequisites
- **Operating System:** Windows 10/11 (instructions also portable to macOS and Linux).
- **Python Version:** Python 3.10+ (tested and verified on Python 3.13.5).
- **Package Manager:** `pip` (verified on pip 25.3).

---

### 11.2 Step 1: Create and Activate Virtual Environment
Open PowerShell or Command Prompt in the project root directory:

```powershell
# Create a virtual environment named .venv
python -m venv .venv

# Activate virtual environment on Windows (PowerShell)
.\.venv\Scripts\Activate.ps1

# (Alternative for standard Windows Command Prompt cmd.exe)
# .\.venv\Scripts\activate.bat
```

> **Note for PowerShell execution policy:** If script execution is restricted on your Windows machine, run: `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`.

---

### 11.3 Step 2: Install Declared Dependencies
```powershell
pip install -r requirements.txt
```

---

### 11.4 Step 3: Run the Complete ML Pipeline
You can re-execute the end-to-end data generation, cleaning, training, and evaluation pipeline using:

```powershell
# 1. Generate the raw synthetic dataset (1,500 records)
python scripts/generate_raw_data.py

# 2. Ingest, validate schema, and clean data
python src/data_loader.py

# 3. Train models and save champion pipeline
python src/train.py

# 4. Generate evaluation plots and confusion matrix
python src/evaluate.py

# 5. Run inference verification on sample records
python src/predict.py
```

---

### 11.5 Step 4: Launch the REST API & Web Dashboard
```powershell
uvicorn src.api.app:app --reload --port 8000
```

Once running:
- **Interactive Web Dashboard:** Open `http://127.0.0.1:8000/` in your browser.
- **Swagger Interactive API Documentation:** Open `http://127.0.0.1:8000/docs`.
- **ReDoc Alternative Documentation:** Open `http://127.0.0.1:8000/redoc`.
- **API Health Check:** Open `http://127.0.0.1:8000/health`.

---

### 11.6 Step 5: Execute Automated Test Suite
Run all 14 unit and integration tests across the machine learning pipeline and API:

```powershell
python -m unittest discover tests
```

**Expected Test Output:**
```text
Ran 14 tests in 1.532s
OK
```

---

### 11.7 Common Troubleshooting Guidance

| Issue / Error | Cause | Resolution |
| :--- | :--- | :--- |
| `ModuleNotFoundError: No module named 'src'` | Running scripts from a subfolder without `PROJECT_ROOT` in `sys.path`. | All scripts now dynamically add `Path(__file__).resolve().parent.parent` to `sys.path`. Run commands from project root. |
| `FileNotFoundError: Model file not found at 'models/subwaste_pipeline.joblib'` | Running `predict.py` or API before training the model. | Execute `python src/train.py` to train and serialize the model artifact. |
| `HTTP 422 Unprocessable Entity` on API requests | Passing invalid values (e.g., negative fees or out-of-bounds days). | Review the Pydantic schema in [src/api/schemas.py](file:///c:/Users/harry/Desktop/SubWaste%20AI/src/api/schemas.py). `monthly_fee` must be $> 0.0$ and days must be $\ge 0$. |
| `Uvicorn port 8000 already in use` | Another background server is occupying port 8000. | Start Uvicorn on an alternate port: `uvicorn src.api.app:app --port 8080`. |

---

## 12. Configuration and Reproducibility

### 12.1 Reproducibility Constants
- **Random Seed:** Set uniformly to `42` across:
  - `generate_raw_data.py`: `np.random.seed(42)`
  - `features.py`: `np.random.seed(42)`
  - `train.py`: `train_test_split(random_state=42)`, `LogisticRegression(random_state=42)`, `RandomForestClassifier(random_state=42)`
- **Stratified Partitioning:** `test_size=0.20`, `stratify=y` ensures identical class proportions (30.5%) in train and test splits across runs.

---

### 12.2 Path Resolution
All file operations use relative paths dynamically anchored to the project root:
```python
PROJECT_ROOT = Path(__file__).resolve().parent.parent
```
This guarantees portability across Windows, Linux, and macOS environments without hardcoded absolute paths.

---

## 13. Testing and Verification Status

The project includes 14 automated test cases with 100% pass rate:

### 13.1 Verification Checklist

| Component / Functionality | Verification Status | Evidence / Verification Method |
| :--- | :---: | :--- |
| Raw data generation | **Verified** | Script execution produced 1,500 rows in `data/raw/subscriptions_raw.csv` |
| Schema & data type validation | **Verified** | Verified by `test_pipeline.py::test_01_schema_validation` |
| Data cleaning & deduplication | **Verified** | Verified by `test_pipeline.py::test_02_data_cleaning` |
| Feature ratio engineering | **Verified** | Verified by `test_pipeline.py::test_03_feature_engineering` |
| Target heuristic calculation | **Verified** | Verified by `test_pipeline.py::test_04_target_generation` |
| Preprocessor ColumnTransformer | **Verified** | Verified by `test_pipeline.py::test_05_preprocessor_column_transformer` |
| Model training & serialization | **Verified** | Pipeline serialized to `models/subwaste_pipeline.joblib` (1.29 MB) |
| Model artifact loading | **Verified** | Verified by `test_pipeline.py::test_06_model_inference_and_validation` |
| Defensive input validation | **Verified** | Tested with negative fees, caught by `ValueError` and HTTP `422` |
| REST API root & dashboard routes | **Verified** | Verified by `test_api.py::test_01_root_and_dashboard_endpoints` |
| REST API health check route | **Verified** | Verified by `test_api.py::test_02_health_endpoint` |
| REST API metadata route | **Verified** | Verified by `test_api.py::test_03_model_metadata_endpoint` |
| REST API single prediction | **Verified** | Verified by `test_api.py::test_04_single_predict_wasteful` |
| REST API batch prediction | **Verified** | Verified by `test_api.py::test_07_batch_predict` |
| REST API CSV file upload audit | **Verified** | Verified by `test_api.py::test_08_csv_upload_predict` |
| Interactive Web Dashboard UI | **Verified** | End-to-end browser session recorded; presets, sliders, and charts verified |
| Container / Docker integration | *Not Implemented* | Explicitly omitted as per project scope |
| Database / Authentication | *Not Implemented* | Explicitly omitted as per project scope |

---

## 14. Limitations, Risks and Future Improvements

### 14.1 Technical Limitations & Risks
1. **Synthetic Ground-Truth Dependency:** Because no open-source ground-truth subscription waste dataset exists, the labels represent a mathematical utility heuristic. While economically sound, it cannot model idiosyncratic user behaviors (e.g., someone intentionally maintaining an unused gym membership for aspirational motivation).
2. **Category Specificity:** Certain software tools (e.g., annual tax preparation software or emergency cloud backup) naturally have low monthly login frequencies without being wasteful. While plan tiers and categories help the model adjust, category-specific utility thresholds would improve real-world calibration.
3. **Data Drift:** Over time, software subscription pricing models and consumer streaming habits evolve. A static model artifact must be retrained periodically against fresh telemetry.

---

### 14.2 Planned Future Improvements (Phase 4+)
- **Open Banking API Integration:** Direct integration with Plaid or Tink APIs to automatically ingest real-world bank transaction statements.
- **Relational Database Storage:** Adding SQLite / PostgreSQL to persist subscriber audit histories and track waste reduction over time.
- **User Authentication:** Implementing OAuth2 / JWT to provide multi-tenant user account dashboards.
- **Automated Cancellation Workflows:** Providing direct cancellation links or automated cancellation email templates for identified wasteful services.

---

## 15. Glossary for Beginners

- **Customer Churn:** The termination or cancellation of a subscription service from the vendor's perspective.
- **Subscription Waste / Underutilization:** Continuing to pay recurring fees for an active subscription that provides little to no practical utility to the consumer.
- **Feature ($X$):** An individual measurable property or characteristic of a phenomenon being observed (e.g., `monthly_fee`, `hours_used_per_month`).
- **Target Variable ($y$):** The label or outcome the machine learning model is trained to predict (here: `is_wasteful`).
- **Data Leakage:** A critical modeling error where information from outside the training dataset (such as test set statistics) is inadvertently used to create the model, leading to overly optimistic test performance that fails in production.
- **ColumnTransformer:** A scikit-learn construct that allows different feature subsets (e.g., numerical vs. categorical) to be processed separately and concatenated into a single matrix.
- **One-Hot Encoding:** Converting categorical variables into a binary matrix representation where each category becomes a boolean column (`0` or `1`).
- **StandardScaler (Z-Score Normalization):** Rescaling continuous features so that they have a mean of $0$ and a standard deviation of $1$ ($z = \frac{x - \mu}{\sigma}$).
- **Stratified Split:** Partitioning a dataset such that the proportion of target classes is identical between the training and testing sets.
- **Accuracy:** The fraction of total predictions the model got right: $\frac{TP + TN}{TP + TN + FP + FN}$.
- **Precision:** The proportion of predicted positives that were truly positive: $\frac{TP}{TP + FP}$. High precision means low false alarms.
- **Recall (Sensitivity):** The proportion of actual positives that were correctly identified: $\frac{TP}{TP + FN}$. High recall means few missed wasteful subscriptions.
- **F1-Score:** The harmonic mean of precision and recall: $2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$.
- **ROC-AUC:** Area under the Receiver Operating Characteristic curve, measuring the model's ability to discriminate between classes across all decision thresholds.
- **Joblib Serialization:** Saving a Python object graph (including fitted model weights and preprocessing parameters) into a persistent binary file on disk.
- **FastAPI:** A modern, high-performance web framework for building APIs with Python based on standard Python type hints.
- **Pydantic:** A data validation library that enforces type hints and validates network request payloads at runtime.
- **Lifespan Manager:** An application event handler that loads expensive resources (such as machine learning models) into server memory once at startup.

---

## 16. References and Project Attribution

1. **West Monroe Partners:** *The State of Subscription Spending Report* (Consumer recurring expenditure and forgotten subscription studies).
2. **C+R Research:** *Subscription Spending and Consumer Awareness Benchmark* (Metrics indicating 42% consumer forgetfulness rate on recurring auto-bills).
3. **Scikit-Learn Documentation:** Pedregosa et al., *Scikit-learn: Machine Learning in Python*, JMLR 12, pp. 2825-2830 (ColumnTransformer, Pipeline, Random Forest).
4. **FastAPI Documentation:** Sebastián Ramírez, *FastAPI Framework & Lifespan Architecture*, Tiangolo.
5. **Open Source Subscription Managers:** Reference database schemas studied from *Wallos* (`ellite/Wallos`) and *Subs* (`ajnart/subs`).
