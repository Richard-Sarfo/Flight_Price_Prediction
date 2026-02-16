# Flight Price Analysis & Prediction - Complete Project Suite

A comprehensive suite of machine learning, data engineering, and DevOps projects for flight price prediction and analysis using various technologies including Python, Apache Spark, Apache Airflow, and Flask.

## 📁 Project Structure

This workspace contains multiple interconnected projects:

### 1. **ML** - Machine Learning Development
Core machine learning project for flight price prediction.
- **Location**: `ML/`
- **Key Files**: 
  - `Flight_price.ipynb` - Jupyter notebook with EDA and model training
  - `train_and_save.py` - Model training and artifact generation
  - `api.py` - Flask API for predictions
  - `Feature_importance.py` - Feature importance analysis
  - `generate_insights_text.py` - Automated insight generation
- **Dataset**: `Dataset/Flight_Price_Dataset_of_Bangladesh.csv`
- **EDA Modules**: Statistical analysis, KPIs, visualization

### 2. **Prediction** - REST API for Model Serving
Production-ready Flask API for flight price predictions.
- **Location**: `Prediction/`
- **Key Components**:
  - `Api.py` - Flask API server
  - `App.py` - Application logic
  - `Train&Save.py` - Model training and serialization
  - `Api_testing.py` - API endpoint testing
  - Trained models and encoders in `model/`
- **Usage**: Python, Flask, scikit-learn

**Quick Start**:
```bash
cd Prediction
pip install -r requirements.txt
python Train&Save.py    # Generate model artifacts
python Api.py           # Start API server
python Api_testing.py   # Test endpoints
```

### 3. **Spark_Analysis** - Big Data ETL with Apache Spark
Distributed data processing and analysis using Apache Spark and Docker.
- **Location**: `Spark_Analysis/`
- **Key Features**:
  - Containerized Spark environment (Docker & Docker Compose)
  - ETL pipeline for movie data processing
  - Data cleaning and extraction functions
  - KPI analysis and visualization
  - Schema validation
- **Data Source**: `etl/tmdb_movies.json`

**Setup**:
```bash
cd Spark_Analysis
docker-compose up
# Access Spark cluster and run analysis
```

### 4. **Airflow** - Workflow Orchestration
Apache Airflow DAG for automated flight price analysis pipeline.
- **Location**: `Airflow/`
- **DAGs**:
  - `flight_price_analysis_dag.py` - Main analysis pipeline
- **Jobs**:
  - `Ingestion.py` - Data ingestion
  - `compute_kpis.py` - KPI computation
- **Infrastructure**: Docker containerized Airflow environment
- **Logs**: Automatic logging in `logs/` directory

**Setup**:
```bash
cd Airflow
docker-compose up
# Access Airflow UI at http://localhost:8080
```

### 5. **DevOps** - Data Pipeline & CI/CD
DevOps practices with ETL pipeline, testing, and deployment strategies.
- **Location**: `DevOps/`
- **Components**:
  - `src/Extract.py` - Data extraction
  - `src/Transform.py` - Data transformation
  - `src/Load.py` - Data loading
  - `src/report.py` - Report generation
  - `Scripts/run_pipeline.py` - Pipeline orchestration
- **Documentation**:
  - Sprint reviews and retrospectives
  - Project planning and tracking
- **Testing**: Automated test suite in `test/`

**Run Pipeline**:
```bash
cd DevOps
pip install -r requirements.txt
python Scripts/run_pipeline.py
```

### 6. **Capstone** - Data Quality & Validation
Data quality monitoring and validation framework.
- **Location**: `Capstone/`
- **Key Features**:
  - `data_quality_validator.py` - Validation scripts
  - `data_quality_documentation.md` - Quality standards
  - `quality_report.json` - Test results
  - Screenshots and logs

### 7. **pbi** - Power BI Dashboard
Business intelligence reports and visualizations.
- **Location**: `pbi/`
- Contains dashboards and reports for stakeholder visualization

---

## 🔄 Data Flow

```
Flight_Price_Dataset.csv
    ↓
├→ ML (Model Development)
│   └→ Trained Models (PKL files)
│       └→ Prediction API (Flask)
│
├→ Spark_Analysis (Big Data Processing)
│   └→ Distributed ETL & KPIs
│
├→ Airflow (Orchestration)
│   └→ Automated Pipeline Execution
│
├→ DevOps (CI/CD Pipeline)
│   └→ Data Quality & Deployment
│
└→ Capstone (Quality Assurance)
    └→ Validation Reports
```

---

## 🛠 Technology Stack

- **Languages**: Python 3.x
- **ML/AI**: scikit-learn, pandas, numpy
- **Web Framework**: Flask
- **Big Data**: Apache Spark
- **Orchestration**: Apache Airflow
- **Containerization**: Docker, Docker Compose
- **Data Validation**: Custom validation frameworks
- **BI Tools**: Power BI
- **Version Control**: Git

---

## 📊 Key Datasets

**Flight Price Dataset of Bangladesh**
- Multiple copies in:
  - `ML/Dataset/Flight_Price_Dataset_of_Bangladesh.csv`
  - `Airflow/data/Flight_Price_Dataset_of_Bangladesh.csv`
  - `Prediction/data/Flight_Price_Dataset_of_Bangladesh.csv`
  - `DevOps/data/`

**Features**:
- Airline information
- Source/Destination airports
- Aircraft type and class
- Booking source and seasonality
- Base fare, taxes, and surcharges
- Advance booking days

---

## 🚀 Getting Started

### Prerequisites
- Python 3.x
- Docker & Docker Compose (for Spark and Airflow projects)
- Git

### Quick Setup

1. **Clone/Navigate to workspace**:
   ```bash
   cd Desktop
   ```

2. **For ML/Prediction API**:
   ```bash
   cd Prediction
   pip install -r requirements.txt
   python Train&Save.py
   python Api.py
   ```

3. **For Spark Analysis**:
   ```bash
   cd Spark_Analysis
   docker-compose up
   ```

4. **For Airflow Orchestration**:
   ```bash
   cd Airflow
   docker-compose up
   ```

5. **For DevOps Pipeline**:
   ```bash
   cd DevOps
   pip install -r requirements.txt
   python Scripts/run_pipeline.py
   ```

---

## 📈 Features

✅ End-to-end machine learning pipeline  
✅ REST API for real-time predictions  
✅ Distributed data processing with Spark  
✅ Automated workflow orchestration with Airflow  
✅ Data quality validation and monitoring  
✅ DevOps best practices and CI/CD  
✅ Business intelligence dashboards  
✅ Comprehensive logging and monitoring  

---

## 📝 Documentation

Each project contains its own README:
- [Prediction README](Prediction/README.md) - API documentation
- [Spark_Analysis README](Spark_Analysis/README.md) - Big data processing
- [Airflow README](Airflow/README.md) - Workflow orchestration
- [DevOps README](DevOps/README.md) - Pipeline details

---

## 👨‍💼 Author

**Richard Anane Sarfo**

---

## 📅 Last Updated

February 16, 2026

---

## 🤝 Contributing

Each project follows git-based version control with documented sprint reviews and retrospectives. See individual project READMEs for contribution guidelines.

---

## 📞 Support

For issues or questions, refer to the specific project documentation or contact the development team.
