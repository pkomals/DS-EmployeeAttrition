# Employee Attrition Prediction

## Project Overview
This project aims to predict employee attrition using machine learning models. The repository follows a structured pipeline approach to ensure efficient data handling, model training, and evaluation.

## Directory Structure
```
.dvc/               # DVC (Data Version Control) files
artifact/           # Stores generated artifacts during training and evaluation
catboost_info/      # Stores CatBoost model-related information
src/EmpAttrition/   # Main source code
  ├── Components/   # Modules for different components of the pipeline
  ├── Notebook/     # Jupyter Notebooks for analysis and experimentation
  ├── Pipeline/     # Code for the end-to-end ML pipeline
  ├── __init__.py   # Package initialization file
  ├── exception.py  # Custom exception handling
  ├── logger.py     # Logging functionality
  ├── utils.py      # Utility functions
.dvcignore          # Ignore patterns for DVC
.gitignore          # Ignore patterns for Git
Dockerfile          # Docker configuration for containerization
LICENSE             # License file for the project
```

## Installation
### Prerequisites
Ensure you have the following installed:
- Python (>=3.8)
- pip
- Git
- DVC (Data Version Control)
- Docker (optional)

### Setting Up the Project
Clone the repository:
```bash
git clone <repo-url>
cd <repo-folder>
```
Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
```
Install required dependencies:
```bash
pip install -r requirements.txt
```

## Running the Project
### Data Versioning with DVC
If using DVC, pull the dataset:
```bash
dvc pull
```

### Running the ML Pipeline
Execute the pipeline:
```bash
python src/EmpAttrition/Pipeline/train.py
```

### Logging and Monitoring
Logs are stored in the `logger.py` module and will be generated during execution.

## Containerization (Optional)
To run the project in a Docker container:
```bash
docker build -t emp_attrition .
docker run -p 8080:8080 emp_attrition
```

