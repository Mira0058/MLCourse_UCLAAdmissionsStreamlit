# Machine Learning Project with Streamlit

A machine learning project with interactive Streamlit visualization.

## Folder Structure

```
project/
├── data/                # Dataset files
├── models/              # Saved models
├── notebooks/           # Jupyter notebooks
├── src/
│   ├── __init__.py      # Makes src a Python package
│   ├── data_processing.py
│   ├── model_training.py
│   ├── visualization.py
│   └── utils.py
├── app.py               # Streamlit application
├── requirements.txt
└── README.md
```

## Setup & Running

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the Streamlit app:
```bash
streamlit run app.py
```
