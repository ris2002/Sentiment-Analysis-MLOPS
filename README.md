# Sentiment Analysis (MLOps Project)

A production-grade **Sentiment Analysis system** with full MLOps integration, including CI/CD, experiment tracking, model registry, deployment, and containerization.

---

## Project Overview

This project demonstrates building and deploying a sentiment analysis model using **MLOps best practices**:

- **CI/CD & Pipeline Orchestration**: Jenkins for automation of pipelines and workflow.
- **Step & Experiment Tracking**: ZenML for pipeline steps and MLflow for experiment tracking.
- **Model Registry & Versioning**: MLflow for registering and managing model versions.
- **Deployment**: Exposed as a REST API using FastAPI.
- **Containerization**: Docker for environment isolation and consistent deployment.

---

## Key Learnings

1. **Proper Naming & Modularization**
   - Clearly name scripts, classes, and functions.
   - Separate functionality into multiple files for maintainability and easier debugging.

2. **Tracking & Debugging**
   - Track pipeline steps and experiments to ensure reproducibility.
   - Write meaningful error messages to identify and fix issues quickly.

3. **Pipeline & Steps**
   - Central part of MLOps workflow.
   - Each pipeline consists of multiple steps, each performing a specific task toward a common goal.

4. **Deployment with FastAPI**
   - Expose the model as an API for easy integration with applications and services.

5. **Containerization with Docker**
   - Isolates the application and packages it into an image.
   - Ensures the program runs consistently across environments.

6. **Model Understanding**
   - Implemented **Logistic Regression** for sentiment classification.
   - Learned to interpret model metrics and results effectively.

7. **MLflow for Experiment Tracking & Model Registry**
   - Track experiments, metrics, and parameters.
   - Register models and manage different versions for production deployment.

---

## Project Structure
sentiment_analysis_mlop/
├─ data/
│ ├─ raw/
│ └─ processed/
├─ notebooks/
│ └─ exploratory_analysis.ipynb
├─ src/
│ ├─ preprocessing/
│ │ └─ email_preprocessor.py
│ ├─ models/
│ │ └─ sentiment_model.py
│ ├─ train_sentiment_model.py
│ └─ evaluate_sentiment_model.py
├─ requirements.txt
└─ README.md

