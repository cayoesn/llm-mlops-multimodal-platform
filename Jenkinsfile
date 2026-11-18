pipeline {
    agent any

    options {
        timestamps()
        timeout(time: 20, unit: 'MINUTES')
    }

    stages {
        stage('Stage 1: Lint, Strict Mypy & Unit Suite') {
            steps {
                sh '''
                    uv venv .venv --python 3.12 --seed
                    uv pip install --python .venv/bin/python -e ".[dev]"
                    .venv/bin/ruff check app tests dags feature_repo
                    .venv/bin/mypy app dags feature_repo
                '''
            }
        }

        stage('Stage 2: Feast, DVC/CML, MLflow, RAGAS & Multimodal Gate') {
            steps {
                sh '''
                    .venv/bin/pytest tests -v --cov=app --cov-fail-under=95
                    .venv/bin/python -m app.mlops.dvc_pipeline
                '''
            }
        }

        stage('Stage 3: Container Build & K8s Manifest Check') {
            steps {
                sh '''
                    docker compose config -q
                    docker build -t llm-mlops-multimodal-platform:${BUILD_NUMBER} .
                '''
            }
        }
    }
}
