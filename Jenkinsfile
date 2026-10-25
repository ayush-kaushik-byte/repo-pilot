pipeline {
    agent any

    options {
        timestamps()
        disableConcurrentBuilds()
    }

    environment {
        IMAGE_NAME = 'repopilot:local'
        DOCKERHUB_REPO = 'ayushk39/repopilot'
    }

    stages {
        stage('Checkout') {
            steps {
                checkout scm
            }
        }

        stage('Test') {
            steps {
                sh '''
                    set -eu
                    python3 --version
                    python3 -m venv .venv
                    .venv/bin/python -m pip install --upgrade pip
                    .venv/bin/python -m pip install -r requirements.txt
                    PYTHONPATH=src .venv/bin/python -m pytest -q
                '''
            }
        }

        stage('Build Docker Image') {
            steps {
                sh '''
                    set -eu
                    docker build \
                        -t "${IMAGE_NAME}" \
                        -t "${DOCKERHUB_REPO}:${GIT_COMMIT}" \
                        .
                '''
            }
        }

        stage('Publish to Docker Hub') {
            steps {
                withCredentials([
                    usernamePassword(
                        credentialsId: 'dockerhub-repopilot',
                        usernameVariable: 'DOCKERHUB_USERNAME',
                        passwordVariable: 'DOCKERHUB_TOKEN'
                    )
                ]) {
                    sh '''
                        set -eu
                        set +x
                        trap 'docker logout >/dev/null 2>&1 || true' EXIT

                        printf '%s' "$DOCKERHUB_TOKEN" |
                            docker login \
                                --username "$DOCKERHUB_USERNAME" \
                                --password-stdin

                        docker push "${DOCKERHUB_REPO}:${GIT_COMMIT}"
                    '''
                }
            }
        }
    }

    post {
        always {
            echo 'RepoPilot CI pipeline finished.'
        }
    }
}
