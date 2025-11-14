pipeline{
    agent any{
        stages{
            stage('Checkout'){
                 
                    url: 'https://github.com/ris2002/Sentiment-Analysis-MLOPS.git',
                    
            }
            stage('install requirements'){
                sh'''
                python3 -m venv venv
                . venv/bin/activate
                pip install --upgrade pip
                pip install -r requirements.txt
                '''
            }
            stage('Train Program'){
                sh 'python run_pipeline.py'
            }
            stage('Build Docker'){
                steps{
                    sh 'docker build -t sentiment-fastapi:latest .'
                }
            }
            stage('Run Docker'){
                steps{
                    sh 'docker run -d --name sentiment-api -p 8000:8000 sentiment-fastapi:latest'
                }
            }
            stage('Test Deployment'){
                steps{
                     sh '''
                curl -X POST "http://localhost:8000/predict" \
                -H "Content-Type: application/json" \
                -d '{"text": "I love this product!"}'
                '''
                }
            }
        }
    }
}