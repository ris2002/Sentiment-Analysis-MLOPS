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
        }
    }
}