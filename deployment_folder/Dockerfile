#Set which version of python you wannt to use in the container
FROM python:3.10-slim
#Create a new workinng directory inside a container
WORKDIR /app
# below '..' represents go one level up from deployment_folder/ to access and Copy  requirements.txt
COPY ../requirements.txt .
#install the requirements.txt
RUN install --no-cache-dir -r requirements.txt
#Copy the whole project into your container
COPY ../ .


