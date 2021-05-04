FROM continuumio/anaconda3:2020.11
 
ADD . /code
WORKDIR /code

ENTRYPOINT ["python", "web_app_deployment/nba_salary_app.py"]