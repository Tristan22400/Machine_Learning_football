# Tutorial on webapp using flask, sqlite and angular

Adapted from:

* https://flask.palletsprojects.com/en/1.1.x/tutorial/
* https://auth0.com/blog/using-python-flask-and-angular-to-build-modern-apps-part-1/
* https://auth0.com/blog/using-python-flask-and-angular-to-build-modern-web-apps-part-2/
* https://auth0.com/blog/using-python-flask-and-angular-to-build-modern-web-apps-part-3/

## Installation

* Install npm from https://www.npmjs.com/get-npm, then install angular and
  javascript dependencies:

        npm install -g @angular/cli
        npm install --save rxjs-compat
        npm install @angular/material @angular/cdk hammerjs

* Install Python Anaconda from https://docs.conda.io/en/latest/miniconda.html,
  then install a Python virtual environment with required dependencies:

        conda env create -n angularflask -f environment.yml

## Usage

To deploy the website locally, launch flask in a first terminal:

    conda activate angularflask
    FLASK_APP=./backend/src/main.py WEBAPP_DB=./db.sql flask run -h 0.0.0.0

In another terminal run the Angular server:

    conda activate angularflask
    cd frontend && ng serve
