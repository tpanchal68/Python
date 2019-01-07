from flask import Flask
from flask_sqlalchemy import SQLAlchemy
import os

basedir = os.path.abspath(os.path.dirname(__file__))
secret_key = os.urandom(24)

app = Flask(__name__)
app.secret_key = secret_key  # Needed for form csrf
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///' + os.path.join(basedir, 'development.db')  # noqa
db = SQLAlchemy(app)
db.init_app(app)

from app import views, models
