from app import db
from werkzeug import generate_password_hash, check_password_hash
import datetime


class User(db.Model):
    __tablename__ = 'users'
    id = db.Column(db.Integer, primary_key=True)
    firstname = db.Column(db.String(100))
    lastname = db.Column(db.String(100))
    email = db.Column(db.String(120), unique=True)
    password = db.Column(db.String(54))
    created_on = db.Column(db.DateTime, server_default=db.func.now())

    def __init__(self, firstname, lastname, email, password, created_on=None):  # noqa
        self.firstname = firstname
        self.lastname = lastname
        self.email = email.lower()
        self.set_password(password)
        self.created_on = datetime.datetime.utcnow()

    def __repr__(self):
        return '<User %r>' % (self.email)

    def set_password(self, password):
        self.password = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password, password)

    def is_authenticated(self):
        return True

    def is_active(self):
        return True

    def is_anonymous(self):
        return False

    def get_id(self):
        return str(self.email)
