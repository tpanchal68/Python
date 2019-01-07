from flask_wtf import Form
from wtforms import BooleanField, TextField, PasswordField, validators  # noqa
from models import db, User


class RegistrationForm(Form):
    firstname = TextField("First name",  [validators.Required("Please enter your first name.")])  # noqa
    lastname = TextField("Last name",  [validators.Required("Please enter your last name.")])  # noqa
    email = TextField('Email Address:', [validators.Required("Please enter your email address."), validators.Email("Please enter your email address.")])  # noqa
    password = PasswordField('Password:', [
        validators.Required(),
        validators.EqualTo('confirm', message='Passwords must match.')
    ])
    confirm = PasswordField('Repeat Password:')

    def __init__(self, *args, **kwargs):
        Form.__init__(self, *args, **kwargs)

    def validate(self):
        if not Form.validate(self):
            return False
        user = User.query.filter_by(email=self.email.data.lower()).first()
        if user:
            self.email.errors.append("That email is already taken")
            return False
        else:
            return True


class LoginForm(Form):
    email = TextField("Email address:",  [validators.Required("Please enter your email address."), validators.Email("Please enter your email address.")])  # noqa
    password = PasswordField('Password:', [validators.Required("Please enter a password.")])  # noqa

    def __init__(self, *args, **kwargs):
        Form.__init__(self, *args, **kwargs)

    def validate(self):
        if not Form.validate(self):
            return False

        user = User.query.filter_by(email=self.email.data.lower()).first()
        if user and user.check_password(self.password.data):
            return True
        else:
            self.email.errors.append("Invalid e-mail or password")
            return False
