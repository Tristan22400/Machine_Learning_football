# flask configuration file

import pathlib

DEBUG = True

SECRET_KEY = 'dev'

DATABASE = pathlib.Path(__file__).parent.resolve() / 'instance' / 'db.sql'

DATABASE_URI = f'sqlite:///{DATABASE}'
