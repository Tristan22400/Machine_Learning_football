import sqlalchemy
import sqlalchemy.orm

from flask import g, current_app


def get_engine():
    if 'engine' not in g:
        g.engine = sqlalchemy.create_engine(
            current_app.config['DATABASE_URI'])
    return g.engine


def get_session():
    if 'session' not in g:
        g.session = sqlalchemy.orm.sessionmaker(bind=get_engine())()
    return g.session
