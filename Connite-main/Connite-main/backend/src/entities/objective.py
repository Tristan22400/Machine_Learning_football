import marshmallow
import sqlalchemy

from .base import Base


class Objective(Base):
    __tablename__ = 'objectives'
    Team = sqlalchemy.Column(sqlalchemy.String, primary_key=True) 
    Top = sqlalchemy.Column(sqlalchemy.Integer)


    def __init__(self, Team, Top):
        self.Team = Team
        self.Top = Top



class ObjectiveSchema(marshmallow.Schema):
    Team = marshmallow.fields.Str()
    Top = marshmallow.fields.Integer()
