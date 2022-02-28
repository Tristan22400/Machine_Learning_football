import marshmallow
import sqlalchemy

from .base import Base


class Calcul(Base):
    __tablename__ = 'calculs'
    Team =sqlalchemy.Column(sqlalchemy.String, primary_key=True)
    Odds = sqlalchemy.Column(sqlalchemy.Integer) 
    Importance = sqlalchemy.Column(sqlalchemy.Integer)



    def __init__(self, Team, Odds, Importance):
        self.Team = Team
        self.Odds = Odds
        self.Importance = Importance


class CalculSchema(marshmallow.Schema):
    Team = marshmallow.fields.Str()
    Odds = marshmallow.fields.Number()
    Importance = marshmallow.fields.Number()
