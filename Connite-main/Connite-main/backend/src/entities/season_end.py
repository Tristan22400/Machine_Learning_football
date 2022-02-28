import marshmallow
import sqlalchemy

from .base import Base


class SeasonEnd(Base):
    __tablename__ = 'season_end'
    Team = sqlalchemy.Column(sqlalchemy.String,primary_key=True)   
    Points = sqlalchemy.Column(sqlalchemy.Integer)
    Wins = sqlalchemy.Column(sqlalchemy.Integer)
    Draws = sqlalchemy.Column(sqlalchemy.Integer)
    Losses = sqlalchemy.Column(sqlalchemy.Integer)
    Goals_scored = sqlalchemy.Column(sqlalchemy.Integer)
    Goals_conceded = sqlalchemy.Column(sqlalchemy.Integer)
    Rank = sqlalchemy.Column(sqlalchemy.Integer)


    def __init__(self, Team, Points, Wins, Draws, Losses, Goals_scored, Goals_conceded, Rank):
        self.Team = Team
        self.Points = Points
        self.Wins = Wins
        self.Draws = Draws
        self.Losses = Losses
        self.Goals_scored = Goals_scored
        self.Goals_conceded = Goals_conceded
        self.Rank = Rank

    


class SeasonEndSchema(marshmallow.Schema):
    Team = marshmallow.fields.Str()
    Points = marshmallow.fields.Number()
    Wins = marshmallow.fields.Number()
    Draws = marshmallow.fields.Number()
    Losses = marshmallow.fields.Number()
    Goals_scored = marshmallow.fields.Number()
    Goals_conceded = marshmallow.fields.Number()
    Rank = marshmallow.fields.Number()


