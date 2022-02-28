import marshmallow
import sqlalchemy

from .base import Base


class Match(Base):
    __tablename__ = 'matchs'
    id =sqlalchemy.Column(sqlalchemy.Integer, primary_key=True)
    homeTeam = sqlalchemy.Column(sqlalchemy.String) 
    awayTeam = sqlalchemy.Column(sqlalchemy.String)
    Season = sqlalchemy.Column(sqlalchemy.String)
    Date = sqlalchemy.Column(sqlalchemy.String)
    Status = sqlalchemy.Column(sqlalchemy.String)
    Winner = sqlalchemy.Column(sqlalchemy.String)
    Goal_Away = sqlalchemy.Column(sqlalchemy.Integer)
    Goal_Home = sqlalchemy.Column(sqlalchemy.Integer)
    Time = sqlalchemy.Column(sqlalchemy.String)


    def __init__(self, homeTeam, awayTeam, Season, Date, Status, Winner, Goal_Away, Goal_Home, Time):
        self.homeTeam = homeTeam
        self.awayTeam = awayTeam
        self.Season = Season
        self.Date = Date
        self.Status = Status
        self.Winner = Winner
        self.Goal_Away = Goal_Away
        self.Goal_Home = Goal_Home
        self.Time = Time
    


class MatchSchema(marshmallow.Schema):
    id = marshmallow.fields.Number()
    homeTeam = marshmallow.fields.Str()
    awayTeam = marshmallow.fields.Str()
    Season = marshmallow.fields.Str()
    Date = marshmallow.fields.Str()
    Status = marshmallow.fields.Str()
    Winner = marshmallow.fields.Str()
    Goal_Away = marshmallow.fields.Number()
    Goal_Home = marshmallow.fields.Number()
    Time = marshmallow.fields.Str()
