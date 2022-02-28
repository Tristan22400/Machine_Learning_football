import marshmallow
import sqlalchemy

from .base import Base


class Ranking(Base):
    __tablename__ = 'ranking'
    Team = sqlalchemy.Column(sqlalchemy.String,primary_key=True)   
    One = sqlalchemy.Column(sqlalchemy.Float)
    Two = sqlalchemy.Column(sqlalchemy.Float)
    Three = sqlalchemy.Column(sqlalchemy.Float)
    Four = sqlalchemy.Column(sqlalchemy.Float)
    Five = sqlalchemy.Column(sqlalchemy.Float)
    Six = sqlalchemy.Column(sqlalchemy.Float)
    Seven = sqlalchemy.Column(sqlalchemy.Float)
    Eight = sqlalchemy.Column(sqlalchemy.Float)
    Nine = sqlalchemy.Column(sqlalchemy.Float)
    Ten = sqlalchemy.Column(sqlalchemy.Float)
    Eleven = sqlalchemy.Column(sqlalchemy.Float)
    Twelve = sqlalchemy.Column(sqlalchemy.Float)
    Thirteen = sqlalchemy.Column(sqlalchemy.Float)
    Fourteen = sqlalchemy.Column(sqlalchemy.Float)
    Fifteen = sqlalchemy.Column(sqlalchemy.Float)
    Sixteen = sqlalchemy.Column(sqlalchemy.Float)
    Seventeen = sqlalchemy.Column(sqlalchemy.Float)
    Eighteen = sqlalchemy.Column(sqlalchemy.Float)
    Nineteen = sqlalchemy.Column(sqlalchemy.Float)
    Twenty = sqlalchemy.Column(sqlalchemy.Float)



    def __init__(self,Team, One,Two,Three,Four,Five,Six,Seven,Eight,Nine,Ten,Eleven,Twelve,Thirteen,Fourteen,Fifteen,Sixteen,Seventeen,Eighteen,Nineteen,Twenty):
        self.Team = Team
        self.One = One
        self.Two = Two
        self.Three = Three
        self.Four = Four
        self.Five = Five
        self.Six = Six
        self.Seven = Seven
        self.Eight = Eight
        self.Nine = Nine
        self.Ten = Ten
        self.Eleven = Eleven
        self.Twelve = Twelve
        self.Thirteen = Thirteen
        self.Fourteen = Fourteen
        self.Fifteen = Fifteen
        self.Sixteen = Sixteen
        self.Seventeen = Seventeen
        self.Eighteen = Eighteen
        self.Nineteen = Nineteen
        self.Twenty = Twenty

    


class RankingSchema(marshmallow.Schema):
    Team = marshmallow.fields.Str()
    One = marshmallow.fields.Number()
    Two = marshmallow.fields.Number()
    Three = marshmallow.fields.Number()
    Four = marshmallow.fields.Number()
    Five = marshmallow.fields.Number()
    Six = marshmallow.fields.Number()
    Seven = marshmallow.fields.Number()
    Eight = marshmallow.fields.Number()
    Nine = marshmallow.fields.Number()
    Ten = marshmallow.fields.Number()
    Eleven = marshmallow.fields.Number()
    Twelve = marshmallow.fields.Number()
    Thirteen = marshmallow.fields.Number()
    Fourteen = marshmallow.fields.Number()
    Fifteen = marshmallow.fields.Number()
    Sixteen = marshmallow.fields.Number()
    Seventeen = marshmallow.fields.Number()
    Eighteen = marshmallow.fields.Number()
    Nineteen = marshmallow.fields.Number()
    Twenty = marshmallow.fields.Number()


