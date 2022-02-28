from ast import Import
import os
from ssl import SSL_ERROR_SSL

import flask
import flask_cors




from . import db
from . import matchs
from . import seasonends
from . import rankings
from . import seasonends
from . import objectives
from . import calculs

from .entities.base import Base
from .entities.match import Match
from .entities.season_end import SeasonEnd
from .entities.ranking import Ranking
from .entities.objective import Objective
from .entities.cacul import Calcul

from .data import response 
from .data_ranking import team_all_rankings
from .data_end_season import final_ranking
from .data_objective import obj
from .data_calcul import calculations

def create_app(test_config=None):

    # creating the Flask application
    app = flask.Flask(__name__, instance_relative_config=True)
    flask_cors.CORS(app)

    # load configuration from config.py
    app.config.from_object('config')

    if test_config is None:
        # load the instance/config.py, if it exists, when not testing
        app.config.from_pyfile('config.py', silent=True)
    else:
        # load the test config if passed in
        app.config.from_mapping(test_config)

    # ensure the instance folder exists
    try:
        os.makedirs(app.instance_path)
    except OSError:
        pass

    # if needed, generate database schema
    with app.app_context():
        Base.metadata.create_all(db.get_engine())

    with app.app_context():
        session = db.get_session()  
        session.query(SeasonEnd).delete()
        session.query(Match).delete()
        session.query(Ranking).delete()
        session.query(Objective).delete()
        session.query(Calcul).delete()
        session.commit()
        for match in response["matches"]:
            new_match = Match(
                homeTeam = match["homeTeam"]["name"],
                awayTeam = match["awayTeam"]["name"],
                Season = match["season"]["endDate"] ,
                Date =  match["utcDate"][0:10],
                Status = match["status"],
                Winner = match["score"]["winner"],
                Goal_Away = match["score"]["fullTime"]["awayTeam"],
                Goal_Home = match["score"]["fullTime"]["homeTeam"],
                Time =  match["utcDate"][11:16]
            )
            session.add(new_match)
        session.commit()

        for rank,team in enumerate(final_ranking):
            new_team = SeasonEnd(Team=team["Team"],Points=team["Points"],Wins=team["Wins"],
            Draws=team["Draws"],Losses=team["Losses"],Goals_scored=team["Goals_scored"],
            Goals_conceded=team["Goals_conceded"],Rank=rank+1)

            session.add(new_team)
        session.commit()

        for all_rankings in team_all_rankings:
            new_all_rankings = Ranking(Team = all_rankings["Team"], One=all_rankings["One"] ,Two= all_rankings["Two"],Three=all_rankings["Three"], Four=all_rankings["Four"],
            Five = all_rankings["Five"], Six = all_rankings["Six"], Seven=all_rankings["Seven"], Eight=all_rankings["Eight"], Nine=all_rankings['Nine'],Ten=all_rankings['Ten'],
            Eleven=all_rankings["Eleven"], Twelve=all_rankings["Twelve"], Thirteen=all_rankings["Thirteen"], Fourteen=all_rankings["Fourteen"],
            Fifteen=all_rankings["Fifteen"], Sixteen=all_rankings["Sixteen"], Seventeen=all_rankings["Seventeen"], Eighteen=all_rankings["Eighteen"],
            Nineteen=all_rankings["Nineteen"], Twenty= all_rankings["Twenty"])
            session.add(new_all_rankings)
        session.commit()

        for team in obj:
            new_objectif = Objective(Team = team, Top= obj[team])
            session.add(new_objectif)
        session.commit()

        for team in calculations:
            new_calcul = Calcul(Team = team["Team"],Odds = team["Odds"], Importance=team["Importance"] )
            session.add(new_calcul)
        session.commit()
        session.close()


    app.register_blueprint(matchs.blueprint)
    app.register_blueprint(seasonends.blueprint)
    app.register_blueprint(rankings.blueprint)
    app.register_blueprint(objectives.blueprint)
    app.register_blueprint(calculs.blueprint)

    return app
