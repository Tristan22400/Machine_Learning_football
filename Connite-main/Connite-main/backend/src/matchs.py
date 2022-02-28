import flask

from .db import get_session
from .entities.match import Match, MatchSchema

# def semaine(date):
#     new_date = date[:8]+str(int(date[8:10])+10)
#     return new_date

a = '2022-02-19'



blueprint = flask.Blueprint('matchs', __name__)


@blueprint.route('/matchs/<date>',methods = ['Get'])

def get_today_matchs(date):
    session = get_session()
    match_objects = session.query(Match).filter(Match.Date == a)

    schema = MatchSchema(many=True)
    matchs = schema.dump(match_objects)

    session.close()
    return flask.jsonify(matchs)




