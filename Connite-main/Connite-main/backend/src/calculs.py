import flask

from .db import get_session
from .entities.cacul import Calcul, CalculSchema

blueprint = flask.Blueprint('calculs', __name__)

@blueprint.route('/calculs/<team>',methods = ['Get'])
def getCalculs(team):
    session = get_session()
    calcul_objects = session.query(Calcul).filter_by(Team = team)
    schema = CalculSchema(many=True)
    calculs = schema.dump(calcul_objects)
    session.close()
    return flask.jsonify(calculs)

