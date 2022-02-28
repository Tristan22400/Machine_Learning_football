import flask

from .db import get_session
from .entities.objective import Objective, ObjectiveSchema

blueprint = flask.Blueprint('objectives', __name__)

@blueprint.route('/objectives/<team>',methods = ['Get'])
def getObjectives(team):
    session = get_session()
    obj_objects = session.query(Objective).filter_by(Team = team)
    schema = ObjectiveSchema(many=True)
    objs = schema.dump(obj_objects)
    session.close()
    return flask.jsonify(objs)

