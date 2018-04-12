from flask import Flask
from flask_restplus import Api, Resource
#import popular
from supporting_scripts.events_mapper import user_frequency
from supporting_scripts.similar_events import event_recommender
from flask_restplus import reqparse


app = Flask(__name__)
api = Api(app)

pagination = reqparse.RequestParser()
pagination.add_argument('identifier', type=str, required=False)


class UserFrequency(Resource):

    def get(self, user_name):
        user_freq = user_frequency(user_name)
        result_dict = {'past_frequency': user_freq[0],
                       'suggested_events_frequency': user_freq[1]}
        return {user_name: result_dict}


class EventRecommender(Resource):

    @api.expect(pagination)
    def get(self, event_id):
        args = pagination.parse_args()
        identifier = args['identifier'].strip().lower()
        if identifier != "cosine":
            identifier = None
        recommended_events = event_recommender(int(event_id), identifier=identifier)
        return {'suggested_events': recommended_events}


api.add_resource(UserFrequency, '/api/users/<user_name>', endpoint='user_fav_words')
api.add_resource(EventRecommender, '/api/events/<event_id>', endpoint='event_recommendations')


if __name__ == '__main__':
    app.run(debug=True)
