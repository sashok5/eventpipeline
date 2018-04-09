from flask import Flask, request
from flask_restplus import Api, Resource
import popular
from supporting_scripts.events_mapper import user_frequency
from supporting_scripts.similar_events import event_recommendor
import json

app = Flask(__name__)
api = Api(app)


class UserFrequency(Resource):

    def get(self, user_name):
        user_freq = user_frequency(user_name)
        if user_freq is not None:
            result_dict = {'past_frequency': user_freq[0],
                           'suggested_events_frequency': user_freq[1]}
            return {user_name: result_dict}
        else:
            return {'error': 'User not found, Please check'}


class EventRecommendor(Resource):

    def get(self, event_id):
        recommended_events = event_recommendor(int(event_id))
        return {'suggested_events': recommended_events}


api.add_resource(UserFrequency, '/api/users/<user_name>', endpoint='user_fav_words')
api.add_resource(EventRecommendor, '/api/events/<event_id>', endpoint='event_recommendations')

if __name__ == '__main__':
    app.run(debug=True)
