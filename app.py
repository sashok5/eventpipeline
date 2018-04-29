from flask import Flask
from flask_restplus import Api, Resource
from flask_restplus import reqparse
import Recommender as R

app = Flask(__name__)
api = Api(app)

pagination = reqparse.RequestParser()
pagination.add_argument('identifier', type=str, required=False)

db = 'postgresql://admin:admin@localhost:5432/groupup'

'''

class UserFrequency(Resource):

    def get(self, user_name):
        user_freq = user_frequency(user_name)
        result_dict = {'past_frequency': user_freq[0],
                       'suggested_events_frequency': user_freq[1]}
        return {user_name: result_dict}


@api.doc(params={'event_id': "event_id"})
class EventRecommender(Resource):

    @api.expect(pagination)
    def get(self, event_id):
        args = pagination.parse_args()
        identifier = args['identifier']
        if identifier is not None:
            identifier = args['identifier'].strip().lower()
            if identifier != "cosine":
                identifier = None
        recommended_events = event_recommender(int(event_id), identifier=identifier)
        return recommended_events
'''


class FillPopular(Resource):

    def get(self):
        recommender = R.Recommender(db)
        recommender.generate_popular()
        return {'popular': 'OK'}


class FillRecommended(Resource):

    def get(self, user_id):
        recommender = R.Recommender(db)
        recommender.recommend_events(user_id)
        return {'recommended': 'OK'}

class FillTopics(Resource):
    def get(self):
        recommender = R.Recommender(db)
        recommender.generate_topics()

class KeywordSearch(Resource):
    def get(self, search_string):
        recommender = R.Recommender(db)
        res = recommender.search_keyword(search_string)
        return {'result': res}


# api.add_resource(UserFrequency, '/api/users/<user_name>', endpoint='user_fav_words')
# api.add_resource(EventRecommender, '/api/events/<event_id>', endpoint='event_recommendations')
api.add_resource(FillPopular, '/api/events/popular', endpoint='generate_popular')
api.add_resource(FillRecommended, '/api/users/<user_id>', endpoint='generate_recommended')
api.add_resource(FillTopics, '/api/topics/', endpoint='generate_topics')
api.add_resource(KeywordSearch, '/api/events/search/<search_string>', endpoint='keywords_search')


if __name__ == '__main__':
    app.run(debug=True)
