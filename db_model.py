from sqlalchemy.orm import sessionmaker, scoped_session, relation, backref
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Table, ForeignKey, Column, func
from sqlalchemy.types import VARCHAR, Integer, BigInteger, Date, Time, Float, TIMESTAMP

BaseModel = declarative_base()
metadata = BaseModel.metadata


# engine = create_engine('postgresql://admin:admin@localhost:5432/groupup', echo=False)


def make_session(engine):
    maker = sessionmaker(autoflush=True, autocommit=False, bind=engine)
    session = scoped_session(maker)
    return session


class User(BaseModel):
    """
    User table.

    """
    __tablename__ = "users"

    user_id = Column(BigInteger, primary_key=True)

    name = Column(VARCHAR)

    email = Column(VARCHAR)

    password_digest = Column(VARCHAR)

    city = Column(VARCHAR)

    state = Column(VARCHAR)

    created_at = Column(TIMESTAMP)

    updated_at = Column(TIMESTAMP)

    def __init__(self, user_id, name, email, city, state, created):
        self.user_id = user_id
        self.name = name
        self.email = email
        self.city = city
        self.state = state
        self.created_at = created
        self.updated_at = created
        self.password_digest = ''

    @classmethod
    def create_user(cls, db_session, user_id, name, email, city, state, created):
        """
        Create a user account for ``user_id`` if it doesn't exist yet.

        """
        user = db_session.query(cls).get(user_id)
        if user is None:
            user = cls(user_id=user_id, name=name, email=email, city=city, state=state, created=created
                       )
            user.password_digest = '$2a$10$Cz58MmQLYjIXeecXTdP8NO8ZuKldiUKbOxWUIlbKweazCg4EFwUtq'
            db_session.add(user)

    @classmethod
    def user_exists(cls, db_session, user_id):
        user = db_session.query(cls).get(user_id)
        if user is None:
            return False
        else:
            return True

    attending = []

    @classmethod
    def events_attending(cls, db_session, user_id):
        res = db_session.query(Attendances.event_id).filter(Attendances.user_id == user_id).all()
        attending = [r for r in res]
        return attending

    @classmethod
    def following_user_events(cls, db_session, user_id):
        res = db_session.query(Event.event_id, func.count(Event.event_id).label('event_count')). \
            join(Attendances, Attendances.event_id == Event.event_id). \
            join(Relationship, Attendances.user_id == Relationship.followed_id) \
            .filter(Relationship.follower_id == user_id, Event.event_date >= func.now()). \
            group_by(Event.event_id).order_by('event_count DESC')
        return res


class Relationship(BaseModel):
    __tablename__ = "relationships"
    id = Column(BigInteger, primary_key=True)
    follower_id = Column(BigInteger)
    followed_id = Column(BigInteger)
    created_at = Column(TIMESTAMP)
    updated_at = Column(TIMESTAMP)


class UserInGroup(BaseModel):
    __tablename__ = "users_in_groups"

    id = Column(BigInteger, primary_key=True)

    user_id = Column(BigInteger)

    group_id = Column(BigInteger)

    def __init__(self, user_id, group_id):
        self.user_id = user_id
        self.group_id = group_id


class UserTopic(BaseModel):
    __tablename__ = "user_topics"

    user_id = Column(BigInteger, primary_key=True)
    topic_id = Column(BigInteger, primary_key=True)
    topic_count = Column(Integer)
    true_score = Column(Float)
    predicted_score = Column(Float)

    def __init__(self, user_id, topic_id, topic_count):
        self.user_id = user_id
        self.topic_id = topic_id
        self.topic_count = topic_count

    @classmethod
    def update_predicted_score(cls, session, user_id, topic_id, predicted_score):
        score = session.query(UserTopic).filter(UserTopic.user_id == user_id, UserTopic.topic_id == topic_id).first()
        if score is None:
            score = UserTopic(user_id, topic_id, 0)
            score.true_score = 0
            score.predicted_score = predicted_score
            session.add(score)
        else:
            score.predicted_score = predicted_score
        session.commit()


class Event(BaseModel):
    __tablename__ = "events"

    event_id = Column(BigInteger, primary_key=True)

    event_host = Column(VARCHAR)

    title = Column(VARCHAR)

    desc = Column(VARCHAR)

    addr = Column(VARCHAR)

    street = Column(VARCHAR)

    city = Column(VARCHAR)

    state = Column(VARCHAR)

    zip = Column(VARCHAR)

    created_at = Column(TIMESTAMP)

    updated_at = Column(TIMESTAMP)

    event_date = Column(Date)

    event_start_time = Column(Time)

    category_id = Column(Integer)

    group_id = Column(Integer)

    rsvp_count = Column(Integer)

    created_by_user_id = Column(Integer)

    meetup_id = Column(VARCHAR)

    def __init__(self, title, desc, addr, host, city, state, zip, created, event_date, category_id, group_id,
                 rsvp_count, meetup_id):
        self.title = title
        self.desc = desc
        self.addr = addr
        self.event_host = host
        self.city = city
        self.state = state
        self.zip = zip
        self.created_at = created
        self.event_date = event_date
        self.category_id = category_id
        self.group_id = group_id
        self.rsvp_count = rsvp_count
        self.event_start_time = '18:00'
        self.created_by_user_id = 1
        self.updated_at = created
        self.meetup_id = meetup_id

    @classmethod
    def users_attending(cls, db_session, event_id):
        attending = db_session.query(Attendances.event_id).filter(Attendances.user_id == user_id).all()
        result = [r for r in attending]
        return result


class EventGroup(BaseModel):
    __tablename__ = "event_groups"

    group_id = Column(BigInteger, primary_key=True)

    category_id = Column(BigInteger)

    city = Column(VARCHAR)

    country = Column(VARCHAR)

    created = Column(TIMESTAMP)

    description = Column(VARCHAR)

    members = Column(Integer)

    group_name = Column(VARCHAR)

    state = Column(VARCHAR)

    def __init__(self, group_id, category_id, city, state, country, created, group_name, description, members):
        self.group_id = group_id
        self.category_id = category_id
        self.city = city
        self.country = country
        self.state = state
        self.created = created
        self.group_name = group_name
        self.description = description
        self.members = members


class EventCategory(BaseModel):
    __tablename__ = "event_categories"

    category_id = Column(BigInteger, primary_key=True)

    name = Column(VARCHAR)

    shortname = Column(VARCHAR)

    created_at = Column(TIMESTAMP)

    updated_at = Column(TIMESTAMP)

    def __init__(self, category_id, name, shortname):
        self.category_id = category_id
        self.name = name
        self.shortname = shortname
        self.created_at = func.now()
        self.updated_at = self.created_at


class Attendances(BaseModel):
    __tablename__ = "attendances"

    user_id = Column(BigInteger, primary_key=True)

    event_id = Column(BigInteger, primary_key=True)

    RSVP_Status = Column(Integer)

    created_at = Column(TIMESTAMP)

    updated_at = Column(TIMESTAMP)

    def __init__(self, user_id, event_id, rsvp):
        self.user_id = user_id
        self.event_id = event_id
        self.RSVP_Status = rsvp
        self.created_at = func.now()
        self.updated_at = self.created_at

    @classmethod
    def generate_attendances(cls, db_session, event_id, group_id, rsvp_count):
        """
        Generates attendances based on the event rsvp count, group and top members of those groups

        """
        users = db_session.query(User.user_id).join(UserInGroup, UserInGroup.user_id == User.user_id).filter(
            UserInGroup.group_id == group_id).limit(rsvp_count).all()
        for row in users:
            attend = Attendances(row.user_id, event_id, 3)
            db_session.add(attend)


class Popular(BaseModel):
    __tablename__ = 'popular'

    event_id = Column(BigInteger, primary_key=True)

    num_of_attending = Column(Integer)

    num_of_clicks = Column(Integer)

    def __init__(self, event_id, num1, num2):
        self.event_id = event_id
        self.num_of_attending = num1
        self.num_of_clicks = num2

    @classmethod
    def get_popular_events(cls, db_session):
        event_attendances = db_session.query(Event.event_id,
                                             func.count(Attendances.event_id).label('attendance_count')) \
            .filter(Event.event_date >= func.now()) \
            .outerjoin(Attendances, Event.event_id == Attendances.event_id) \
            .group_by(Event.event_id).limit(20)

        result = []

        for i, event in enumerate(event_attendances):
            event_click = db_session.query(func.count(EventView.event_id).label('click_count')).filter(
                EventView.event_id == event.event_id).group_by(EventView.event_id).first()
            result.append((event, event_click))

        return result


class Recommended(BaseModel):
    __tablename__ = 'recommended'

    id = Column(BigInteger, primary_key=True)

    event_id = Column(BigInteger)

    user_id = Column(BigInteger)

    rank = Column(Integer)

    type = Column(Integer)

    def __init__(self, event_id, user_id, rank, type):
        self.event_id = event_id
        self.user_id = user_id
        self.rank = rank
        self.type = type


class EventSimilarity(BaseModel):
    __tablename__ = "event_similarities"

    id = Column(BigInteger, primary_key=True)

    event_id_1 = Column(BigInteger)

    event_id_2 = Column(BigInteger)

    similarity = Column(Float)

    def __init__(self, event_id_1, event_id_2, similarity):
        self.event_id_1 = event_id_1
        self.event_id_2 = event_id_2
        self.similarity = similarity


class EventTopic(BaseModel):
    __tablename__ = "event_topics"

    topic_id = Column(BigInteger, primary_key=True)

    event_id = Column(BigInteger, primary_key=True)

    score = Column(Float)

    def __init__(self, topic_id, event_id, score):
        self.topic_id = topic_id
        self.event_id = event_id
        self.score = score


class EventView(BaseModel):
    __tablename__ = "event_views"
    id = Column(BigInteger, primary_key=True)
    event_id = Column(BigInteger)
    user_id = Column(BigInteger)
    created_at = Column(TIMESTAMP)
    updated_at = Column(TIMESTAMP)


class Topic(BaseModel):
    __tablename__ = "topics"

    topic_id = Column(BigInteger, primary_key=True)

    topic_string = Column(VARCHAR)

    def __init__(self, topic_id, topic_string):
        self.topic_id = topic_id
        self.topic_string = topic_string


class TopicTerm(BaseModel):
    __tablename__ = "topic_terms"

    topic_id = Column(BigInteger, primary_key=True)
    term = Column(VARCHAR, primary_key=True)
    score = Column(Float)

    def __init__(self, topic_id, term, score):
        self.topic_id = topic_id
        self.term = term
        self.score = score


class UserTerm(BaseModel):
    __tablename__ = "user_terms"
    id = Column(BigInteger, primary_key=True)
    term = Column(VARCHAR)
    score = Column(Float)
