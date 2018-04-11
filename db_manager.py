from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine, MetaData, Table, func
from sqlalchemy import Column, Integer, String
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import scoped_session, sessionmaker
from sqlalchemy.orm import synonym

Base = declarative_base()
DBSession = scoped_session(sessionmaker())
metadata = MetaData()
engine = create_engine('postgresql://admin:admin@localhost:5432/groupup', echo=False)


class Popular(Base):
    __tablename__ = 'popular'
    event_id = Column(Integer, primary_key=True)
    num_of_attending = Column(Integer)
    num_of_clicks = Column(Integer)


def init_sqlalchemy():
    DBSession.remove()
    DBSession.configure(bind=engine, autoflush=False, expire_on_commit=False)
    Base.metadata.create_all(engine)


# Create events table
Events = Table('events', Base.metadata, autoload=True, autoload_with=engine)

# Create users table
Users = Table('users', Base.metadata, autoload=True, autoload_with=engine)

# Create attendances table
Attendances = Table('attendances', Base.metadata, autoload=True, autoload_with=engine)

# Create relationships such as who is following whom
Relationships = Table('relationships', Base.metadata, autoload=True, autoload_with=engine)

# Create Tags, such as event categories and also user interests
Tags = Table('tags', Base.metadata, autoload=True, autoload_with=engine)

# Create EventViews, tracking which users viewed events
EventViews = Table('event_views', Base.metadata, autoload=True, autoload_with=engine)
