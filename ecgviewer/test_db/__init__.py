from flask import Flask
from flask_sqlalchemy import SQLAlchemy
import os
basedir = os.path.abspath(os.path.dirname(__name__))
class DevConfig:
    DATABASE_URI = os.environ.get('DEV_DATABASE_URL') or 'sqlite://' + os.path.join(basedir, 'sqlite.db')
    EDFFILE_PATH = os.environ.get('DEV_DATABASE_URL') or '/home/jaemincho/works/Data/wellysis/AU'
    SQLLITE3_DATABASE_URI = os.environ.get('SQLLITE3_DATABASE_URI') or 'sqlite:///' + os.path.join(basedir, 'test.db')
    #SQLLITE3_DATABASE_URI = os.environ.get('SQLLITE3_DATABASE_URI') or os.path.join(basedir, 'test.db')

app = Flask(__name__)
app.config.from_object(DevConfig)
db = SQLAlchemy(app)

class TestLink(db.Model):
    __tablename__ = "testlink"
    ecgtest_id = db.Column(db.Integer, db.ForeignKey('ecgtest.id'), primary_key = True)
    testgroup_id = db.Column(db.Integer, db.ForeignKey('testgroup.id'), primary_key = True)


class SampleLink(db.Model):
    __tablename__ = "samplelink"
    ecgtest_id = db.Column(db.Integer, db.ForeignKey('ecgtest.id'), primary_key = True)
    samplegroup_id = db.Column(db.Integer, db.ForeignKey('samplegroup.id'), primary_key = True)
    page = db.Column(db.Integer, primary_key = True)

    ecgtest = db.relationship("EcgTest", back_populates="samples")
    samplegroup = db.relationship("SampleGroup", back_populates="ecgtests")


class EcgTest(db.Model):
    __tablename__ = "ecgtest"
    id = db.Column(db.Integer, primary_key=True)
    region = db.Column(db.String)   # KR | AU | UK | SG
    test_id = db.Column(db.String)
    duration = db.Column(db.String)   # 24 | 48 | 72
    condition = db.Column(db.String)   # Normal | Abnormal | Unknown
    edf_path = db.Column(db.String)
    details_path = db.Column(db.String)

    testgroup_id = db.relationship('TestGroup', secondary = 'testlink')
    samples = db.relationship('SampleLink', back_populates='ecgtest')


class TestGroup(db.Model):
    __tablename__ = "testgroup"
    id = db.Column(db.Integer, primary_key = True)
    group_name = db.Column(db.String)
    group_status = db.Column(db.String)   # open | close

    ecgtest_id = db.relationship('EcgTest', secondary='testlink')


class SampleGroup(db.Model):
    __tablename__ = "samplegroup"
    id = db.Column(db.Integer, primary_key=True)
    group_name = db.Column(db.String)
    group_status = db.Column(db.String)   # open | close

    ecgtests = db.relationship('SampleLink', back_populates='samplegroup')


class PreprocessGroup(db.Model):
    __tablename__ = "preprocessgroup"
    id = db.Column(db.Integer, primary_key=True)
    group_name = db.Column(db.String)
    group_status = db.Column(db.String)   # open | close
    path = db.Column(db.String)
