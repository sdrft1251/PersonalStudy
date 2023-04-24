import os
import pyedflib
from io import BytesIO
import matplotlib.pyplot as plt
from scipy.signal import lfilter, sosfilt
from scipy.signal import butter, iirnotch, lfilter
from viewer.config import Config

from viewer import app
from flask import send_file

def make_tree(path):
    tree = dict(name=path, children=[])
    try: lst = os.listdir(path)
    except OSError:
        pass #ignore errors
    else:
        for name in lst:
            fn = os.path.join(path, name)
            if os.path.isdir(fn):
                tree['children'].append(make_tree(fn))
            else:
                tree['children'].append(dict(name=fn))
    return tree

@app.route("/")
def home():
    tree = make_tree(Config.ECG_FILE_PATH)
    return tree

def data_load(path):
    f = pyedflib.EdfReader(path)
    sigbufs = f.readSignal(0)
    f._close()
    print(sigbufs)
    return sigbufs

@app.route("/test")
def read_data():
    edf_path = Config.TEST_FILE_PATH
    print(edf_path)
    f = pyedflib.EdfReader(edf_path)
    sigbufs = f.readSignal(0)
    f._close()
    print(sigbufs)
    return "success"

def ploting_data(signal):
    fig = plt.figure(figsize=(15, 2))
    plt.title("Test ECG")
    plt.plot(signal, 'k--', label='ecg')

    img = BytesIO()
    plt.savefig(img, format='png', dpi=200)
    img.seek(0)
    return img

@app.route("/imagetest")
def plot_data():
    edf_path = Config.TEST_FILE_PATH
    sigbufs = data_load(edf_path)
    sigbufs = sigbufs[1000:2000]

    fig = plt.figure(figsize=(15, 2))
    plt.title("Test ECG")
    plt.plot(sigbufs, 'k--', label='ecg')
    img = BytesIO()
    plt.savefig(img, format='png', dpi=200)
    img.seek(0)
    return send_file(img, mimetype='image/png')


######################## High & Low pass filter ########################
## A high pass filter allows frequencies higher than a cut-off value
def butter_highpass(cutoff, fs, order=5):
    sos = butter(order, cutoff, 'hp', fs=fs, output='sos')
    return sos

## A low pass filter allows frequencies lower than a cut-off value
def butter_lowpass(cutoff, fs, order=5):
    sos = butter(order, cutoff, 'lp', fs=fs, output='sos')
    return sos

def final_filter(data, fs, order=5):
    highpass_sos = butter_highpass(0.5, fs, order=order)
    x = sosfilt(highpass_sos, data)
    lowpass_sos = butter_highpass(2.5, fs, order=order)
    y = sosfilt(lowpass_sos, x)
    return y


@app.route("/preprocessingtest")
def preprocess_data():
    edf_path = Config.TEST_FILE_PATH
    sigbufs = data_load(edf_path)
    ecg = final_filter(sigbufs, fs=256, order=5)
    ecg = ecg[1000:2000]
    img = ploting_data(ecg)
    return send_file(img, mimetype='image/png')



############### DB Sample code ##############
from flask import Flask

from config import config

db = None

def db_init(db_uri:str):
    import sqlite3
    global db
    db = sqlite3.connect(db_uri, check_same_thread=False)
    db.row_factory = sqlite3.Row

def create_app(config_name:str) -> Flask:
    app = Flask(__name__)
    app.config.from_object(config[config_name])
    config[config_name].init_app(app)

    # DB 추가 시 사용
    # db_init(app.config['DATABASE_URI'])

    from .main import main as main_blueprint
    app.register_blueprint(main_blueprint)

    return app


#################### For DB #############################
from sqlalchemy import create_engine, ForeignKey, Column, Integer, String, Boolean
engine = create_engine('sqlite:///test.db', echo = True)
from sqlalchemy.ext.declarative import declarative_base
Base = declarative_base()
from sqlalchemy.orm import relationship

class EcgTestSchema(Base):
    __tablename__ = 'ecgtestschema'
    id = Column(Integer, primary_key = True)
    region = Column(String)
    test_id = Column(String)
    duration = Column(Integer)
    is_sample = Column(Boolean)
    is_normal = Column(Boolean)
    edf_path = Column(String)
    details_path = Column(String)

    datagroup_id = relationship('DataGroup', secondary = 'link')

   
class DataGroup(Base):
    __tablename__ = 'datagroup'
    id = Column(Integer, primary_key = True)
    group_type = Column(String)
    group_status = Column(String)
    item_count = Column(Integer)

    ecgtest_id = relationship(EcgTestSchema, secondary='link')


class Link(Base):
    __tablename__ = 'link'
    ecgtest_id = Column(Integer, ForeignKey('ecgtestschema.id'), primary_key = True)
    datagroup_id = Column(Integer, ForeignKey('datagroup.id'), primary_key = True)

Base.metadata.create_all(engine)

t1 = EcgTestSchema(region="AU", test_id="a01", duration=24, is_sample=False, is_normal=True, edf_path="/home/test/1", details_path="/home/test/1")
t2 = EcgTestSchema(region="AU", test_id="a02", duration=48, is_sample=False, is_normal=True, edf_path="/home/test/2", details_path="/home/test/2")
t3 = EcgTestSchema(region="AU", test_id="a03", duration=72, is_sample=False, is_normal=True, edf_path="/home/test/3", details_path="/home/test/3")

g1 = DataGroup(group_type="pre_processed", group_status="default", item_count=1)
g2 = DataGroup(group_type="group1", group_status="default", item_count=1)
g3 = DataGroup(group_type="group2", group_status="default", item_count=1)

t1.datagroup_id.append(g1)
t1.datagroup_id.append(g2)
t2.datagroup_id.append(g2)
t2.datagroup_id.append(g3)
t3.datagroup_id.append(g1)
t3.datagroup_id.append(g2)


from sqlalchemy.orm import sessionmaker
Session = sessionmaker(bind = engine)
session = Session()
session.add(t1)
session.add(t2)
session.add(t3)
session.add(g1)
session.add(g2)
session.add(g3)
session.commit()

from sqlalchemy.orm import sessionmaker
Session = sessionmaker(bind = engine)
session = Session()

print("==============================")
for x in session.query( EcgTestSchema, DataGroup).filter(Link.ecgtest_id == EcgTestSchema.id, 
   Link.datagroup_id == DataGroup.id).order_by(Link.ecgtest_id).all():
   print ("ECG_TEST_ID: {} GROUP: {}".format(x.EcgTestSchema.test_id, x.DataGroup.group_type))

print("==============================")
for x in session.query(EcgTestSchema):
    print(x.id, x.test_id)

print("==============================")
for x in session.query(DataGroup):
    print(x.id, x.group_type)

print("==============================")
for x in session.query(Link):
    print(x.ecgtest_id, x.datagroup_id)

print("==============================")
for x in session.query(EcgTestSchema):
    print(x.id, x.test_id, x.datagroup_id)