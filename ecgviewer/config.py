import os

basedir = os.path.abspath(os.path.dirname(__name__))

class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'hard to guess string'

    @staticmethod
    def init_app(app):
        pass

class DevConfig(Config):
    DATABASE_URI = os.environ.get('DEV_DATABASE_URL') or 'sqlite://' + os.path.join(basedir, 'sqlite.db')
    EDFFILE_PATH = os.environ.get('DEV_DATABASE_URL') or '/home/jaemincho/works/Data/wellysis/AU'
    #SQLLITE3_DATABASE_URI = os.environ.get('SQLLITE3_DATABASE_URI') or 'sqlite:///' + os.path.join(basedir, 'test.db')
    SQLLITE3_DATABASE_URI = os.environ.get('SQLLITE3_DATABASE_URI') or os.path.join(basedir, 'test.db')

class TestConfig(Config):
    TESTING = True
    DATABASE_URI = os.environ.get('DEV_DATABASE_URL') or \
        'sqlite://' + os.path.join(basedir, 'sqlite.db')

class ProdConfig(Config):
    pass


config = {
    'development': DevConfig,
    'testing': TestConfig,
    'production': ProdConfig,

    'default': DevConfig
}