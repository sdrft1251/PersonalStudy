import os

basedir = os.path.abspath(os.path.dirname(__name__))

class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'hard to guess string'

    @staticmethod
    def init_app(app):
        pass

class DevConfig(Config):
    DATABASE_URI = os.environ.get('DEV_DATABASE_URL')
    EDFFILE_PATH = os.environ.get('DEV_DATABASE_URL')
    SQLALCHEMY_DATABASE_URI = os.environ.get('SQLALCHEMY_DATABASE_URI')

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