from flask import Blueprint

dbtest = Blueprint('dbtest', __name__)

from . import views