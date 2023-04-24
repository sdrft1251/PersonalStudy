import json

from flask import Flask, request, make_response
from config import Config
from app.event_handler import event_handler

token = Config.SLACK_TOKEN

def create_app() -> Flask:
    app = Flask(__name__)
    app.config.from_object(Config)
    Config.init_app(app)

    @app.route("/slack", methods=["GET", "POST"])
    def hears():
        slack_event = json.loads(request.data)
        if "challenge" in slack_event:
            return make_response(slack_event["challenge"], 200, {"content_type": "application/json"})
        if "event" in slack_event:
            event_type = slack_event["event"]["type"]
            return event_handler(event_type, slack_event, token)
        return make_response("슬랙 요청에 이벤트가 없습니다.", 404, {"X-Slack-No-Retry": 1})

    return app




