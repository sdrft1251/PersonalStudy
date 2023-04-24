import requests
from flask import make_response

def event_handler(event_type, slack_event, token):
    if event_type == "app_mention":
        channel = slack_event["event"]["channel"]
        text_data = slack_event["event"]["text"]
        text = text_handler(text_data)
        post_message(token,channel,text)
        return make_response("앱 멘션 메시지가 보내졌습니다.", 200, )
    message = "[%s] 이벤트 핸들러를 찾을 수 없습니다." % event_type
    return make_response(message, 200, {"X-Slack-No-Retry": 1})

def post_message(token, channel, text):
    response = requests.post("https://slack.com/api/chat.postMessage",
        headers={"Authorization": "Bearer "+token},
        data={"channel": channel,"text": text}
    )
    print(response)

def text_handler(text_data):
    if text_data.split(" ")[-1].lower() == "hi":
        return "Hellow~ ^^"
    elif text_data.split(" ")[1].lower() == "stock":
        return "Stock!"
    else:
        return "Sorry I can't understand now..."