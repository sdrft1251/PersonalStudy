# -*- coding: utf-8 -*-

import json
from flask import Flask, request, make_response
from slacker import Slacker

from moment import moment_from_yahoo
from slacker import Slacker
from common import crawling_naver_finance

import win32com.client
import time

token = "SLACK_TOKEN"
slack = Slacker(token)
app = Flask(__name__)


def moment_information(stock_code):
    try:
        result_dict, fin_info_str = moment_from_yahoo.summary_data(stock_code=stock_code)

        result_str = ""
        for key, val in result_dict.items():
            result_str = result_str + str(key) + " : "
            result_str = result_str + str(val) + " / "
        result_str = result_str + "\n" + fin_info_str
        return result_str
    except:
        return "Failed..."

def cal_values(stock_code, chart_type="year", last_quarter="2019/12", market_growth = 0.0792):
    srim_values = crawling_naver_finance.s_rim_value(stock_code=stock_code, chart_type=chart_type, last_quarter=last_quarter, market_growth = market_growth)
    return srim_values



def event_handler(event_type, slack_event):
    channel = slack_event["event"]["channel"]

    string_slack_event = str(slack_event)
    if slack_event['event']['type'] == "app_mention":
        if string_slack_event.find("{'type': 'user', 'user_id': ") != -1:
            now_time = time.time()
            if int(float(now_time)) - int(float(slack_event['event']['ts'])) <=3:

                try:
                    user_query = slack_event['event']['blocks'][0]['elements'][0]['elements'][1]['text']
                    user_quert_split = user_query.split()
                    answer = ""

                    if user_quert_split[0] == "p":
                        answer = moment_information(stock_code=user_quert_split[1])
                    elif user_quert_split[0] == "v":
                        answer = cal_values(stock_code=user_quert_split[1], chart_type="year", last_quarter="2019/12", market_growth = 0.0792)
                    else :
                        answer = "Don't understand..."
                    
                    slack.chat.post_message(channel, answer)
                    return make_response("ok", 200,)
                except IndexError:
                    pass

    message = "[%s] cannot find event handler" % event_type
    return make_response(message, 200, {"X-Slack-No-Retry": 1})

@app.route('/', methods=['POST'])

def hello_there():
    slack_event = json.loads(request.data)

    if "challenge" in slack_event:
        return make_response(slack_event["challenge"], 200, {"content_type": "application/json"})

    if "event" in slack_event:
        event_type = slack_event["event"]["type"]
        return event_handler(event_type, slack_event)

    return make_response("There are no slack request events", 404, {"X-Slack-No-Retry": 1})





if __name__ == '__main__':

    app.run(debug=True)
