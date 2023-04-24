from flask import Flask
from flask import jsonify, request
import json
import os

global idx
idx=0

def create_app():
    app = Flask(__name__)

    @app.route('/invocations', methods=['POST'])
    def data_receive():
        datas = request.json.get("inputs")
        global idx

        with open(f"/home/wellysis-tft/Desktop/code/CES2022/data/from_ex/data_{idx}.json", "w") as write_file:
                json.dump({'inputs': datas}, write_file)
        idx += 1
        print(f"Input data is : {datas} | IDX : {idx}")
        return jsonify({ 'normal': 1.0, 'abnormal': 0.0, 'result': 'normal'})

    return app


if __name__ == "__main__":
    app = create_app()
    app.run(debug=True, host="0.0.0.0", port="47202")