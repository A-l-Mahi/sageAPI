from flask import Flask, jsonify, request
import os

from utils import Download, pipeline

app = Flask(__name__)

BLOB_NAME = "MasakhaNEWS_News_Topic_Classification_for_African_.pdf"


@app.route('/executePrompt', methods = ['GET'])

def download_blob():

    param1 = request.args.get('file_name', '')
    param2 = request.args.get('prompt', '')


    data = Download.get_blob(param1)

    response = pipeline.build_prompt(param2)

    data  = {
        'response':response,
        'status':"success"
    }

    return jsonify(data)


if __name__ == '__main__':
    app.run(debug=True)

