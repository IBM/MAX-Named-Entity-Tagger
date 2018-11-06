import os
import logging
from flask import Flask
from api import api
from flask_cors import CORS

app = Flask(__name__)
# load default config
app.config.from_object('config')
# load override config if exists
if 'APP_CONFIG' in os.environ:
    app.config.from_envvar('APP_CONFIG')
api.init_app(app)
# enable CORS if flag is set
if os.getenv('CORS_ENABLE') == 'true':
    CORS(app, origins='*')
    app.logger.setLevel(logging.INFO)
    app.logger.info('MAX Model Server is currently ' + \
    'allowing cross-origin requests - (CORS ENABLED)')    

if __name__ == '__main__':
    app.run(host='0.0.0.0')
