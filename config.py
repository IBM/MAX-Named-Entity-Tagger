# Flask settings
DEBUG = False

# Flask-restplus settings
RESTPLUS_MASK_SWAGGER = False

# Application settings

# API metadata
API_TITLE = 'Model Asset Exchange Server'
API_DESC = 'An API for serving models'
API_VERSION = '0.1'

# default model
MODEL_NAME = 'Named Entity Recognition'
MODEL_ID = 'ner_model'
DEFAULT_MODEL_PATH = 'assets'
MODEL_LICENSE = 'Apache 2'

MODEL_META_DATA = {
    'id': '{}'.format(MODEL_ID),
    'name': '{}'.format(MODEL_NAME),
    'description': '{} model trained on a subset of the Groningen Meaning Bank (GMB) dataset'.format(MODEL_NAME),
    'type': 'Natural Language Processing',
    'license': '{}'.format(MODEL_LICENSE)
}
