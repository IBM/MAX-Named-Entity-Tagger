#
# Copyright 2018-2019 IBM Corp. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

# Flask settings
DEBUG = False

# Flask-restplus settings
RESTPLUS_MASK_SWAGGER = False

# Application settings
API_TITLE = 'MAX Named Entity Tagger'
API_DESC = 'Locate and tag named entities in text'
API_VERSION = '1.2.0'

MODEL_NAME = 'Named Entity Recognition'
MODEL_ID = 'ner_model'
DEFAULT_MODEL_PATH = 'assets'
MODEL_LICENSE = 'Apache 2'

MODEL_META_DATA = {
    'id': '{}'.format(MODEL_ID),
    'name': '{}'.format(MODEL_NAME),
    'description': '{} model trained on a subset of the Groningen Meaning Bank (GMB) dataset'.format(MODEL_NAME),
    'type': 'Natural Language Processing',
    'license': '{}'.format(MODEL_LICENSE),
    'source': 'https://developer.ibm.com/exchanges/models/all/max-named-entity-tagger/'
}
