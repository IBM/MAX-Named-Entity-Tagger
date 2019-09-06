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

import pytest
import requests


def test_swagger():

    model_endpoint = 'http://localhost:5000/swagger.json'

    r = requests.get(url=model_endpoint)
    assert r.status_code == 200
    assert r.headers['Content-Type'] == 'application/json'

    json = r.json()
    assert 'swagger' in json
    assert json.get('info') and json.get('info').get('title') == 'MAX Named Entity Tagger'


def test_metadata():

    model_endpoint = 'http://localhost:5000/model/metadata'

    r = requests.get(url=model_endpoint)
    assert r.status_code == 200

    metadata = r.json()
    assert metadata['id'] == 'max-named-entity-tagger'
    assert metadata['name'] == 'MAX Named Entity Tagger'
    assert metadata['description'] == 'Named Entity Recognition model trained on a subset of '\
        'the Groningen Meaning Bank (GMB) dataset'
    assert metadata['license'] == 'Apache 2'
    assert metadata['type'] == 'Natural Language Processing'
    assert metadata['source'] == 'https://developer.ibm.com/exchanges/models/all/max-named-entity-tagger/'


if __name__ == '__main__':
    pytest.main([__file__])
