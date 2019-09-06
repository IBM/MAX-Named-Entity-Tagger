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


def test_labels():
    model_endpoint = 'http://localhost:5000/model/labels'

    r = requests.get(url=model_endpoint)
    assert r.status_code == 200

    labels = r.json()
    tags = labels['labels']
    assert labels['count'] == 17
    assert tags[0]['name'] == 'O'
    assert tags[0]['id'] == '0'
    assert tags[-1]['id'] == '16'
    assert tags[-1]['name'] == 'I-ORG'


def test_response():
    model_endpoint = 'http://localhost:5000/model/predict'
    text = 'John lives in Brussels and works for the EU'
    test_json = {
        "text": text
    }
    expected_tags = ["B-PER", "O", "O", "B-GEO", "O", "O", "O", "O", "B-ORG"]
    expected_terms = ["John", "lives", "in", "Brussels", "and", "works", "for", "the", "EU"]

    r = requests.post(url=model_endpoint, json=test_json)

    assert r.status_code == 200
    response = r.json()
    assert response['status'] == 'ok'
    assert response['prediction']['terms'] == expected_terms
    assert response['prediction']['tags'] == expected_tags


if __name__ == '__main__':
    pytest.main([__file__])
