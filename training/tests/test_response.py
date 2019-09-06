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


# def test_labels():
#     '''Test expected outcome of training run on sample data'''
#     model_endpoint = 'http://localhost:5000/model/labels'

#     r = requests.get(url=model_endpoint)
#     assert r.status_code == 200

#     # only expect a subset of the labels to be present
#     labels = r.json()
#     tags = labels['labels']
#     assert labels['count'] == 9
#     assert tags[0]['name'] == 'B-PER'
#     assert tags[0]['id'] == '0'
#     assert tags[-1]['id'] == '8'
#     assert tags[-1]['name'] == 'B-MISC'


def test_response():
    model_endpoint = 'http://localhost:5000/model/predict'
    text = 'Jean Pierre lives in New York.'
    test_json = {
        "text": text
    }
    expected_tags = ["B-PER", "I-PER", "O", "O", "B-LOC", "I-LOC", "O"]
    expected_terms = ["Jean", "Pierre", "lives", "in", "New", "York", "."]

    r = requests.post(url=model_endpoint, json=test_json)

    assert r.status_code == 200
    response = r.json()
    assert response['status'] == 'ok'
    assert response['prediction']['terms'] == expected_terms
    assert response['prediction']['tags'] == expected_tags


if __name__ == '__main__':
    pytest.main([__file__])
