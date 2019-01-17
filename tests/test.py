import pytest
import requests

def test_swagger():

    model_endpoint = 'http://localhost:5000/swagger.json'

    r = requests.get(url=model_endpoint)
    assert r.status_code == 200
    assert r.headers['Content-Type'] == 'application/json'

    json = r.json()
    assert 'swagger' in json
    assert json.get('info') and json.get('info').get('title') == 'Model Asset Exchange Microservice'


def test_metadata():

    model_endpoint = 'http://localhost:5000/model/metadata'

    r = requests.get(url=model_endpoint)
    assert r.status_code == 200

    metadata = r.json()
    assert metadata['id'] == 'ner_model'
    assert metadata['name'] == 'Named Entity Recognition'
    assert metadata['description'] == 'Named Entity Recognition model trained on a subset of the Groningen Meaning Bank (GMB) dataset'
    assert metadata['license'] == 'Apache 2'


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
