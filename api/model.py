from flask_restplus import Namespace, Resource, fields
from flask import request
from werkzeug.datastructures import FileStorage

from config import MODEL_META_DATA
from core.backend import ModelWrapper

api = Namespace('model', description='Model information and inference operations')

model_meta = api.model('ModelMetadata', {
    'id': fields.String(required=True, description='Model identifier'),
    'name': fields.String(required=True, description='Model name'),
    'description': fields.String(required=True, description='Model description'),
    'license': fields.String(required=False, description='Model license')
})


@api.route('/metadata')
class Model(Resource):
    @api.doc('get_metadata')
    @api.marshal_with(model_meta)
    def get(self):
        """Return the metadata associated with the model"""
        return MODEL_META_DATA

model_wrapper = ModelWrapper()

model_label = api.model('ModelLabel', {
    'id': fields.String(required=True, description='Label identifier'),
    'name': fields.String(required=True, description='Entity label'),
    'description': fields.String(required=False, description='Meaning of entity label')
})

labels_response = api.model('LabelsResponse', {
    'count': fields.Integer(required=True, description='Number of labels returned'),
    'labels': fields.List(fields.Nested(model_label), description='Entity labels that can be predicted by the model')
})

# Reference: http://gmb.let.rug.nl/manual.php
tag_desc = {
    'B-PER': 'Person; entities are limited to individuals that are human or have human characteristics, such as divine entities. B- tag indicates start of a new phrase.',
    'I-PER': 'Person; entities are limited to individuals that are human or have human characteristics, such as divine entities.',
    'B-GEO': 'Location; entities are limited to geographical entities such as geographical areas and landmasses, bodies of water, and geological formations. B- tag indicates start of a new phrase.',
    'I-GEO': 'Location; entities are limited to geographical entities such as geographical areas and landmasses, bodies of water, and geological formations.',
    'B-ORG': 'Organization; entities are limited to corporations, agencies, and other groups of people defined by an established organizational structure. B- tag indicates start of a new phrase.',
    'I-ORG': 'Organization; entities are limited to corporations, agencies, and other groups of people defined by an established organizational structure',
    'B-GPE': 'Geo-political Entity; entities are geographical regions defined by political and/or social groups. A GPE entity subsumes and does not distinguish between a city, a nation, its region, its government, or its people. B- tag indicates start of a new phrase.',
    'I-GPE': 'Geo-political Entity; entities are geographical regions defined by political and/or social groups. A GPE entity subsumes and does not distinguish between a city, a nation, its region, its government, or its people',
    'B-TIM': 'Time; limited to references to certain temporal entities that have a name, such as the days of the week and months of a year. B- tag indicates start of a new phrase.',
    'I-TIM': 'Time; limited to references to certain temporal entities that have a name, such as the days of the week and months of a year.',
    'B-EVE': 'Event; incidents and occasions that occur during a particular time. B- tag indicates start of a new phrase.',
    'I-EVE': 'Event; incidents and occasions that occur during a particular time.',
    'B-ART': 'Artifact; limited to manmade objects, structures and abstract entities, including buildings, facilities, art and scientific theories. B- tag indicates start of a new phrase.',
    'I-ART': 'Artifact; limited to manmade objects, structures and abstract entities, including buildings, facilities, art and scientific theories.',
    'B-NAT': 'Natural Object; entities that occur naturally and are not manmade, such as diseases, biological entities and other living things. B- tag indicates start of a new phrase.',
    'I-NAT': 'Natural Object; entities that occur naturally and are not manmade, such as diseases, biological entities and other living things.',
    'O': 'No entity type'
}

@api.route('/labels')
class Labels(Resource):
    @api.doc('get_labels')
    @api.marshal_with(labels_response)
    def get(self):
        '''Return the list of labels that can be predicted by the model'''
        result = {}
        result['labels'] = [{'id': l[0], 'name': l[1], 'description': tag_desc[l[1]]} for l in model_wrapper.id_to_tag.items()]
        result['count'] = len(model_wrapper.id_to_tag)
        return result

input_example = 'John lives in Brussels and works for the EU'
ent_example = ['I-PER', 'O', 'O', 'I-LOC', 'O', 'O', 'O', 'O', 'I-ORG']
term_example = ['John', 'lives', 'in', 'Brussels', 'and', 'works', 'for', 'the', 'EU']

model_input = api.model('ModelInput', {
    'text': fields.String(required=True, description='Text for which to predict entities', example=input_example)
})

model_prediction = api.model('ModelPrediction', {
    'tags': fields.List(fields.String, required=True, description='List of predicted entity tags, one per term in the input text.', example=ent_example),
    'terms': fields.List(fields.String, required=True, 
        description='Terms extracted from input text pre-processing. Each term has a corresponding predicted entity tag in the "tags" field.',
        example=term_example)
})

predict_response = api.model('ModelPredictResponse', {
    'status': fields.String(required=True, description='Response status message'),
    'prediction': fields.Nested(model_prediction, description='Model prediction')
})

# Set up parser for input data (http://flask-restplus.readthedocs.io/en/stable/parsing.html)
input_parser = api.parser()
# Example parser for file input
input_parser.add_argument('file', type=FileStorage, location='files', required=True)


@api.route('/predict')
class Predict(Resource):

    @api.doc('predict')
    @api.expect(model_input)
    @api.marshal_with(predict_response)
    def post(self):
        '''Make a prediction given input data'''
        result = {'status': 'error'}

        j = request.get_json()
        text = j['text']
        entities, terms = model_wrapper.predict(text)
        model_pred = {
            'tags': entities,
            'terms': terms
        }
        result['prediction'] = model_pred
        result['status'] = 'ok'

        return result
