FROM codait/max-base

# Fill in these with a link to the bucket containing the model and the model file name
ARG model_bucket=http://max-assets.s3-api.us-geo.objectstorage.softlayer.net/max-named-entity-tagger
ARG model_file=assets.tar.gz

RUN wget -nv --show-progress --progress=bar:force:noscroll ${model_bucket}/${model_file} --output-document=assets/${model_file}
RUN tar -x -C assets/ -f assets/${model_file} -v
RUN rm assets/${model_file}

COPY requirements.txt /workspace
RUN pip install -r requirements.txt

COPY . /workspace

EXPOSE 5000

CMD python app.py
