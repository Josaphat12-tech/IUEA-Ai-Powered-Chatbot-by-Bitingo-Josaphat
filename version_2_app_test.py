import os
from pathlib import Path

from dotenv import load_dotenv
from flask import Flask, render_template, request
#from flask_caching import Cache
from flask_compress import Compress

from llama_index import GPTVectorStoreIndex, LLMPredictor, ServiceContext, download_loader
from langchain import OpenAI
import openai
from config import API_KEY


os.environ["OPENAI_API_KEY"] = API_KEY
load_dotenv()
openai.api_key = os.environ["OPENAI_API_KEY"]



app = Flask(__name__)
#compress = Compress(app)
#cache = Cache(app, config={'CACHE_TYPE': 'simple'})


FILES = "./files"
PDF_FILE = "file.pdf"


def init():
    if not os.path.exists(FILES):
        os.mkdir(FILES)


def load_pdf_file():
    PDFReader = download_loader("PDFReader")
    loader = PDFReader()
    return loader.load_data(file=Path(PDF_FILE))


def get_index():
    documents = load_pdf_file()

    llm_predictor = LLMPredictor(llm=OpenAI(temperature=0.7, model_name="text-davinci-003"))

    service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor, chunk_size_limit=2048)
    index = GPTVectorStoreIndex.from_documents(documents, service_context=service_context)

    return index


@app.route('/', methods=['GET', 'POST'])
#@cache.cached(timeout=60)
def home():
    init()
    index = get_index()

    if request.method == 'POST':
        prompt = request.form.get('prompt')

        if prompt == "exit":
            return render_template('exit.html')

        query_index = index.as_query_engine()
        response = query_index.query(prompt)
        response = str(response)

        if response.startswith("\n"):
            response = response[1:]

        return render_template('home.html', response=response, prompt=prompt)

    return render_template('home.html')


if __name__ == '__main__':
    app.run(debug=False)
