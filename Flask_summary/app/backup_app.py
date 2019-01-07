from flask import Flask, flash, render_template, request, redirect, url_for
import requests
import bs4
import urlparse
import textract
import urllib2
import re
import unicodedata
from nltk import sent_tokenize, word_tokenize, pos_tag
from HTMLParser import HTMLParser
import string
import normalization
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from scipy.sparse.linalg import svds
import networkx
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer as Summarizer
from sumy.nlp.stemmers import Stemmer
from sumy.utils import get_stop_words
import numpy as np
from werkzeug.utils import secure_filename
import os
import logging
from logging import handlers
import sys

wnl = WordNetLemmatizer()
html_parser = HTMLParser()
stop = stopwords.words('english')
stopword_list = nltk.corpus.stopwords.words('english')
stopword_list = stopword_list + ['mr', 'mrs', 'come', 'go', 'get',
                                 'tell', 'listen', 'one', 'two', 'three',
                                 'four', 'five', 'six', 'seven', 'eight',
                                 'nine', 'zero', 'join', 'find', 'make',
                                 'say', 'ask', 'tell', 'see', 'try', 'back',
                                 'also']

MAX_SIZE = 10 * 1024 * 1024  # 10MB file size
this_file, this_file_ext = os.path.splitext(os.path.basename(__file__))
LOG_FILE = this_file + ".log"
LOG_PATH = os.path.join("log", LOG_FILE)
log = logging.getLogger("root")
log.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(asctime)s--%(levelname)s--%(name)s-### %(message)s")  # noqa
stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setFormatter(formatter)
log.addHandler(stream_handler)
file_handler = handlers.RotatingFileHandler(LOG_PATH, mode='a', maxBytes=MAX_SIZE, backupCount=10)  # noqa
file_handler.setFormatter(formatter)
log.addHandler(file_handler)

UPLOAD_FOLDER = 'uploaded_files'
ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'doc', 'docx'])

app = Flask(__name__)
log.info("app: {}".format(app))
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


@app.route('/')
@app.route('/home')
def home():
    return render_template('home.html')


@app.route('/about')
def about():
    return render_template('about.html')


@app.route('/summary', methods=['GET', 'POST'])
def summary():

    if request.method == 'POST':
        # result = request.form
        text_input = request.form['textOptionVar']
        if text_input == "url":
            url = request.form['urlText']
            page_title, dnld_text = extract_url_text(url)
            log.info("dnld_text type: {}".format(type(dnld_text)))
            doc_summary = process_text_summary(dnld_text)
        elif text_input == "file_select":
            if 'file' not in request.files:
                log.info("Missing files to upload")
                flash('No file part')
                return redirect(request.url)
            file = request.files['file']
            if file.filename == '':
                flash('No selected file')
                return redirect(request.url)
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            log.info("filename: {}".format(filename))
            filepath = os.path.join(UPLOAD_FOLDER, filename)
            log.info("filepath: {}".format(filepath))
            page_title, dnld_text = extract_document_text(filepath)
            log.info("dnld_text type: {}".format(type(dnld_text)))
            doc_summary = process_text_summary(dnld_text)
        else:
            log.info("Text Input Option not selected.  Please select one.")

    for line in doc_summary:
        log.info("sum line: {}".format(line))

    return render_template('summary.html', page_title=page_title, doc_summary=doc_summary)  # noqa


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def extract_document_text(filepath):
    """
    This function extracts the extension of the filepath provided.  If
    pdf, doc, docx, or txt extension is detected, it reads the file and
    tries to extract text from it.
    """
    log.debug(">>> extract_document_text...")
#         file_path, filename = os.path.split(filepath)
#         doc_file, file_extension = os.path.splitext(filename)
#         log.info("file_path: {}\ndoc_file: {}\nfile_extension: {}".format(file_path, doc_file, file_extension))  # noqa
    doc_file, file_extension = os.path.splitext(filepath)
    log.info("doc_file: {}\nfile_extension: {}".format(doc_file, file_extension))  # noqa
    if file_extension == ".doc" or file_extension == ".docx":
        log.info("{} file received.".format(file_extension))
        page_title = "None"
        log.info("DOC Page Title: {}".format(page_title))
        try:
            del self.page_title
        except:
            log.info("self.page_title variable is not set yet.  That's OK.")  # noqa
        dnld_text = textract.process(filepath)
        dnld_text = unicode(dnld_text, "utf-8")
    elif file_extension == ".pdf":
        log.info("{} file received.".format(file_extension))
        page_title = "None"
        log.info("PDF Page Title: {}".format(page_title))
        try:
            del self.page_title
        except:
            log.info("self.page_title variable is not set yet.  That's OK.")  # noqa
        # dnld_text = self.pdf_to_text(filepath)
        dnld_text = textract.process(filepath)
        try:
            dnld_text = unicode(dnld_text, "utf-8")
        except:
            dnld_text = dnld_text
    elif file_extension == ".txt":
        log.info("{} file received.".format(file_extension))
        page_title = "None"
        log.info("txt Page Title: {}".format(page_title))
        try:
            del self.page_title
        except:
            log.info("self.page_title variable is not set yet.  That's OK.")  # noqa
        with open(filepath, 'r') as f:
            dnld_text = f.read()
        dnld_text = unicode(dnld_text, "utf-8")
    else:
        log.info("{} file NOT SUPPORTED.".format(file_extension))
        sys.exit()

    return page_title, dnld_text


def extract_url_text(url):
    """
        This function extracts the extension of the url.  If pdf or txt
        extension is detected, it treats differently.  For all the other
        extension, it treats as html page and tries to extract text from it.
    """
    log.debug(">>> extract_url_text...")

    path = urlparse.urlparse(url).path
    ext = os.path.splitext(path)[1]
    log.info("ext: {}".format(ext))
    if ext == ".pdf":
        log.info("PDF page received.")
        page_title = "None"
        log.info("PDF Page Title: {}".format(page_title))
        file_name = "pdf_document.pdf"
        download_file(url, file_name)
        # dnld_text = pdf_to_text(file_name)
        dnld_text = textract.process(file_name)
        try:
            dnld_text = unicode(dnld_text, "utf-8")
        except:
            dnld_text = dnld_text
    elif ext == ".txt":
        log.info("txt page received.")
        page_title = "None"
        log.info("txt Page Title: {}".format(page_title))
        dnld_text = requests.get(url).text
        try:
            dnld_text = unicode(dnld_text, "utf-8")
        except:
            dnld_text = dnld_text
    else:
        log.info("HTML page received.")
        page = requests.get(url)  # noqa
        soup = bs4.BeautifulSoup(page.content.decode('utf8'), 'html.parser')  # noqa
        page_title = soup.title.string
        log.info("HTML Page Title: {}".format(page_title))
        to_summarize = map(lambda p: p.text, soup.find_all('p'))
        log.info("HTML Page type: {}".format(type(to_summarize)))
        log.info("HTML Page len: {}".format(len(to_summarize)))
        dnld_text = "\n".join((to_summarize))

    return page_title, dnld_text


def download_file(download_url, file_name):
    response = urllib2.urlopen(download_url)
    with open(file_name, 'wb') as dnld_file:
        dnld_file.write(response.read())
        log.info("temp_pdf_doc.pdf write Completed")


def process_text_summary(dnld_text):

    algorithm = request.form['algorithmVar']
    num_sentences = int(request.form['sum_sentences'])
    log.info("algorithm: {}".format(algorithm))
    log.info("num_sentences: {}".format(num_sentences))
    docs = remove_non_ascii_chars(dnld_text)
    sentences = text_to_sentence_list(docs)
    log.info(sentences)

    if algorithm == "LSA_Text_Summarizer":
        log.info("Performing summary with {} --> \n".format(algorithm))
        docs = clean_and_normalize(sentences)
        page_summary = lsa_text_summarizer(docs, num_sentences)
        doc_summary = list()
        for index in page_summary:
            doc_summary.append(sentences[index])
    elif algorithm == "TextRank_Text_Summarizer":
        log.info("Performing summary with {} --> \n".format(algorithm))
        docs = clean_and_normalize(sentences)
        page_summary = textrank_text_summarizer(docs, num_sentences)  # noqa
        doc_summary = list()
        for index in page_summary:
            doc_summary.append(sentences[index])
    elif algorithm == "Sumy_Text_Summarizer":
        log.info("Performing summary with {} --> \n".format(algorithm))
        LANGUAGE = "english"
        text = "\n".join((sentences))
        parser = PlaintextParser(text, Tokenizer(LANGUAGE))
        doc_summary = sumy_text_summarizer(parser, LANGUAGE, num_sentences)  # noqa
    else:
        log.info("Not supported Algorithm.")

    return doc_summary


def lsa_text_summarizer(text,
                           num_sentences=4,
                           num_topics=1,
                           feature_type='frequency',
                           sv_threshold=0.5):  # noqa

    vec, dt_matrix = build_feature_matrix(text, feature_type)

    td_matrix = dt_matrix.transpose()
    td_matrix = td_matrix.multiply(td_matrix > 0)

    u, s, vt = low_rank_svd(td_matrix, num_topics)
    min_sigma_value = max(s) * sv_threshold
    s[s < min_sigma_value] = 0

    salience_scores = np.sqrt(np.dot(np.square(s), np.square(vt)))
    top_sentence_indices = salience_scores.argsort()[-num_sentences:][::-1]
    top_sentence_indices.sort()

    return top_sentence_indices


def remove_non_ascii_chars(document):
    document = re.sub('\n', ' ', document)
    log.info("Document type: {}".format(type(document)))
    if isinstance(document, str):
        document = document
    elif isinstance(document, unicode):
        return unicodedata.normalize('NFKD', document).encode('ascii', 'ignore')  # noqa
    else:
        raise ValueError('Document is neither string nor unicode!')
    document = document.strip()

    return document


def text_to_sentence_list(document):
    sentences = sent_tokenize(document)
    sentences = [sentence.strip() for sentence in sentences]
    return sentences


def clean_and_normalize(corpus,
                        lemmatize=True,
                        only_text_chars=False,
                        tokenize=False):

    normalized_corpus = []
    for text in corpus:
        cleaned_text = text.lower()
        # Remove URLs
        cleaned_text = re.sub(r"http\S+", " ", cleaned_text)
        # Remove non-ascii characters
        printable = set(string.printable)
        cleaned_text = filter(lambda x: x in printable, cleaned_text)
        # Remove unescape special characters
        cleaned_text = cleaned_text.decode('iso-8859-1')
        cleaned_text = HTMLParser().unescape(cleaned_text)
        cleaned_text = remove_stop_words(cleaned_text)
        # Remove leading and trailing hyphens
        cleaned_text = cleaned_text.strip('-')
        # Strip all but requested chars
        if only_text_chars:
            cleaned_text = re.sub('[^A-Za-z]', ' ', cleaned_text)
        else:
            cleaned_text = re.sub('[^A-Za-z0-9\?\.\,\!]', ' ', cleaned_text)  # noqa
        # Abbreviations Normalization
        cleaned_text = normalization.abbreviations(cleaned_text)
        # Contractions Normalization
        cleaned_text = normalization.contractions(cleaned_text)
        # Lemmatize Normalization
        if lemmatize:
            cleaned_text = lemmatize_text(cleaned_text)
        # Normalize
        # Combines all normalization methods above
        cleaned_text = normalization.normalize(cleaned_text)

        if tokenize:
            cleaned_text = tokenize_text(cleaned_text)
            normalized_corpus.append(cleaned_text)
        else:
            normalized_corpus.append(cleaned_text)

    return normalized_corpus


def remove_stop_words(document):
    """Returns document without stop words"""
    document = ' '.join([i for i in document.split() if i not in stop])
    return document


def lemmatize_text(text):
    lemmatized_tokens = [wnl.lemmatize(i, j[0].lower()) if j[0].lower() in ['a', 'n', 'v'] else wnl.lemmatize(i) for i, j in pos_tag(word_tokenize(text))]  # noqa
    lemmatized_text = ' '.join(lemmatized_tokens)

    return lemmatized_text


def tokenize_text(text):
    tokens = nltk.word_tokenize(text)
    tokens = [token.strip() for token in tokens]

    return tokens


def build_feature_matrix(documents, feature_type):

    feature_type = feature_type.lower().strip()
    if feature_type == 'binary':
        vectorizer = CountVectorizer(binary=True, min_df=1, ngram_range=(1, 1))  # noqa
    elif feature_type == 'frequency':
        vectorizer = CountVectorizer(binary=False, min_df=1, ngram_range=(1, 1))  # noqa
    elif feature_type == 'tfidf':
        vectorizer = TfidfVectorizer(min_df=1, ngram_range=(1, 1))
    else:
        raise Exception("Wrong feature type entered. Possible values: 'binary', 'frequency', 'tfidf'")  # noqa

    feature_matrix = vectorizer.fit_transform(documents).astype(float)

    return vectorizer, feature_matrix


def low_rank_svd(matrix, num_topics):
    u, s, vt = svds(matrix, k=num_topics)
    return u, s, vt


def textrank_text_summarizer(text,
                            num_sentences=4,
                            num_topics=1,
                            feature_type='tfidf',
                            sv_threshold=0.5):  # noqa

    vec, dt_matrix = build_feature_matrix(text, feature_type)
    similarity_matrix = (dt_matrix * dt_matrix.T)

    similarity_graph = networkx.from_scipy_sparse_matrix(similarity_matrix)  # noqa
    scores = networkx.pagerank(similarity_graph)

    ranked_sentences = sorted(((score, index) for index, score in scores.items()), reverse=True)  # noqa

    top_sentence_indices = [ranked_sentences[index][1] for index in range(num_sentences)]  # noqa
    top_sentence_indices.sort()

    return top_sentence_indices


def sumy_text_summarizer(parser, LANGUAGE, num_sentences=4):
    stemmer = Stemmer(LANGUAGE)
    summarizer = Summarizer(stemmer)
    summarizer.stop_words = get_stop_words(LANGUAGE)
    summary = list()
    for sentence in summarizer(parser.document, num_sentences):
        summary.append(str(sentence).decode('ascii', 'ignore'))

    return summary
