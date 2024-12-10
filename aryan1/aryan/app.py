import requests
from bs4 import BeautifulSoup
from transformers import BertForQuestionAnswering, BertTokenizer
import torch
from flask import Flask, request, render_template

# Initialize the Flask app
app = Flask(__name__)

# Initialize the pre-trained BERT model and tokenizer
model = BertForQuestionAnswering.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
tokenizer_for_bert = BertTokenizer.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')

# URL to fetch content from
URL = ['https://en.wikipedia.org/wiki/Goods_and_Services_Tax_(India)','https://en.wikipedia.org/wiki/GST','https://www.gst.gov.in/','https://wiki.koha-community.org/wiki/GST_Rewrite_RFC']

def bert_question_answer(question, passage, max_len=500):
    # Tokenize input question and passage
    input_ids = tokenizer_for_bert.encode(question, passage, max_length=max_len, truncation=True)

    # Getting number of tokens in 1st sentence (question) and 2nd sentence (passage)
    sep_index = input_ids.index(102)
    len_question = sep_index + 1
    len_passage = len(input_ids) - len_question

    # Need to separate question and passage
    segment_ids = [0] * len_question + [1] * (len_passage)

    # Getting start and end scores for the answer
    start_token_scores = model(torch.tensor([input_ids]), token_type_ids=torch.tensor([segment_ids]))[0]
    end_token_scores = model(torch.tensor([input_ids]), token_type_ids=torch.tensor([segment_ids]))[1]

    # Convert scores tensors to numpy arrays
    start_token_scores = start_token_scores.detach().numpy().flatten()
    end_token_scores = end_token_scores.detach().numpy().flatten()

    # Get the most likely start and end of the answer
    start_index = start_token_scores.argmax()
    end_index = end_token_scores.argmax()

    # Convert the token indices back to words
    answer = tokenizer_for_bert.convert_tokens_to_string(tokenizer_for_bert.convert_ids_to_tokens(input_ids[start_index:end_index + 1]))
    return answer.strip()
# Select a random URL or a default one from the list
import random

def extract_text_from_url(url):
    # Fetch the content of the URL
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    
    # Extract the main content of the Wikipedia article
    content = soup.find('div', {'class': 'mw-parser-output'})
    
    # Get text from the article content, stripping unwanted extra whitespaces
    text = content.get_text(separator=' ', strip=True)
    return text

@app.route('/ask_question', methods=['POST'])
def ask_question():
    question = request.form['question']

    # Select a random URL or use a default URL
    url = random.choice(URL)  # or use a specific URL directly
    passage = extract_text_from_url(url)

    # Get the answer using BERT
    answer = bert_question_answer(question, passage)

    return render_template('index.html', answer=answer, question=question)


@app.route('/')
def index():
    return render_template('index.html', answer=None)

if __name__ == "__main__":
    app.run(debug=True)
