from flask import Flask, render_template, request, jsonify
from transformers import BartTokenizer, BartForConditionalGeneration
import pdfplumber
import re

app = Flask(__name__, template_folder='templates', static_folder='static')

@app.route('/',methods=['GET'])
def indexPage():
    return render_template('index.html')

@app.route('/about',methods=['GET'])
def aboutPage():
    return render_template('about.html')

@app.route('/upload',methods=['GET'])
def uploadPage():
    return render_template('upload.html')


def clean_summary(summary):
    summary = re.sub(r'\s+', ' ', summary).strip()
    summary = re.sub(r'[^a-zA-Z0-9.,!?\'"-]', ' ', summary)
    summary = re.sub(r'\s([?.!,"\'-](?:\s|$))', r'\1', summary)

    if summary and summary[0].islower():
        summary = summary[0].upper() + summary[1:]
    
    if summary and summary[-1] not in {'.', '!', '?'}:
        summary += '.'
        
    return summary

def generate_summary(text, max_length, min_length, num_beams,model_type):
    
    tokenizer = BartTokenizer.from_pretrained(f'./Models/{model_type}')
    model = BartForConditionalGeneration.from_pretrained(f'./Models/{model_type}')
    
    inputs = tokenizer.encode(text, return_tensors='pt', max_length=1024, truncation=True)
    summary_ids = model.generate(
        inputs,
        max_length=150,
        min_length=30,
        num_beams=2,
        length_penalty=2.0,
        early_stopping=True
    )
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

def pdf_loader(file_path):
    text = ""
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text()
    return text


@app.route('/summarize', methods=['POST'])
def summarize():
    if 'researchPaper' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    model_type = request.form['modelType']
    qualityIndex = int(request.form['qualityIndex'])
    max_limit = int(request.form['max_limit'])
    min_limit = int(request.form['min_limit'])
    author = request.form['authorName']
    paperTitle = request.form['paperTitle']
    query_pdf = request.files['researchPaper']
    
    text = pdf_loader(query_pdf)
    
    if not text:
        return jsonify({'error': 'No text provided'}), 400
    print("Request Arrived")
    raw_summary = generate_summary(text,max_limit, min_limit, qualityIndex,model_type)
    cleaned_summary = clean_summary(raw_summary)
    print("Request Sent")
    return render_template('fetch.html',
                           summary=cleaned_summary,
                           author=author,
                           paperTitle=paperTitle,
                           model_type=model_type,
                           qualityIndex=qualityIndex,
                           max_limit=max_limit,
                           min_limit=min_limit)


if __name__ == '__main__':
    app.run(debug=True,port=3000)