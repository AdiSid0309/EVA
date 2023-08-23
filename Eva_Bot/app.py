import pickle
from flask import Flask, render_template, request

app = Flask(__name__)

# Load the pickled Naive Bayes classifier and TF-IDF vectorizer
with open('naive_bayes_classifier.pickle', 'rb') as f:
    classifier = pickle.load(f)

with open('tfidf_vectorizer.pickle', 'rb') as f:
    vectorizer = pickle.load(f)

# Initialize chat history list
chat_history = []

@app.route('/')
def index():
    return render_template('index.html', chat_history=chat_history)

@app.route('/get_answer', methods=['POST'])
def get_answer():
    question = request.form['question']

    # Vectorize the user's question using the loaded TF-IDF vectorizer
    question_vectorized = vectorizer.transform([question])

    # Get the answer from the Naive Bayes classifier
    answer = classifier.predict(question_vectorized)[0]

    # Add the query and response to the chat history
    chat_history.append({'user': question, 'bot': answer})

    return render_template('index.html', question=question, answer=answer, chat_history=chat_history)

if __name__ == '__main__':
    app.run(debug=True)