# import nltk
# nltk.download('popular')
# from nltk.stem import WordNetLemmatizer
# lemmatizer = WordNetLemmatizer()
import pickle
import numpy as np

from keras.models import load_model
model = load_model('model.h5')
import json
import random
import os
from waitress import serve
intents = json.loads(open('data.json').read())
words = pickle.load(open('texts.pkl','rb'))
classes = pickle.load(open('labels.pkl','rb'))


from flask import Flask, render_template, request

app = Flask(__name__,template_folder='templates')
app.static_folder = 'static'

@app.route("/")
def home():
    return render_template("index.html")

questions = {
    1: 'Do you have interest in Human anatomy?',
    2: 'Do you have interest in Human Psychology?',
    3: 'Do you have interest in Designing of Drugs?',
    4: 'Do you have interest in Human Physiology?',
    5: 'Do you have interest in Machinery and manufacturing of goods',
    6: 'Do you have an interest in making all kind of devices and equipment?',
    7: 'Do you have an interest in developing software applications?',
    8: 'Do you have an interest in making of buildings, roads and bridges?',
    9: 'Do you have an interest in designing and developing chemicals ?',
    10: 'Do you have an interest in Space and Science?',
    11: 'Do you have an interest interest in providing people with knowledge and teaching them in something you are good at?',
    12: 'Do you have an interest in being a part of defense?',
    13: 'Do you have an interest in Literature?',
    14: 'Do you have an interest in Economy of the world?',
    15: 'Do you have an interest in working with the accounts of a company or an individual?',
    16: 'Do you have an interest in Practicing Law?',
    17: 'Do you have an interest in Managing projects and operations?',
    18: 'Do you have an interest in selling product and services by researching and advertising?',
    19: 'Do you have an interest in making income on your own?',
    20: 'Do you have an interest in working for the public?',
    21: 'Do you have an interest in Housing management?',
    22: 'Do you have an interest in Journalism?',
    23: 'Do you have an interest in the creation a curation of content, information (digital or physical)?',
    }

characters = [
    {'name': 'Doctor',                          'answers': {1: 1, 2: 0, 3: 0, 4: 0, 5:0, 6:0, 7:0, 8:0, 9:0, 10:0, 11:0, 12:0, 13:0, 14:0, 15:0, 16:0, 17:0, 18:0, 19:0, 20:0, 21:0, 22:0, 23:0}},
    {'name': 'Therapist',                       'answers': {1: 0, 2: 1, 3: 0, 4: 0, 5:0, 6:0, 7:0, 8:0, 9:0, 10:0, 11:0, 12:0, 13:0, 14:0, 15:0, 16:0, 17:0, 18:0, 19:0, 20:0, 21:0, 22:0, 23:0}},
    {'name': 'Pharmacy',                        'answers': {1: 0, 2: 0, 3: 1, 4: 0, 5:0, 6:0, 7:0, 8:0, 9:0, 10:0, 11:0, 12:0, 13:0, 14:0, 15:0, 16:0, 17:0, 18:0, 19:0, 20:0, 21:0, 22:0, 23:0}},
    {'name': 'Physio-therapist',                'answers': {1: 0, 2: 0, 3: 0, 4: 1, 5:0, 6:0, 7:0, 8:0, 9:0, 10:0, 11:0, 12:0, 13:0, 14:0, 15:0, 16:0, 17:0, 18:0, 19:0, 20:0, 21:0, 22:0, 23:0}},
    {'name': 'Mechanical Engineering',          'answers': {1: 0, 2: 0, 3: 0, 4: 0, 5:1, 6:0, 7:0, 8:0, 9:0, 10:0, 11:0, 12:0, 13:0, 14:0, 15:0, 16:0, 17:0, 18:0, 19:0, 20:0, 21:0, 22:0, 23:0}},
    {'name': 'Electronic Engineering',          'answers': {1: 0, 2: 0, 3: 0, 4: 0, 5:0, 6:1, 7:0, 8:0, 9:0, 10:0, 11:0, 12:0, 13:0, 14:0, 15:0, 16:0, 17:0, 18:0, 19:0, 20:0, 21:0, 22:0, 23:0}},
    {'name': 'Computer Science Engineering',    'answers': {1: 0, 2: 0, 3: 0, 4: 0, 5:0, 6:0, 7:1, 8:0, 9:0, 10:0, 11:0, 12:0, 13:0, 14:0, 15:0, 16:0, 17:0, 18:0, 19:0, 20:0, 21:0, 22:0, 23:0}},
    {'name': 'Civil Engineering',               'answers': {1: 0, 2: 0, 3: 0, 4: 0, 5:0, 6:0, 7:0, 8:1, 9:0, 10:0, 11:0, 12:0, 13:0, 14:0, 15:0, 16:0, 17:0, 18:0, 19:0, 20:0, 21:0, 22:0, 23:0}},
    {'name': 'Chemical Engineering',            'answers': {1: 0, 2: 0, 3: 0, 4: 0, 5:0, 6:0, 7:0, 8:0, 9:1, 10:0, 11:0, 12:0, 13:0, 14:0, 15:0, 16:0, 17:0, 18:0, 19:0, 20:0, 21:0, 22:0, 23:0}},
    {'name': 'Aeronautics Engineering',         'answers': {1: 0, 2: 0, 3: 0, 4: 0, 5:0, 6:0, 7:0, 8:0, 9:0, 10:1, 11:0, 12:0, 13:0, 14:0, 15:0, 16:0, 17:0, 18:0, 19:0, 20:0, 21:0, 22:0, 23:0}},
    {'name': 'Teacher',                         'answers': {1: 0, 2: 0, 3: 0, 4: 0, 5:0, 6:0, 7:0, 8:0, 9:0, 10:0, 11:1, 12:0, 13:0, 14:0, 15:0, 16:0, 17:0, 18:0, 19:0, 20:0, 21:0, 22:0, 23:0}},
    {'name': 'Defense',                         'answers': {1: 0.5, 2: 0.5, 3: 0.5, 4:0.5,5:0.5, 6:0.5, 7:0.5, 8:0.5, 9:0.5, 10:0.5, 11:0, 12:1, 13:0, 14:0, 15:0, 16:0, 17:0, 18:0, 19:0, 20:0, 21:0, 22:0, 23:0}},
    {'name': 'Author',                          'answers': {1: 0, 2: 0, 3: 0, 4: 0, 5:0, 6:0, 7:0, 8:0, 9:0, 10:0, 11:0, 12:0, 13:1, 14:0, 15:0, 16:0, 17:0, 18:0, 19:0, 20:0, 21:0, 22:0, 23:0}},
    {'name': 'Economist',                       'answers': {1: 0, 2: 0, 3: 0, 4: 0, 5:0, 6:0, 7:0, 8:0, 9:0, 10:0, 11:0, 12:0, 13:0, 14:1, 15:0, 16:0, 17:0, 18:0, 19:0, 20:0, 21:0, 22:0, 23:0}},
    {'name': 'Accounting',                      'answers': {1: 0, 2: 0, 3: 0, 4: 0, 5:0, 6:0, 7:0, 8:0, 9:0, 10:0, 11:0, 12:0, 13:0, 14:0, 15:1, 16:0, 17:0, 18:0, 19:0, 20:0, 21:0, 22:0, 23:0}},
    {'name': 'Attorney',                        'answers': {1: 0, 2: 0, 3: 0, 4: 0, 5:0, 6:0, 7:0, 8:0, 9:0, 10:0, 11:0, 12:0, 13:0, 14:0, 15:0, 16:1, 17:0, 18:0, 19:0, 20:0, 21:0, 22:0, 23:0}},
    {'name': 'Management',                      'answers': {1: 0, 2: 0, 3: 0, 4: 0, 5:0, 6:0, 7:0, 8:0, 9:0, 10:0, 11:0, 12:0, 13:0, 14:0, 15:0, 16:0, 17:1, 18:0, 19:0, 20:0, 21:0, 22:0, 23:0}},
    {'name': 'Marketing',                       'answers': {1: 0, 2: 0, 3: 0, 4: 0, 5:0, 6:0, 7:0, 8:0, 9:0, 10:0, 11:0, 12:0, 13:0, 14:0, 15:0, 16:0, 17:0, 18:1, 19:0, 20:0, 21:0, 22:0, 23:0}},
    {'name': 'Entrepreneur',                    'answers': {1: 0, 2: 0, 3: 0, 4: 0, 5:0, 6:0, 7:0, 8:0, 9:0, 10:0, 11:0, 12:0, 13:0, 14:0, 15:0, 16:0, 17:0, 18:0, 19:1, 20:0, 21:0, 22:0, 23:0}},
    {'name': 'Civil Services',                  'answers': {1: 0, 2: 0, 3: 0, 4: 0, 5:0, 6:0, 7:0, 8:0, 9:0, 10:0, 11:0, 12:0, 13:0, 14:0, 15:0, 16:0, 17:0, 18:0, 19:0, 20:1, 21:0, 22:0, 23:0}},
    {'name': 'Hotel Management',                'answers': {1: 0, 2: 0, 3: 0, 4: 0, 5:0, 6:0, 7:0, 8:0, 9:0, 10:0, 11:0, 12:0, 13:0, 14:0, 15:0, 16:0, 17:0, 18:0, 19:0, 20:0, 21:1, 22:0, 23:0}},
    {'name': 'Journalist',                      'answers': {1: 0, 2: 0, 3: 0, 4: 0, 5:0, 6:0, 7:0, 8:0, 9:0, 10:0, 11:0, 12:0, 13:0, 14:0, 15:0, 16:0, 17:0, 18:0, 19:0, 20:0, 21:0, 22:1, 23:0}},
    {'name': 'Content Creator',                 'answers': {1: 0, 2: 0, 3: 0, 4: 0, 5:0, 6:0, 7:0, 8:0, 9:0, 10:0, 11:0, 12:0, 13:0, 14:0, 15:0, 16:0, 17:0, 18:0, 19:0, 20:0, 21:0, 22:0, 23:1}},
    {'name': 'Engineer',                        'answers': {1: 0, 2: 0, 3: 0, 4: 0, 5:0.75, 6:0.75, 7:0.75, 8:0.75, 9:0.75, 10:0.75, 11:0.5, 12:0.5, 13:0.5, 14:0, 15:0, 16:0, 17:0.5, 18:0, 19:0.75, 20:0.5, 21:0, 22:0.5, 23:0.5}},
    {'name': 'Teacher',                         'answers': {1: 0, 2: 0.5, 3: 0, 4: 0, 5:0, 6:0, 7:0, 8:0, 9:0, 10:0.75, 11:1, 12:0, 13:1, 14:0.5, 15:0.5, 16:0, 17:0.75, 18:0, 19:0.75, 20:0.75, 21:0, 22:0, 23:0.5}},
    {'name': 'MBBS',                            'answers': {1: 0.75, 2: 0.75, 3: 0.75, 4: 0.75, 5:0, 6:0, 7:0, 8:0, 9:0, 10:0, 11:0.5, 12:0, 13:0, 14:0, 15:0, 16:0, 17:0.5, 18:0, 19:0, 20:0, 21:0.5, 22:0, 23:0}},
    {'name': 'Entrepreneur',                    'answers': {1: 0, 2: 0.5, 3:0, 4:0, 5:0.5, 6:0.5, 7:0.5, 8:0.5, 9:0.5, 10:0.5, 11:0.25, 12:0, 13:0.75, 14:0.5, 15:0.5, 16:0, 17:0.75, 18:1, 19:1, 20:0.5, 21:0, 22:0, 23:0.75}},
]

questions_so_far = []
answers_so_far = []



@app.route("/get")
def get_bot_response():
    userText = request.args.get('msg')

    global questions_so_far, answers_so_far

    # question = questions()
    userText = request.args.get('msg')
    question = request.args.get('qnum')
    answer = request.args.get('answer') # get value from userText 
    val = request.args
    print("question from args",question)
    print("usertext",userText)
    print("args",request.args)
    print("type f val",type(val))
    # ans = eval(val)
    print("json",val.items())
    # print("json",ans)
    yesList = ['yes', 'YES', 'Y', 'y']
    noList = ['no','NO','Nope', 'n']
    maybeList = ['maybe','m','probably', 'M']
    maybenotList = ['maybe not','probably not','MN', 'mn']
    dontknowList = ['dont know','Dont know','dk', 'DK','Dk']
    if(userText in yesList):
        answer = 1
    elif(userText in noList):
        answer = 0.01
    elif(userText in maybeList):
        answer = 0.75
    elif(userText in maybenotList):
        answer = 0.25
    elif(userText in dontknowList):
        answer = 0.5
    # elif(userText == 'done' or userText == 'stop'):
    #      break()
    print("question",question)
    print("answer",answer)
    if question and answer:
        questions_so_far.append(int(question))
        answers_so_far.append(float(answer))
        print("questions so far",questions_so_far)
        print("questions so far",answers_so_far)
    
    
    
    probabilities = calculate_probabilites(questions_so_far, answers_so_far)
    print("probabilities", probabilities)

    questions_left = list(set(questions.keys()) - set(questions_so_far))
    print("questioons",questions_left)
    if len(questions_left) == 0:
        result = sorted(
            probabilities, key=lambda p: p['probability'], reverse=True)[0]
        result=result['name']
        return [result]
        # render_template('index.html', result=result['name'])
    else:
        #questio_text
        next_question = random.choice(questions_left)
        question_text=questions[next_question]
        return [question_text ,next_question]
        # render_template('index.html', question=next_question, question_text=questions[next_question])
    
    # return chatbot_response(userText)


def calculate_probabilites(questions_so_far, answers_so_far):
    probabilities = []
    for character in characters:
        probabilities.append({
            'name': character['name'],
            'probability': calculate_character_probability(character, questions_so_far, answers_so_far)
        })

    return probabilities


def calculate_character_probability(character, questions_so_far, answers_so_far):
    # Prior
    P_character = 1 / len(characters)

    # Likelihood
    P_answers_given_character = 1
    P_answers_given_not_character = 1
    for question, answer in zip(questions_so_far, answers_so_far):
        P_answers_given_character *= 1 - \
            abs(answer - character_answer(character, question))

        P_answer_not_character = np.mean([1 - abs(answer - character_answer(not_character, question))
                                          for not_character in characters
                                          if not_character['name'] != character['name']])
        P_answers_given_not_character *= P_answer_not_character

    # Evidence
    P_answers = P_character * P_answers_given_character + \
        (1 - P_character) * P_answers_given_not_character

    # Bayes Theorem
    P_character_given_answers = (
        P_answers_given_character * P_character) / P_answers

    return P_character_given_answers


def character_answer(character, question):
    if question in character['answers']:
        return character['answers'][question]
    return 0.5

if __name__ == "__main__":
    app.debug = False
    port = int(os.environ.get('PORT', 33507))
    serve(app, port=port)
    # app.run(debug=True, use_reloader=False)