import os
import pickle

from flask import Flask, jsonify, request

app = Flask(__name__)

url = ""

@app.route("/", methods=["GET"])
def index():
    return_string = "<h1>NBA Salary Predictor</h1>" + "\n" + "<h2>Estimate an NBA player's average career salary based on PER, games, and win shares</h2>"  + "\n" + "<h3>Here is a sample URL with a query string for the player Kendrick Perkins that you can use to test the classifier: https://nba-salary-app-adrian-michael.herokuapp.com/predict?name=Kendrick%20Perkins&per=10.7&games=782&ws=27.9</h3> " + "\n" + "<h3>Please replace the values in the URL with those of your choosing to test this classifier further</h3>"
    return return_string, 200
@app.route("/predict", methods=["get"])
def predict():
    name = per = request.args.get("name", "")
    per = request.args.get("per", "")
    games = request.args.get("games", "")
    ws = request.args.get("ws", "")
    per = float(per)
    games = float(games)
    ws = float(ws)
    per = categorical_per(per)
    games = categorical_games(games)
    ws = categorical_ws(ws)
    prediction = predict_interviews_well([per, games, ws])
    if prediction is not None: 
        salary_range = get_salary_range(prediction)
        pred_string = name + salary_range 
        result = {"prediction": pred_string}
        return jsonify(result), 200
    else: 
        return "Error making prediction", 400
def tdidt_predict(header, tree, instance):
    info_type = tree[0]
    if info_type == "Attribute":
        attribute_index = header.index(tree[1])
        instance_value = instance[attribute_index]
        for i in range(2, len(tree)):
            value_list = tree[i]
            if value_list[1] == instance_value: 
                return tdidt_predict(header, value_list[2], instance)
    else: 
        return tree[1]

def predict_interviews_well(instance):
    infile = open("web_app_deployment/tree.p", "rb")
    header, tree = pickle.load(infile)
    infile.close()
    print("header:", header)
    print("tree:", tree)
    try: 
        return tdidt_predict(header, tree, instance)
    except: 
        return None 
def get_salary_range(prediction):
    if prediction == 1:
        return "'s salary is under $1,609,676.10 per year."
    elif prediction == 2:
        return "'s salary is between $1,609,676.10 and $3,149,352.20 per year."
    elif prediction == 3: 
        return "'s salary is between $3,149,352.20 and $4,689,028.30 per year."
    elif prediction == 4: 
        return "'s salary is between $4,689,028.30 and $6,228,704.40 per year."
    elif prediction == 5: 
        return "'s salary is between $6,228,704.40 and $7,768,380.50 per year."
    elif prediction == 6: 
        return "'s salary is between $7,768,380.50 and $9,308,056.60 per year."
    elif prediction == 7: 
        return "'s salary is between $9,308,056.60 and $10,847,732.70 per year."
    elif prediction == 8: 
        return "'s salary is between $10,847,732.70 and $12,387,408.80 per year."
    elif prediction == 9: 
        return "'s salary is between $12,387,408.80 and $13,927,084.90 per year."
    elif prediction == 10: 
        return "'s salary is over $13,927,084.90 per year."
    return ""
def categorical_games(games):
    if games < 296.6:
        return 1
    elif games >= 296.6 and games < 428.2:
        return 2
    elif games >= 428.2 and games < 559.8:
        return 3
    elif games >= 559.8 and games < 691.4:
        return 4
    elif games >= 691.4 and games < 823.0:
        return 5
    elif games >= 823.0 and games < 954.6:
        return 6
    elif games >= 954.6 and games < 1086.2:
        return 7
    elif games >= 1086.2 and games < 1217.8:
        return 8
    elif games >= 1217.8 and games < 1349.4:
        return 9
    elif games > 1349.4:
        return 10
    return 0
def categorical_per(per):
    if per < 11.08:
        return 1
    elif per >= 11.08 and per < 15.16:
        return 2
    elif per >= 15.16 and per < 19.24:
        return 3
    elif per >= 19.24 and per < 23.32:
        return 4
    elif per > 23.32:
        return 5
    return 0
def categorical_ws(ws):
    if ws < 45.64:
        return 1
    elif ws >= 45.64 and ws < 92.88:
        return 2
    elif ws >= 92.88 and ws < 140.12:
        return 3
    elif ws >= 140.12 and ws < 187.36:
        return 4
    elif ws > 187.36:
        return 5
    return 0
if __name__ == "__main__":
    port = os.environ.get("PORT", 5000)
    app.run(debug=False, host="0.0.0.0", port=port) #set debug to false for production