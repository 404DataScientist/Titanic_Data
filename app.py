from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

# Load trained model
model = joblib.load(r"C:\Users\sarod\Deployment\Titanic\Titanic_Data\titanic_model.pkl")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if request.method == "POST":
        pclass = int(request.form["pclass"])
        sex = request.form["sex"]
        age = float(request.form["age"])
        sibsp = int(request.form["sibsp"])
        parch = int(request.form["parch"])
        fare = float(request.form["fare"])
        embarked = request.form["embarked"]
        who = request.form["who"]
        adult_male = request.form["adult_male"]
        alone = request.form["alone"]

        # Prepare input data matching model's expected columns
        input_data = pd.DataFrame([{
            "pclass": pclass,
            "sex": sex,
            "age": age,
            "sibsp": sibsp,
            "parch": parch,
            "fare": fare,
            "embarked": embarked,
            "class": pclass,  # duplicate for 'class' column
            "who": who,
            "adult_male": adult_male,
            "alone": alone
        }])

        prediction = model.predict(input_data)[0]
        result = "Survived" if prediction == 1 else "Did not survive"

        return render_template("result.html", result=result)

if __name__ == "__main__":
    app.run(debug=True)
