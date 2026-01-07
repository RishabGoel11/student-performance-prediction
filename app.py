from flask import Flask, render_template, request
import pickle
import pandas as pd
app = Flask(__name__)

# Load full preprocessing + model pipeline
model = pickle.load(open("model/logistic_pipeline.pkl", "rb"))

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None

    if request.method == "POST":
        try:
            age = int(request.form["age"])
            studytime = int(request.form["studytime"])
            failures = int(request.form["failures"])
            absences = int(request.form["absences"])
            Medu = int(request.form["Medu"])
            Fedu = int(request.form["Fedu"])
            traveltime = int(request.form["traveltime"])
            Dalc = int(request.form["Dalc"])
            Walc = int(request.form["Walc"])
            health = int(request.form["health"])

            input_df = pd.DataFrame([{
                "age": age,
                "studytime": studytime,
                "failures": failures,
                "absences": absences,
                "Medu": Medu,
                "Fedu": Fedu,
                "traveltime": traveltime,
                "Dalc": Dalc,
                "Walc": Walc,
                "health": health
            }])

            result = model.predict(input_df)[0]
            prediction = "PASS ✅" if result == 1 else "FAIL ❌"

        except Exception as e:
            prediction = f"Error: {e}"

    return render_template("index.html", prediction=prediction)



if __name__ == "__main__":
    app.run(debug=False, use_reloader=False)
