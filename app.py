from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import pandas as pd

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load full preprocessing + model pipeline
model = pickle.load(open("model/logistic_pipeline.pkl", "rb"))


@app.route("/predict", methods=["POST"])
def predict():
    """API endpoint for student performance prediction"""
    try:
        data = request.get_json()
        
        # Extract input features
        age = int(data.get("age"))
        studytime = int(data.get("studytime"))
        failures = int(data.get("failures"))
        absences = int(data.get("absences"))
        Medu = int(data.get("Medu"))
        Fedu = int(data.get("Fedu"))
        traveltime = int(data.get("traveltime"))
        Dalc = int(data.get("Dalc"))
        Walc = int(data.get("Walc"))
        health = int(data.get("health"))

        # Create input dataframe
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
        
        # Make prediction
        result = int(model.predict(input_df)[0])
        prediction = "PASS" if result == 1 else "FAIL"
        
        return jsonify({
            "success": True,
            "prediction": prediction,
            "status": result
        }), 200

    except KeyError as e:
        return jsonify({
            "success": False,
            "error": f"Missing required field: {str(e)}"
        }), 400
    except ValueError as e:
        return jsonify({
            "success": False,
            "error": f"Invalid input value: {str(e)}"
        }), 400
    except Exception as e:
        return jsonify({
            "success": False,
            "error": f"Error: {str(e)}"
        }), 500


@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint"""
    return jsonify({"status": "healthy"}), 200


if __name__ == "__main__":
    app.run(debug=False, use_reloader=False, host="0.0.0.0", port=5000)