from flask import Flask, render_template, request
import pickle
import pandas as pd

app = Flask(__name__)

# Load saved models and encoders
model = pickle.load(open("model.pkl", "rb"))
location_encoder = pickle.load(open("location_encoder.pkl", "rb"))
brand_encoder = pickle.load(open("brand_encoder.pkl", "rb"))
final_columns = pickle.load(open("columns.pkl", "rb"))
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    print('j')
    # Read inputs exactly matching training feature names
    brand = request.form["Brand"]
    location = request.form["Location"]
    fuel = request.form["Fuel_Type"]
    seats = int(request.form["Seats"])
    transmission = request.form["Transmission"]
    owner_type = request.form["Owner_Type"]
    mileage = float(request.form["Mileage"])
    power = float(request.form["Power"])
    engine = int(request.form["Engine"])
    print('jo')
    # Create DataFrame
    input_df = pd.DataFrame([{
        "Brand": brand,
        "Location": location,
        "Fuel_Type": fuel,
        "Seats": seats,
        "Transmission": transmission,
        "Owner_Type": owner_type,
        "Mileage": mileage,
        "Power": power,
        "Engine": engine
    }])
    print("ji")
    # Apply encoders
    input_df[["Location"]] = brand_encoder.transform(input_df[["Location"]])
    input_df[["Brand"]] = location_encoder.transform(input_df[["Brand"]])

    # One-hot encode categorical small columns
    small_cat_cols = ["Fuel_Type", "Seats", "Transmission", "Owner_Type"]
    input_df = pd.get_dummies(input_df, columns=small_cat_cols, drop_first=True)

    # Ensure column order matches training
    input_df = input_df.reindex(columns=final_columns, fill_value=0)

    # Predict
    prediction = model.predict(input_df)[0]
    response_data = request.form.to_dict()
    response_data["Predicted_Price"] = f"around {prediction} ₹"
    print(response_data)
    return render_template("result.html", data=response_data)

if __name__ == "__main__":
    app.run(host="0.0.0.0")
