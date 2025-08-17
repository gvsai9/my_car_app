from flask import Flask, render_template, request
import pickle
import pandas as pd
from datetime import datetime
current_year = datetime.now().year
application = Flask(__name__)

# Load saved models and encoders
model = pickle.load(open("model.pkl", "rb"))
location_encoder = pickle.load(open("location_encoder.pkl", "rb"))
brand_encoder = pickle.load(open("brand_encoder.pkl", "rb"))
final_columns = pickle.load(open("columns.pkl", "rb"))
@application.route("/")
def home():
    return render_template("index.html")

@application.route("/predict", methods=["POST"])
def predict():
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
    year=int(request.form["Year"])
    # Create DataFrame
    input_df = pd.DataFrame([{
        "Brand": brand,
        "Location": location,
        "Fuel_Type": fuel,
        "Seats": seats,
        "Transmission": transmission,
        "Owner_Type": owner_type,
        "Mileage_numeric": mileage,
        "Power_numeric": power,
        "Engine_numeric": engine,
        "Year":year
    }])
    # Apply encoders
    input_df[["Location"]] = location_encoder.transform(input_df[["Location"]])
    input_df[["Brand"]] =    brand_encoder.transform(input_df[["Brand"]])
    input_df['Car_Age'] = current_year - input_df['Year']  # replace 2025 with current year if dynamic
    input_df['Power_per_Engine'] = input_df['Power_numeric'] / input_df['Engine_numeric']
    input_df['Mileage_per_Year'] = input_df['Mileage_numeric'] / (input_df['Car_Age'] + 1) 
    # One-hot encode categorical small columns
    input_df = input_df.drop(columns=['Year'] , errors='ignore')

    small_cat_cols = ["Fuel_Type", "Seats", "Transmission", "Owner_Type"]
    input_df = pd.get_dummies(input_df, columns=small_cat_cols, drop_first=False)

    # Ensure column order matches training
    input_df = input_df.reindex(columns=final_columns, fill_value=0)

    # Predict
    prediction = model.predict(input_df)[0]
    response_data = request.form.to_dict()
    response_data["Predicted_Price"] = prediction
    print(response_data)
    return render_template("result.html", data=response_data)

if __name__ == "__main__":
    application.run(host="0.0.0.0",debug=True)
