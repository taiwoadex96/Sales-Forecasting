from flask import Flask, request, render_template
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

app = Flask(__name__)

@app.route('/')
def index():
    # Renders the input configuration data form
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict_datapoint():
    try:
        # 1. Harvest incoming values from form input elements
        data_packet = CustomData(
            store=int(request.form.get('store')),
            date_str=request.form.get('date_str'),
            holiday_flag=int(request.form.get('holiday_flag')),
            temperature=float(request.form.get('temperature')),
            fuel_price=float(request.form.get('fuel_price')),
            cpi=float(request.form.get('cpi')),
            unemployment=float(request.form.get('unemployment')),
            sales_lag_1=float(request.form.get('sales_lag_1')),
            sales_lag_2=float(request.form.get('sales_lag_2')),
            sales_lag_4=float(request.form.get('sales_lag_4')),
            rolling_mean_4=float(request.form.get('rolling_mean_4')),
            rolling_std_4=float(request.form.get('rolling_std_4')),
            ema_4=float(request.form.get('ema_4'))
        )

        # 2. Reconstruct features into an aligned pandas dataframe structure
        final_features_df = data_packet.get_data_as_data_frame()

        # 3. Initialize prediction engine and compute inference forecast
        prediction_pipeline = PredictPipeline()
        raw_prediction = prediction_pipeline.predict(final_features_df)

        # 4. Format array value cleanly into readable currency expression
        formatted_prediction = f"{raw_prediction[0]:,.2f}"

        # 5. Direct results to prediction layout screen
        return render_template('index.html', results=formatted_prediction)

    except Exception as e:
        return f"Operational Server Error encountered: {str(e)}", 500

if __name__ == "__main__":
    # Boot server locally on development port 5000
    app.run(host="0.0.0.0", port=5000, debug=True)