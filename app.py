from flask import Flask, request, jsonify
from predict_sales import predict_sales, check_stock_sufficiency, create_mysql_connection
from datetime import datetime, timedelta
import pandas as pd
from flask_jwt_extended import (
    JWTManager, create_access_token,
    jwt_required, get_jwt_identity, set_access_cookies,
    unset_jwt_cookies
)

app = Flask(__name__)

# Setup JWT config
app.config["JWT_SECRET_KEY"] = "secret_key"  
app.config["JWT_ACCESS_TOKEN_EXPIRES"] = timedelta(days=30)
jwt = JWTManager(app)

# Static credentials
VALID_CREDENTIALS = {
    "username": "admin",
    "password": "Limer!@#123"
}

@app.route('/login', methods=['POST'])
def login():
    data = request.get_json()
    username = data.get('username')
    password = data.get('password')

    if username == VALID_CREDENTIALS['username'] and password == VALID_CREDENTIALS['password']:
        access_token = create_access_token(identity=username)
        return jsonify({
            'status': 1,
            'message': 'login data',
            'data': {'access_token': access_token}
        }), 200
    else:
        return jsonify({
            'status': 0,
            'message': 'Invalid credentials',
            'data': []
        }), 200

@app.route('/predict', methods=['POST'])
@jwt_required()
def predict():
    current_user = get_jwt_identity()
    try:
        data = request.get_json()
        product_ids = data.get('product_id', [])
        store_id = data.get('store_id')
        start_date_str = data.get('start_date')
        end_date_str = data.get('end_date')

        if not isinstance(product_ids, list):
            product_ids = [product_ids]

        holiday_dates_raw = data.get('filter', {}).get('holiday_dates', [])
        holiday_dates = [datetime.strptime(date, "%Y-%m-%d").date() for date in holiday_dates_raw]

        # If holiday_dates are provided, override start_date and days_to_forecast logic
        if holiday_dates:
            forecast_dates = sorted(holiday_dates)
            days_to_forecast = len(forecast_dates)
            start_date = forecast_dates[0]
            end_date = forecast_dates[-1]
            use_holiday_filter_only = True
        else:
            if not (start_date_str and end_date_str):
                return jsonify({
                    'status': 2,
                    'message': 'Both start_date and end_date are required when holiday_dates are not given.',
                    'data': []
                }), 200

            start_date = datetime.strptime(start_date_str, "%Y-%m-%d").date()
            end_date = datetime.strptime(end_date_str, "%Y-%m-%d").date()

            if start_date > end_date:
                return jsonify({
                    'status': 2,
                    'message': 'start_date must be before end_date.',
                    'data': []
                }), 200

            days_to_forecast = (end_date - start_date).days + 1
            use_holiday_filter_only = False


        days_to_forecast = (end_date - start_date).days + 1

        result = {}
        failed_products = []

        for product_id in product_ids:
            try:

                predictions_df, total_forecasted_quantity, total_forecasted_amount, engine, product_name = predict_sales(
                    product_id, store_id=store_id, days_to_forecast=days_to_forecast, start_date=start_date, holiday_dates=holiday_dates
                )

                print(f"Product ID: {product_id}, Store ID: {store_id}")
                print("Predictions:\n", predictions_df.head())

                if predictions_df.empty:
                    failed_products.append({
                        'product_id': product_id,
                        'product_name': product_name or "Unknown"
                    })
                    continue

                stock_report = check_stock_sufficiency(
                    engine, product_id, store_id, total_forecasted_quantity
                )

                if use_holiday_filter_only:
                    filtered_predictions = predictions_df[
                        pd.to_datetime(predictions_df['date'], dayfirst=True).dt.date.isin(holiday_dates)
                    ]
                else:
                    filtered_predictions = predictions_df[
                        (pd.to_datetime(predictions_df['date'], dayfirst=True).dt.date >= start_date) &
                        (pd.to_datetime(predictions_df['date'], dayfirst=True).dt.date <= end_date)
                    ]

                result[str(product_id)] = {
                    'predictions': filtered_predictions.to_dict(orient='records'),
                    'total_forecasted_quantity': round(float(total_forecasted_quantity), 2),
                    'total_forecasted_amount': round(float(total_forecasted_amount), 2) if total_forecasted_amount is not None else None,
                    'stock_sufficiency': stock_report,
                    'product_name': product_name
                }

            except Exception as e:
                print(f"Error with product {product_id}: {e}")
                failed_products.append({
                    'product_id': product_id,
                    'product_name': "Unknown"
                })

        return jsonify({
            'status': 1,
            'message': 'predictions data' if result else 'No predictions available',
            'data': result,
            'failed_products': failed_products
        })

    except Exception as e:
        return jsonify({
            'status': 0,
            'message': 'Error occurred during prediction',
            'data': [str(e)]
        }), 200

if __name__ == '__main__':
    app.run(debug=True)
