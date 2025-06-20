import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model
from datetime import datetime, timedelta
from sqlalchemy import create_engine

SEQUENCE_LENGTH = 7
MODEL_PATHS = {
    'hybrid_model': 'models/hybrid_sales_model_t.keras',
    'scaler_tab': 'models/scaler_tabular_t.pkl',
    'scaler_seq': 'models/scaler_sequence_t.pkl',
    'scaler_y': 'models/scaler_target_t.pkl'
}

def load_artifacts():
    hybrid_model = load_model(MODEL_PATHS['hybrid_model'])
    scaler_tab = joblib.load(MODEL_PATHS['scaler_tab'])
    scaler_seq = joblib.load(MODEL_PATHS['scaler_seq'])
    scaler_y = joblib.load(MODEL_PATHS['scaler_y'])
    return hybrid_model, scaler_tab, scaler_seq, scaler_y

def create_mysql_connection():
    engine = create_engine(
        "mysql+pymysql://root:Redspark@localhost:3306/updated?charset=utf8mb4",
        pool_pre_ping=True,
        pool_recycle=3600
    )
    return engine

def fetch_product_data(engine, product_id, store_id):
    query = f"""
        SELECT 
            pls.id AS pricelookup_store_id,
            p.id AS pricelookup_id,
            pls.name AS product_name,
            DATE(o.created_at) AS sale_date,
            SUM(d.pricelookup_qty) AS total_quantity_sold,
            AVG(d.pricelookup_item_price) AS average_selling_price,
            AVG(p.standard_price) AS standard_price,
            SUM(d.finalOriginalPrice) AS total_sale_value,
            AVG(o.total_item) AS average_items_in_order
        FROM order_store_detail d
        JOIN orders_store o ON d.order_id = o.id
        JOIN pricelookup p ON d.pricelookup_id = p.id
        JOIN pricelookup_store pls ON pls.pricelookup_id = p.id AND pls.store_id = o.store_id
        WHERE pls.id = {product_id} AND o.store_id = {store_id}
        GROUP BY pls.id, DATE(o.created_at)
        ORDER BY DATE(o.created_at);
    """
    return pd.read_sql(query, engine)

def clean_and_preprocess(df):
    df['sale_date'] = pd.to_datetime(df['sale_date'])

    df['total_quantity_sold'] = df['total_quantity_sold'].fillna(0)
    df['average_selling_price'] = df['average_selling_price'].fillna(df['average_selling_price'].median())
    df['standard_price'] = df['standard_price'].fillna(df['standard_price'].median())
    df['average_items_in_order'] = df['average_items_in_order'].fillna(df['average_items_in_order'].median())

    df['day'] = df['sale_date'].dt.day
    df['month'] = df['sale_date'].dt.month
    df['year'] = df['sale_date'].dt.year
    df['day_of_week'] = df['sale_date'].dt.weekday
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    df['quarter'] = df['sale_date'].dt.quarter
    df['season'] = (df['month'] % 12 // 3 + 1).astype(int)
    df['price_ratio'] = np.where(df['standard_price'] > 0.01, df['average_selling_price'] / df['standard_price'], 1.0)
    df['day_of_year'] = df['sale_date'].dt.dayofyear

    return df

def generate_future_features(start_date, days, product_df, holiday_dates=None):
    medians = {
        'price_ratio': product_df['price_ratio'].median(),
        'average_selling_price': product_df['average_selling_price'].median(),
        'standard_price': product_df['standard_price'].median(),
        'average_items_in_order': product_df['average_items_in_order'].median()
    }

    features = []
    current_date = start_date
    for _ in range(days):
        current_date += timedelta(days=1)
        is_holiday = 1 if holiday_dates and current_date.date() in holiday_dates else 0
        features.append([
            current_date.day,
            current_date.month,
            current_date.year,
            current_date.weekday(),
            1 if current_date.weekday() >= 5 else 0,
            (current_date.month - 1) // 3 + 1,
            (current_date.month % 12) // 3 + 1,
            medians['price_ratio'],
            current_date.timetuple().tm_yday,
            medians['average_selling_price'],
            medians['standard_price'],
            medians['average_items_in_order'],
            is_holiday 
        ])

    columns = [
        'day', 'month', 'year', 'day_of_week', 'is_weekend', 'quarter',
        'season', 'price_ratio', 'day_of_year', 'average_selling_price',
        'standard_price', 'average_items_in_order', 'is_holiday'
    ]

    return pd.DataFrame(features, columns=columns)

def get_standard_price(engine, product_id):
    query = f"""
        SELECT standard_price 
        FROM pricelookup_store 
        WHERE id = {product_id}
        LIMIT 1;
    """
    result = pd.read_sql(query, engine)
    if not result.empty:
        return float(result['standard_price'].iloc[0])
    return None

def predict_sales(product_id, store_id, days_to_forecast=30, start_date=None, holiday_dates=None):
    model, scaler_tab, scaler_seq, scaler_y = load_artifacts()

    engine = create_mysql_connection()
    raw_df = fetch_product_data(engine, product_id, store_id)
    product_df = clean_and_preprocess(raw_df)

    if product_df.empty:
        raise ValueError(f"No data found for product {product_id}")

    product_name = product_df['product_name'].iloc[0]
    standard_price = get_standard_price(engine, product_id)

    sequence = product_df['total_quantity_sold'].values[-SEQUENCE_LENGTH:]
    predictions = []

    if start_date:
        last_date = pd.Timestamp(start_date) - pd.Timedelta(days=1)
    else:
        last_date = max(product_df['sale_date'].max(), pd.Timestamp(datetime.today().date()))

    future_features = generate_future_features(last_date, days_to_forecast, product_df, holiday_dates)

    for day in range(days_to_forecast):
        tab_features = future_features.iloc[day].values.reshape(1, -1)
        tab_features_scaled = scaler_tab.transform(tab_features)

        seq_scaled = scaler_seq.transform(sequence.reshape(-1, 1))
        seq_scaled = seq_scaled.reshape(1, SEQUENCE_LENGTH, 1)

        pred_scaled = model.predict([tab_features_scaled, seq_scaled], verbose=0)
        pred = float(scaler_y.inverse_transform(pred_scaled).flatten()[0])
        pred = max(0, pred)

        predicted_amount = round(pred * standard_price, 2) if standard_price else 0

        predictions.append({
            'predicted_quantity': round(pred, 4),
            'predicted_amount': predicted_amount
        })

        sequence = np.append(sequence[1:], pred)

    prediction_dates = [last_date + timedelta(days=i+1) for i in range(days_to_forecast)]

    prediction_data = []
    for d, p in zip(prediction_dates, predictions):
        is_holiday = 1 if holiday_dates and d.date() in holiday_dates else 0
        is_weekend = 1 if d.weekday() >= 5 else 0
        prediction_data.append({
            'date': d.strftime('%d/%m/%Y'),
            'predicted_quantity': p['predicted_quantity'],
            'predicted_amount': p['predicted_amount'],
            'is_weekend': is_weekend,
            'is_holiday': is_holiday,
            'standard_price':standard_price
        })

    predictions_df = pd.DataFrame(prediction_data)

    total_forecasted_quantity = round(sum(p['predicted_quantity'] for p in predictions), 2)
    total_forecasted_amount = round(sum(p['predicted_amount'] for p in predictions), 2) if standard_price else None

    return predictions_df, total_forecasted_quantity, total_forecasted_amount, engine, product_name

def get_ingredients_for_product(engine, product_id):
    query = f"""
        SELECT p.id AS product_id, p.name AS product_name, 
               i.id AS ingredient_id, i.name AS ingredient_name, 
               pli.qty AS qty_per_product_unit
        FROM pricelookup p
        JOIN pricelookup_ingredient pli ON p.id = pli.pricelookup_id
        JOIN ingredients i ON pli.ingredient_id = i.id
        WHERE p.id = {product_id};
    """
    return pd.read_sql(query, engine)

def get_store_stock_for_product_ingredients(engine, product_id, store_id):
    query = f"""
        SELECT ingr.id AS ingredient_id, ingr.name AS ingredient_name,
               pli.qty AS qty_per_product_unit,
               istd.stockqty, istd.lastweek_left_stockqty,
               uom.conversion_factor
        FROM pricelookup_ingredient pli
        JOIN ingredients ingr ON pli.ingredient_id = ingr.id
        LEFT JOIN ingredient_store istd ON istd.ingredient_id = ingr.id AND istd.store_id = {store_id}
        LEFT JOIN unit_of_measures uom ON CASE WHEN ingr.display_uom_id != 0 THEN ingr.display_uom_id ELSE ingr.master_standard_uom END = uom.id
        WHERE pli.pricelookup_id = {product_id};
    """
    return pd.read_sql(query, engine)

def check_stock_sufficiency(engine, product_id, store_id, total_forecasted_quantity):
    ingredients_df = get_ingredients_for_product(engine, product_id)
    stock_df = get_store_stock_for_product_ingredients(engine, product_id, store_id)

    merged_df = pd.merge(
        ingredients_df[['ingredient_id', 'ingredient_name', 'qty_per_product_unit']],
        stock_df[['ingredient_id', 'ingredient_name', 'qty_per_product_unit', 'stockqty', 'conversion_factor']],
        on=['ingredient_id', 'ingredient_name', 'qty_per_product_unit'],
        how='left'
    )

    merged_df['required_qty'] = merged_df['qty_per_product_unit'] * total_forecasted_quantity
    merged_df['adjusted_stockqty'] = merged_df['stockqty'] * merged_df['conversion_factor']
    merged_df['status'] = np.where(merged_df['adjusted_stockqty'] >= merged_df['required_qty'], 'Sufficient', 'Needs Refill')

    return merged_df[['ingredient_name', 'qty_per_product_unit', 'required_qty', 'adjusted_stockqty', 'status']].to_dict(orient='records')