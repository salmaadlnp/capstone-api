from fastapi import FastAPI
import os
import mysql.connector
import pickle
import pandas as pd

main = FastAPI(title="Product Category Prediction API (From MySQL)")

# Load model dan scaler
current_dir = os.path.dirname(__file__)
model_path = os.path.join(current_dir, "WebMinnersbaru.pkl")

with open(model_path, "rb") as f:
    saved_objects = pickle.load(f)
    model = saved_objects['model']
    scaler = saved_objects['scaler']

# Mapping angka ke nama produk
product_mapping = {
    0: 'Air Biasa',
    1: 'Air Mineral (Besar)',
    2: 'Air Mineral (Kecil)',
    3: 'Alvredo Pasta',
    4: 'Americano',
    5: 'Apple Fiz',
    6: 'Apple Tea',
    7: 'Bananas Coffee',
    8: 'Berry Tea',
    9: 'Biscoff Butter',
    10: 'Blackforest',
    11: 'Blue Paradise',
    12: 'Butterscoth Latte',
    13: 'Cake Velvet',
    14: 'Cappucino',
    15: 'Caramel Macchiato',
    16: 'Cheesecake',
    17: 'Cheseecake Latte',
    18: 'Chicken Katsu',
    19: 'Churros',
    20: 'Cireng',
    21: 'Coffee Latte',
    22: 'Cookies N Cream',
    23: 'Croffle Ice Cream',
    24: 'Dark Chocolate',
    25: 'Espresso',
    26: 'French Fries',
    27: 'Fried Rice Erthree',
    28: 'Hazelnut Latte',
    29: 'Japanese Chicken Curry',
    30: 'Javanese',
    31: 'Kebab',
    32: 'Kopi Cream',
    33: 'Lovely',
    34: 'Lumpia',
    35: 'Lychee Tea',
    36: 'Mango Tea',
    37: 'Matcha Cloud',
    38: 'Milo Cream',
    39: 'Mix Platter',
    40: 'Naktamala',
    41: 'Pandan Latte',
    42: 'Peach Tea',
    43: 'Pempek',
    44: 'Piccolo',
    45: 'Pisang Keju',
    46: 'Rice Beef Blackpaper',
    47: 'Rice Beef Bulgogi',
    48: 'Risol Mayo',
    49: 'Roti Bakar',
    50: 'Sea Salted Caramel Latte',
    51: 'Signature Erthree Coffee',
    52: 'Singkong',
    53: 'Spaghetti Bolognese',
    54: 'Spaghetti Brulle',
    55: 'Sweet Honey Karage',
    56: 'Taro',
    57: 'Tiramissyou Latte',
    58: 'Tropical Mango',
    59: 'V60',
    60: 'Vanila Regal',
    61: 'Vietnam Drip',
    62: 'Cheesey Fries Beef',
    63: 'Erthree Toast',
    64: 'Berry Punch',
    65: 'Bundle Package',
    66: 'Butter Sea Salt Latte',
    67: 'Cheessy Fries Beef',
    68: 'Pistachio Latte',
    69: 'Spanish',
    70: 'Big Platter'
}

# Koneksi ke database MySQL
def get_mysql_connection():
    return mysql.connector.connect(
        host="localhost",
        user="root",
        password="",  # sesuaikan kalau kamu pakai password
        database="data_erthree"  # ganti dengan nama database kamu
    )

@main.get("/predict-from-db/{product_id}")
def predict_from_database(product_id: int):
    conn = get_mysql_connection()
    cursor = conn.cursor(dictionary=True)
    
    # Ambil data dari MySQL berdasarkan ID
    cursor.execute("SELECT Product_Name, Product_Price, Quantity, Total, Month, Quantity_Monthly, Day, Year FROM product WHERE id = %s", (product_id,))
    row = cursor.fetchone()

    if not row:
        return {"error": "Produk tidak ditemukan di database"}

    # Ubah ke DataFrame & scaling
    df = pd.DataFrame([row])
    df_scaled = scaler.transform(df)
    df_scaled = pd.DataFrame(df_scaled, columns=df.columns)

    # Prediksi
    prediction = model.predict(df_scaled)[0]
    label_map = {0: "Sedikit", 1: "Sedang", 2: "Banyak"}
    product_name_real = product_mapping.get(row['Product_Name'], "Unknown Product")

    return {
        "product": product_name_real,
        "predicted_category": label_map.get(prediction, "Unknown")
    }
