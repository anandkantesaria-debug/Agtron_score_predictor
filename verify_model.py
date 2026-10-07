from catboost import CatBoostRegressor

model = CatBoostRegressor()
model.load_model("models/agtron_model.cbm")

print("Model loaded successfully")