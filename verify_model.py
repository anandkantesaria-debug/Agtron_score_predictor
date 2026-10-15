from catboost import CatBoostRegressor
import pandas as pd

# model = CatBoostRegressor()
# model.load_model("models/agtron_model.cbm")

def verify_model():

    model = CatBoostRegressor()
    model.load_model(
        "models/agtron_model.cbm"
    )

    sample = pd.DataFrame({
        "Consumed": [500.0],
        "Temp": [440.7],
        "Quench": [73],
        "Month":["1"],
        "Blend": ["284"]
    })
  
    sample = sample.astype({
        "Blend": "string",
        "Month": "string",
        "Consumed": "float64",
        "Temp": "float64",
        "Quench": "int"
    })
    
    print(sample.dtypes)

    prediction = model.predict(sample)[0]

    print(f"Prediction: {prediction}")

    assert 15 <= prediction <= 65

    print("Verification successful")

if __name__ == "__main__":
    verify_model()