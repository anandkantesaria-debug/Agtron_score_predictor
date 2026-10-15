from catboost import CatBoostRegressor
import pandas as pd


def test_model_prediction():

    model = CatBoostRegressor()
    model.load_model("models/agtron_model.cbm")
    print(model.feature_names_)
    expected_features = [
        "Consumed",
        "Temp",
        "Quench",
        "Month",
        "Blend"        
    ]

    assert model.feature_names_ == expected_features
    assert len(model.feature_names_) == 5

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

    assert prediction is not None
    assert 15 <= prediction <= 65, (f"Prediction {prediction:.1f} outside expected range")

    print("Model test passed")
    print(f"Prediction: {prediction:.1f}")

if __name__ == "__main__":
    test_model_prediction()