from catboost import CatBoostRegressor

def test_load_model():

    model = CatBoostRegressor()
    model.load_model("models/agtron_model.cbm")

    assert model is not None