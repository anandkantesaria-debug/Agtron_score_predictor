# %%
import pandas as pd
from sklearn.model_selection import train_test_split
from catboost import CatBoostRegressor
from sklearn.metrics import r2_score, mean_absolute_error, root_mean_squared_error
import mlflow
import mlflow.catboost

mlflow.set_tracking_uri("http://127.0.0.1:5000")

from mlflow import MlflowClient

client = MlflowClient()
champion = client.get_model_version_by_alias(
    name="AgtronModel",
    alias="champion"
)

run_id = champion.run_id
run = client.get_run(run_id)
champion_rmse = run.data.metrics["rmse"]
print("Champion RMSE:", champion_rmse)
print("Champion Version:", champion.version)

df = pd.read_csv("Data_time_moisture.csv")

df["Roaster"] = df["Roaster"].astype(str)
df["Blend"] = (
    df["Blend"]
    .astype(int)
    .astype(str)
)
df["Month"] = df["Month"].astype(str)
# %%
#

# %%

X = df[
    [
        "Consumed",
        "Temp",
        "Quench",
        "Month",
        "Blend"
    ]
]
# %%
y_agtron = df["Agtron"]


X_train, X_new, y_train_agtron, y_new_agtron = train_test_split(
    X,
    y_agtron,
    test_size=0.20,
    random_state=28
)
X_val, X_test, y_val_agtron, y_test_agtron = train_test_split(
    X_new,
    y_new_agtron,
    test_size=0.50,
    random_state=28
)

categorical_features = [
    "Month",
    "Blend"
    ]



# X_train, X_val, y_train, y_val already created

# def objective(trial):

#     params = {
#         "task_type": "GPU",
#         "devices": "0",
#         "loss_function": "RMSE",
#         "eval_metric": "MAE",
#         "iterations": trial.suggest_int("iterations", 500, 3000),
#         "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.3, log=True),
#         "depth": trial.suggest_int("depth", 4, 10),

#         "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1, 20),

#         "random_strength": trial.suggest_float(
#             "random_strength", 0.0, 10.0
#         ),

#         "bagging_temperature": trial.suggest_float(
#             "bagging_temperature", 0.0, 10.0
#         ),

#         "border_count": trial.suggest_int(
#             "border_count", 32, 255
#         ),

#         "verbose": 0,
#         "random_seed": 28
#     }

#     model = CatBoostRegressor(**params)

#     model.fit(
#         X_train,
#         y_train_agtron,
#         cat_features=categorical_features,
#         eval_set=(X_val, y_val_agtron),
#         use_best_model=True,
#         early_stopping_rounds=100,
#         verbose=False
#     )

#     preds = model.predict(X_val)

#     mae = mean_absolute_error(y_val_agtron, preds)

#     return mae


# study = optuna.create_study(
#     direction="minimize",
#     study_name="CatBoost_Agtron"
# )

# study.optimize(
#     objective,
#     n_trials=100,
#     show_progress_bar=True
# )

# print("Best MAE:", study.best_value)
# print("Best Parameters:")
# print(study.best_params)

# best_params = study.best_params

# final_model = CatBoostRegressor(
#     **best_params,
#     loss_function="RMSE",
#     eval_metric="MAE",
#     random_seed=28
# )

# final_model.fit(
#     X_train,
#     y_train_agtron,
#     eval_set=(X_val,y_val_agtron),
#     verbose=100
# )


with mlflow.start_run(run_name="Agtron_CatBoost_2026_10_13"):

    agtron_model = CatBoostRegressor(
    iterations=3000,depth=10,learning_rate=0.14295264980157282,loss_function='RMSE',verbose=200)
    mlflow.log_params(agtron_model.get_params())
    agtron_model.fit(X_train, y_train_agtron, cat_features=categorical_features,eval_set=(X_val,y_val_agtron))

    agtron_model.save_model("models/agtron_model.cbm")
    model_info = mlflow.catboost.log_model(cb_model=agtron_model,artifact_path="AgtronModel")
    registered_model =mlflow.register_model(model_uri=model_info.model_uri, name="AgtronModel")
    new_version = registered_model.version
    pred_agtron = agtron_model.predict(X_test)
    r2 = r2_score(y_test_agtron, pred_agtron)
    mae = mean_absolute_error(y_test_agtron, pred_agtron)
    rmse = root_mean_squared_error(y_test_agtron, pred_agtron)
    new_rmse=rmse

    print(f"Champion RMSE: {champion_rmse}")
    print(f"New Model RMSE: {new_rmse}")

    if new_rmse < champion_rmse:
        client.set_registered_model_alias(
            name="AgtronModel",
            alias="champion",
            version=new_version
        )
        print(
            f"Version {new_version} promoted to champion"
        )   
    else:
        print(
            f"Version {champion.version} remains champion"
        )

    mlflow.log_metric("r2", r2)
    mlflow.log_metric("mae", mae)
    mlflow.log_metric("rmse", rmse)

    mlflow.set_tag("Model", "Agtron")
    mlflow.set_tag("Algorithm", "CatBoost")
    mlflow.set_tag("Status", "Production Candidate")


# %%