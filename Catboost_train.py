# %%
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from catboost import CatBoostRegressor
from sklearn.metrics import r2_score, mean_absolute_error, root_mean_squared_error
import mlflow
import mlflow.catboost

# Allow overriding MLflow tracking URI via environment variable; defaults to local server
mlflow_uri = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
mlflow.set_tracking_uri(mlflow_uri)

# Path the Streamlit app loads. Only the promoted champion is written here.
SERVING_MODEL_PATH = "models/agtron_model.cbm"

from mlflow import MlflowClient

client = MlflowClient(tracking_uri=mlflow_uri)

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
        iterations=1500,
        depth=6,
        learning_rate=0.08,
        loss_function="RMSE",
        eval_metric="RMSE",
        early_stopping_rounds=50,
        random_seed=28,
        verbose=100
    )
    mlflow.log_params(agtron_model.get_params())
    agtron_model.fit(
        X_train,
        y_train_agtron,
        cat_features=categorical_features,
        eval_set=(X_val, y_val_agtron),
        use_best_model=True
    )

    # 1. Evaluate candidate model on holdout test set
    pred_agtron = agtron_model.predict(X_test)
    r2 = r2_score(y_test_agtron, pred_agtron)
    mae = mean_absolute_error(y_test_agtron, pred_agtron)
    rmse = root_mean_squared_error(y_test_agtron, pred_agtron)
    new_rmse = rmse

    # 2. Log metrics and tags to MLflow run
    mlflow.log_metric("r2", r2)
    mlflow.log_metric("mae", mae)
    mlflow.log_metric("rmse", rmse)

    mlflow.set_tag("Model", "Agtron")
    mlflow.set_tag("Algorithm", "CatBoost")

    # 3. Log model artifact to MLflow
    model_info = mlflow.catboost.log_model(cb_model=agtron_model, artifact_path="AgtronModel")

    # 4. Evaluate existing champion on the exact same test holdout for a fair comparison
    champion = None
    champion_rmse = None
    try:
        champion = client.get_model_version_by_alias(
            name="AgtronModel",
            alias="champion"
        )
        champion_model = mlflow.catboost.load_model(f"models:/{champion.name}@{champion.aliases[0] if champion.aliases else 'champion'}")
        champion_preds = champion_model.predict(X_test)
        champion_rmse = root_mean_squared_error(y_test_agtron, champion_preds)
        print(f"Current Champion Version: {champion.version}")
        print(f"Current Champion RMSE on test set: {champion_rmse:.4f}")
    except Exception as e:
        print(f"No existing champion found or failed to evaluate champion on test set: {e}")

    # 5. Register model in registry only after evaluation
    registered_model = mlflow.register_model(model_uri=model_info.model_uri, name="AgtronModel")
    new_version = registered_model.version
    print(f"New Model (v{new_version}) RMSE on test set: {new_rmse:.4f}")

    # 6. Champion comparison and promotion decision
    if champion_rmse is None or new_rmse < champion_rmse:
        mlflow.set_tag("Status", "Champion")
        client.set_registered_model_alias(
            name="AgtronModel",
            alias="champion",
            version=new_version
        )
        print(
            f"Version {new_version} promoted to champion"
        )
        # Only the champion is written to the path the Streamlit app serves.
        # Save to a temp file first, then atomically swap it in, so the app
        # never loads a partially written model.
        os.makedirs(os.path.dirname(SERVING_MODEL_PATH), exist_ok=True)
        tmp_model_path = SERVING_MODEL_PATH + ".tmp"
        agtron_model.save_model(tmp_model_path)
        os.replace(tmp_model_path, SERVING_MODEL_PATH)
        print(
            f"Champion model saved to {SERVING_MODEL_PATH}"
        )   
    else:
        mlflow.set_tag("Status", "Challenger")
        print(
            f"Version {champion.version} remains champion"
        )


# %%