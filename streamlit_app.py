from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


PROJECT_ROOT = Path(__file__).resolve().parent
DATASET_PATH = PROJECT_ROOT / "Dataset" / "CarPrice_Assignment.csv"
TARGET = "price"
IGNORED_FEATURES = ["car_ID", "CarName"]
FEATURE_LABELS = {
    "fueltype": "Fuel type",
    "aspiration": "Aspiration",
    "doornumber": "Door number",
    "carbody": "Car body",
    "drivewheel": "Drive wheel",
    "enginelocation": "Engine location",
    "wheelbase": "Wheelbase",
    "carlength": "Car length",
    "carwidth": "Car width",
    "carheight": "Car height",
    "curbweight": "Curb weight",
    "enginetype": "Engine type",
    "cylindernumber": "Cylinder number",
    "enginesize": "Engine size",
    "fuelsystem": "Fuel system",
    "boreratio": "Bore ratio",
    "compressionratio": "Compression ratio",
    "peakrpm": "Peak RPM",
    "citympg": "City MPG",
    "highwaympg": "Highway MPG",
}


st.set_page_config(
    page_title="Car price explorer",
    page_icon=":material/directions_car:",
    layout="wide",
)


@st.cache_data
def load_dataset() -> pd.DataFrame:
    if not DATASET_PATH.is_file():
        raise FileNotFoundError(f"Car price dataset not found: {DATASET_PATH}")

    dataset = pd.read_csv(DATASET_PATH)
    if TARGET not in dataset.columns:
        raise ValueError(f"Dataset is missing the required {TARGET!r} column.")
    return dataset.dropna(subset=[TARGET]).copy()


@st.cache_resource
def train_predictor(
    dataset: pd.DataFrame,
) -> tuple[Pipeline, float, float, np.ndarray, np.ndarray]:
    features = dataset.drop(columns=[TARGET, *IGNORED_FEATURES], errors="ignore")
    target = dataset[TARGET].astype(float)
    categorical_features = features.select_dtypes(exclude=np.number).columns.tolist()
    numeric_features = features.select_dtypes(include=np.number).columns.tolist()

    preprocessor = ColumnTransformer(
        [
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                categorical_features,
            ),
            ("numeric", "passthrough", numeric_features),
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        features, target, test_size=0.2, random_state=42
    )
    evaluation_model = Pipeline(
        [
            ("preprocessor", preprocessor),
            (
                "regressor",
                RandomForestRegressor(
                    n_estimators=250,
                    min_samples_leaf=2,
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )
    evaluation_model.fit(X_train, y_train)
    predictions = evaluation_model.predict(X_test)

    final_model = Pipeline(
        [
            ("preprocessor", preprocessor),
            (
                "regressor",
                RandomForestRegressor(
                    n_estimators=250,
                    min_samples_leaf=2,
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )
    final_model.fit(features, target)

    return (
        final_model,
        float(np.sqrt(mean_squared_error(y_test, predictions))),
        float(r2_score(y_test, predictions)),
        y_test.to_numpy(),
        predictions,
    )


dataset = load_dataset()
model, validation_rmse, validation_r2, actual_prices, predicted_prices = train_predictor(
    dataset
)
features = dataset.drop(columns=[TARGET, *IGNORED_FEATURES], errors="ignore")

st.title("Car price explorer")
st.caption("Estimate a car's price and explore the vehicle dataset in one place.")

with st.container(horizontal=True):
    st.metric("Cars in dataset", f"{len(dataset):,}", border=True)
    st.metric("Average listed price", f"${dataset[TARGET].mean():,.0f}", border=True)
    st.metric("Validation R-squared", f"{validation_r2:.2f}", border=True)

prediction_column, charts_column = st.columns([0.9, 1.1], gap="large")

with prediction_column:
    st.subheader("Estimate a price")
    st.caption("Set the specifications, then submit to see the estimate.")

    with st.form("car_price_form"):
        input_values: dict[str, object] = {}
        for feature in features.columns:
            series = features[feature]
            label = FEATURE_LABELS.get(feature, feature.replace("_", " ").capitalize())
            if pd.api.types.is_numeric_dtype(series):
                minimum = float(series.min())
                maximum = float(series.max())
                default = float(series.median())
                if pd.api.types.is_integer_dtype(series):
                    input_values[feature] = int(
                        st.number_input(
                            label,
                            min_value=int(minimum),
                            max_value=int(maximum),
                            value=int(round(default)),
                            step=1,
                            key=f"feature_{feature}",
                        )
                    )
                else:
                    input_values[feature] = st.number_input(
                        label,
                        min_value=minimum,
                        max_value=maximum,
                        value=default,
                        step=0.1,
                        format="%.2f",
                        key=f"feature_{feature}",
                    )
            else:
                options = sorted(series.dropna().unique().tolist())
                default = series.mode().iloc[0]
                input_values[feature] = st.selectbox(
                    label,
                    options,
                    index=options.index(default),
                    key=f"feature_{feature}",
                )

        submitted = st.form_submit_button(
            "Estimate price",
            type="primary",
            icon=":material/calculate:",
            width="stretch",
        )

    if submitted:
        estimate = float(model.predict(pd.DataFrame([input_values]))[0])
        st.success(f"Estimated price: **${estimate:,.0f}**")
        st.caption("This is a data-driven estimate, not a guaranteed market valuation.")

with charts_column:
    st.subheader("Dataset at a glance")
    histogram_counts, bin_edges = np.histogram(dataset[TARGET], bins=16)
    price_distribution = pd.DataFrame(
        {
            "Price range": [
                f"${start:,.0f}-${end:,.0f}"
                for start, end in zip(bin_edges[:-1], bin_edges[1:])
            ],
            "Cars": histogram_counts,
        }
    )
    st.bar_chart(price_distribution, x="Price range", y="Cars")

    median_prices = (
        dataset.groupby("carbody", as_index=False)[TARGET]
        .median()
        .sort_values(TARGET, ascending=False)
        .rename(columns={"carbody": "Car body", TARGET: "Median price"})
    )
    st.bar_chart(median_prices, x="Car body", y="Median price")

    st.subheader("Holdout predictions")
    prediction_comparison = pd.DataFrame(
        {"Actual price": actual_prices, "Predicted price": predicted_prices}
    )
    st.scatter_chart(prediction_comparison, x="Actual price", y="Predicted price")

    st.caption(
        f"Holdout validation: R-squared {validation_r2:.2f} | "
        f"RMSE ${validation_rmse:,.0f}"
    )

st.subheader("Explore price, weight, and horsepower")
scatter = px.scatter_3d(
    dataset,
    x="curbweight",
    y="horsepower",
    z=TARGET,
    color="carbody",
    hover_data=["fueltype", "drivewheel"],
    labels={
        "curbweight": "Curb weight",
        "horsepower": "Horsepower",
        TARGET: "Price",
        "carbody": "Car body",
    },
)
st.plotly_chart(scatter, key="car_price_scatter")

with st.expander("Browse the source data"):
    st.dataframe(dataset.drop(columns=["car_ID"], errors="ignore"), hide_index=True)
