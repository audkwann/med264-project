import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score, mean_squared_error


cancers = ["Lung", "Lymph", "Uterine"]
df = pd.read_csv("californiahealthmaps_alldata/californiahealthmaps_mssa_all.csv")

all_features = ["QNSES","PopAll","PerRural","PerUninsured","PerForeignBorn",
            "PerWhite","PerBlack","PerAsian","PerHispanic","PerDocVisit",
            "PerMenPrev","PerWomenPrev"]
race_features = ["PerWhite","PerBlack","PerAsian","PerHispanic"]
medical_features = ["PerDocVisit","PerMenPrev","PerWomenPrev"]
socioeconomic_features = ["QNSES","PerRural"]
insurance_features = ["PerUninsured"]
foreign_features = ["PerForeignBorn"]
population_features = ["PopAll"]
features = {"All": all_features, 
            "Race": race_features, 
            "Medical": medical_features, 
            "Socioeconomic": socioeconomic_features, 
            "Insurance": insurance_features,
            "Foreign": foreign_features, 
            "Population Count": population_features}

results = []

for cancer in cancers:
    for sex in ["Both", "Male", "Female"]:
        for years in ["05yr", "10yr"]:
            cancer_df = df[(df.Cancer == cancer) & (df.Sex == sex) & (df.Years == years)]
            cancer_df = cancer_df.dropna(subset=["AAIR"])
            if len(cancer_df) < 100:
                continue
            assert cancer_df["AreaID"].is_unique   
            
            for name, feature in features.items():
                if cancer == "Uterine" and name == "Medical":
                    feature = ["PerDocVisit","PerWomenPrev"]
                X, y = cancer_df[feature], cancer_df["AAIR"]

                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=42)

                model = make_pipeline(
                    SimpleImputer(strategy="median"), # fills in any missing feature values wiht the median
                    StandardScaler(), # rescales each feature to have mean 0 and std 1
                    # RidgeCV() # linear regression model
                    RidgeCV(alphas=np.logspace(-3, 3, 50)),
                )
                model.fit(X_train, y_train)

                pred = model.predict(X_test)
                results.append({
                    "cancer": cancer,
                    "sex": sex, 
                    "years": years, 
                    "feature": name,
                    "n_train": len(y_train),
                    "n_test": len(y_test),
                    "test_r2": r2_score(y_test, pred),
                    "test_rmse": mean_squared_error(y_test, pred) ** 0.5,
                    "ridgecv_alpha": model.named_steps["ridgecv"].alpha_, # selected by the RidgeCV algorithm
                })

pd.DataFrame(results).to_csv("lung_lymph_uterine/model_results.csv", index=False)

# Generate heatmap
results = pd.read_csv("lung_lymph_uterine/model_results.csv")
order = ["All", "Race", "Medical", "Socioeconomic", "Insurance", "Foreign", "Population Count"]
results["group"] = results["cancer"] + " | " + results["sex"] + " | " + results["years"]
heat = results.pivot(index="group", columns="feature", values="test_r2")[order]

fig, ax = plt.subplots(figsize=(9, 12))
sns.heatmap(heat, 
            annot=True, 
            fmt=".2f", 
            cmap="RdYlGn", 
            center=0, 
            ax=ax,
            cbar_kws={"label": "Test R²"})
ax.set_xlabel("Feature")
ax.set_ylabel("Cancer Type")
ax.set_title("Test Set R²")
plt.tight_layout()
plt.savefig("lung_lymph_uterine/results_heatmap.png", dpi=200)
