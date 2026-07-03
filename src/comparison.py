import pandas as pd


def save_results(results):

    df = pd.DataFrame(results)

    df = df.sort_values(
        by="R2",
        ascending=False
    )

    df.to_csv(
        "outputs/model_comparison.csv",
        index=False
    )

    print("\n==============================")
    print("Model Comparison")
    print("==============================")

    print(df)

    return df