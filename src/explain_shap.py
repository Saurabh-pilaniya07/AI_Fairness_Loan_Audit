import shap
import matplotlib.pyplot as plt

def shap_analysis(model, X_test):

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    # Handle classification output
    if isinstance(shap_values, list):
        shap_values_to_plot = shap_values[1]
    else:
        shap_values_to_plot = shap_values[:, :, 1]

    # Save SHAP plot
    shap.summary_plot(shap_values_to_plot, X_test, show=False)
    plt.savefig("outputs/shap_summary.png")
    plt.close()

    print("SHAP plot saved.")

    return shap_values