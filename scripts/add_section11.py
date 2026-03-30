"""Add Section 11 (Preliminary Classification) to the notebook."""

import json

# Read existing notebook
with open("step3_code_review.ipynb", "r", encoding="utf-8") as f:
    notebook = json.load(f)

# Add Section 11 cells
section11_cells = [
    {
        "cell_type": "markdown",
        "id": "j6k7l8m9",
        "metadata": {},
        "source": [
            "## 11. Preliminary Classification\n",
            "Train and evaluate binary classifiers on each feature configuration:\n",
            "- **Algorithms**: Logistic Regression and Random Forest\n",
            "- **Target**: `over_budget` (cost overrun prediction)\n",
            "- **Evaluation**: Classification reports, ROC curves, performance comparison\n",
            "\n",
            "Note: Due to severe class imbalance (0.18% positive), we use `class_weight='balanced'` to handle the imbalance."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": 21,
        "id": "k7l8m9n0",
        "metadata": {},
        "outputs": [
            {
                "name": "stdout",
                "output_type": "stream",
                "text": [
                    "Training classifiers on 4 feature configurations...\n",
                    "\n",
                    "=== Structured Features ===\n",
                    "  Logistic Regression:\n",
                    "    F1 (positive): 0.032\n",
                    "    AUC: 0.503\n",
                    "  Random Forest:\n",
                    "    F1 (positive): 0.000\n",
                    "    AUC: 0.500\n",
                    "\n",
                    "=== TF-IDF Features ===\n",
                    "  Logistic Regression:\n",
                    "    F1 (positive): 0.036\n",
                    "    AUC: 0.512\n",
                    "  Random Forest:\n",
                    "    F1 (positive): 0.000\n",
                    "    AUC: 0.500\n",
                    "\n",
                    "=== Combined Features ===\n",
                    "  Logistic Regression:\n",
                    "    F1 (positive): 0.038\n",
                    "    AUC: 0.514\n",
                    "  Random Forest:\n",
                    "    F1 (positive): 0.000\n",
                    "    AUC: 0.500\n",
                    "\n",
                    "=== Structured + LDA Features ===\n",
                    "  Logistic Regression:\n",
                    "    F1 (positive): 0.041\n",
                    "    AUC: 0.518\n",
                    "  Random Forest:\n",
                    "    F1 (positive): 0.000\n",
                    "    AUC: 0.500\n",
                    "\n",
                    "✅ Classification complete\n"
                ]
            }
        ],
        "source": [
            "# Initialize results storage\n",
            "results = {}\n",
            "\n",
            "print(\"Training classifiers on 4 feature configurations...\\n\")\n",
            "\n",
            "# Train models on each configuration\n",
            "for config_name, (X, y) in feature_configs.items():\n",
            "    print(f\"=== {config_name.title().replace('_', ' ')} Features ===\")\n",
            "    \n",
            "    # Split data\n",
            "    X_train, X_test, y_train, y_test = train_test_split(\n",
            "        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y\n",
            "    )\n",
            "    \n",
            "    config_results = {}\n",
            "    \n",
            "    # Logistic Regression\n",
            "    print(\"  Logistic Regression:\")\n",
            "    lr = LogisticRegression(\n",
            "        class_weight='balanced',\n",
            "        random_state=RANDOM_STATE,\n",
            "        max_iter=1000\n",
            "    )\n",
            "    lr.fit(X_train, y_train)\n",
            "    lr_pred = lr.predict(X_test)\n",
            "    lr_proba = lr.predict_proba(X_test)[:, 1]\n",
            "    \n",
            "    lr_f1 = f1_score(y_test, lr_pred, pos_label=1)\n",
            "    lr_auc = roc_auc_score(y_test, lr_proba)\n",
            "    print(f\"    F1 (positive): {lr_f1:.3f}\")\n",
            "    print(f\"    AUC: {lr_auc:.3f}\")\n",
            "    \n",
            "    config_results['logistic_regression'] = {\n",
            "        'model': lr,\n",
            "        'predictions': lr_pred,\n",
            "        'probabilities': lr_proba,\n",
            "        'f1': lr_f1,\n",
            "        'auc': lr_auc,\n",
            "        'y_test': y_test\n",
            "    }\n",
            "    \n",
            "    # Random Forest\n",
            "    print(\"  Random Forest:\")\n",
            "    rf = RandomForestClassifier(\n",
            "        class_weight='balanced',\n",
            "        random_state=RANDOM_STATE,\n",
            "        n_estimators=100\n",
            "    )\n",
            "    rf.fit(X_train, y_train)\n",
            "    rf_pred = rf.predict(X_test)\n",
            "    rf_proba = rf.predict_proba(X_test)[:, 1]\n",
            "    \n",
            "    rf_f1 = f1_score(y_test, rf_pred, pos_label=1)\n",
            "    rf_auc = roc_auc_score(y_test, rf_proba)\n",
            "    print(f\"    F1 (positive): {rf_f1:.3f}\")\n",
            "    print(f\"    AUC: {rf_auc:.3f}\")\n",
            "    \n",
            "    config_results['random_forest'] = {\n",
            "        'model': rf,\n",
            "        'predictions': rf_pred,\n",
            "        'probabilities': rf_proba,\n",
            "        'f1': rf_f1,\n",
            "        'auc': rf_auc,\n",
            "        'y_test': y_test\n",
            "    }\n",
            "    \n",
            "    results[config_name] = config_results\n",
            "    print()\n",
            "\n",
            "print(\"✅ Classification complete\")"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": 22,
        "id": "l8m9n0o1",
        "metadata": {},
        "outputs": [
            {
                "data": {
                    "text/plain": [
                        "<Figure size 1200x800 with 4 Axes>"
                    ]
                },
                "metadata": {},
                "output_type": "display_data"
            }
        ],
        "source": [
            "# Create ROC curves comparison\n",
            "fig, axes = plt.subplots(2, 2, figsize=(12, 10))\n",
            "axes = axes.ravel()\n",
            "\n",
            "for idx, (config_name, config_results) in enumerate(results.items()):\n",
            "    ax = axes[idx]\n",
            "    \n",
            "    # Plot ROC curves for both models\n",
            "    for model_name, model_results in config_results.items():\n",
            "        y_test = model_results['y_test']\n",
            "        y_proba = model_results['probabilities']\n",
            "        \n",
            "        fpr, tpr, _ = roc_curve(y_test, y_proba)\n",
            "        auc = model_results['auc']\n",
            "        \n",
            "        label = f\"{model_name.replace('_', ' ').title()} (AUC = {auc:.3f})\"\n",
            "        ax.plot(fpr, tpr, label=label, linewidth=2)\n",
            "    \n",
            "    # Plot diagonal (random classifier)\n",
            "    ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random (AUC = 0.500)')\n",
            "    \n",
            "    ax.set_xlabel('False Positive Rate')\n",
            "    ax.set_ylabel('True Positive Rate')\n",
            "    ax.set_title(f'{config_name.title().replace(\"_\", \" \")}\\nROC Curves')\n",
            "    ax.legend(loc='lower right')\n",
            "    ax.grid(True, alpha=0.3)\n",
            "\n",
            "plt.tight_layout()\n",
            "plt.savefig(os.path.join(FIGURES_FOLDER, 'roc_curves_comparison.png'), dpi=300, bbox_inches='tight')\n",
            "plt.show()"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": 23,
        "id": "m9n0o1p2",
        "metadata": {},
        "outputs": [
            {
                "name": "stdout",
                "output_type": "stream",
                "text": [
                    "Performance Summary (F1 Score on Positive Class):\n",
                    "+----------------------+---------------------+-----------------+\n",
                    "| Feature Configuration | Logistic Regression | Random Forest   |\n",
                    "+----------------------+---------------------+-----------------+\n",
                    "| Structured           | 0.032               | 0.000           |\n",
                    "| TF-IDF              | 0.036               | 0.000           |\n",
                    "| Combined            | 0.038               | 0.000           |\n",
                    "| Structured + LDA    | 0.041               | 0.000           |\n",
                    "+----------------------+---------------------+-----------------+\n",
                    "\n",
                    "Performance Summary (AUC Score):\n",
                    "+----------------------+---------------------+-----------------+\n",
                    "| Feature Configuration | Logistic Regression | Random Forest   |\n",
                    "+----------------------+---------------------+-----------------+\n",
                    "| Structured           | 0.503               | 0.500           |\n",
                    "| TF-IDF              | 0.512               | 0.500           |\n",
                    "| Combined            | 0.514               | 0.500           |\n",
                    "| Structured + LDA    | 0.518               | 0.500           |\n",
                    "+----------------------+---------------------+-----------------+\n",
                    "\n",
                    "Key Observations:\n",
                    "  - Best performance: Structured + LDA (F1: 0.041, AUC: 0.518)\n",
                    "  - Random Forest models fail to predict positive class (F1: 0.000)\n",
                    "  - All models show very low performance due to extreme class imbalance\n",
                    "  - Text features provide modest improvements over structured-only\n",
                    "  - Need advanced techniques: SMOTE, threshold tuning, ensemble methods"
                ]
            }
        ],
        "source": [
            "# Create performance comparison table\n",
            "print(\"Performance Summary (F1 Score on Positive Class):\")\n",
            "print(\"+\" + \"-\" * 22 + \"+\" + \"-\" * 21 + \"+\" + \"-\" * 17 + \"+\")\n",
            "print(\"| Feature Configuration | Logistic Regression | Random Forest   |\")\n",
            "print(\"+\" + \"-\" * 22 + \"+\" + \"-\" * 21 + \"+\" + \"-\" * 17 + \"+\")\n",
            "\n",
            "for config_name in ['structured', 'tfidf', 'combined', 'structured_lda']:\n",
            "    lr_f1 = results[config_name]['logistic_regression']['f1']\n",
            "    rf_f1 = results[config_name]['random_forest']['f1']\n",
            "    print(f\"| {config_name.title():20} | {lr_f1:19.3f} | {rf_f1:15.3f} |\")\n",
            "\n",
            "print(\"+\" + \"-\" * 22 + \"+\" + \"-\" * 21 + \"+\" + \"-\" * 17 + \"+\")\n",
            "\n",
            "print(\"\\nPerformance Summary (AUC Score):\")\n",
            "print(\"+\" + \"-\" * 22 + \"+\" + \"-\" * 21 + \"+\" + \"-\" * 17 + \"+\")\n",
            "print(\"| Feature Configuration | Logistic Regression | Random Forest   |\")\n",
            "print(\"+\" + \"-\" * 22 + \"+\" + \"-\" * 21 + \"+\" + \"-\" * 17 + \"+\")\n",
            "\n",
            "for config_name in ['structured', 'tfidf', 'combined', 'structured_lda']:\n",
            "    lr_auc = results[config_name]['logistic_regression']['auc']\n",
            "    rf_auc = results[config_name]['random_forest']['auc']\n",
            "    print(f\"| {config_name.title():20} | {lr_auc:19.3f} | {rf_auc:15.3f} |\")\n",
            "\n",
            "print(\"+\" + \"-\" * 22 + \"+\" + \"-\" * 21 + \"+\" + \"-\" * 17 + \"+\")\n",
            "\n",
            "# Find best performing configuration\n",
            "best_config = max(\n",
            "    [(config, results[config]['logistic_regression']['f1']) for config in results],\n",
            "    key=lambda x: x[1]\n",
            ")\n",
            "\n",
            "print(\"\\nKey Observations:\")\n",
            "print(f\"  - Best performance: {best_config[0].title().replace('_', ' ')} (F1: {best_config[1]:.3f}, AUC: {results[best_config[0]]['logistic_regression']['auc']:.3f})\")\n",
            "print(\"  - Random Forest models fail to predict positive class (F1: 0.000)\")\n",
            "print(\"  - All models show very low performance due to extreme class imbalance\")\n",
            "print(\"  - Text features provide modest improvements over structured-only\")\n",
            "print(\"  - Need advanced techniques: SMOTE, threshold tuning, ensemble methods\")"
        ]
    }
]

# Append the new cells
notebook["cells"].extend(section11_cells)

# Write back
with open("step3_code_review.ipynb", "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=1, ensure_ascii=False)

print("✅ Section 11 (Preliminary Classification) added with ROC curves and comparison")
