"""Add Section 12 (Next Steps) to the notebook."""

import json

# Read existing notebook
with open("step3_code_review.ipynb", "r", encoding="utf-8") as f:
    notebook = json.load(f)

# Add Section 12 cells
section12_cells = [
    {
        "cell_type": "markdown",
        "id": "n0o1p2q3",
        "metadata": {},
        "source": [
            "## 12. Next Steps\n",
            "\n",
            "### Current Status\n",
            "We have successfully built and evaluated a complete data pipeline for predicting federal contract cost overruns using a 50K-contract sample. The pipeline includes:\n",
            "\n",
            "- **Data Processing**: PIID-group sampling, physical deliverable filtering, outcome label construction with adaptive thresholds\n",
            "- **Exploratory Analysis**: 9 visualizations showing class imbalance, feature distributions, and correlations\n",
            "- **Text Processing**: Two-track NLP approach (LDA for long descriptions, TF-IDF for all descriptions)\n",
            "- **Feature Engineering**: 4 feature configurations combining structured and text features\n",
            "- **Classification**: Logistic Regression and Random Forest with balanced class weights\n",
            "\n",
            "### Key Findings\n",
            "\n",
            "1. **Severe Class Imbalance**: Only 0.18% of contracts experience cost overruns at the 1% threshold\n",
            "2. **Text Feature Value**: Structured+LDA performs best (F1: 0.041, AUC: 0.518), showing text features add predictive power\n",
            "3. **Model Limitations**: Random Forest fails to predict positive class; all models need improvement\n",
            "\n",
            "### Immediate Next Steps\n",
            "\n",
            "1. **Address Class Imbalance**\n",
            "   - Implement SMOTE or ADASYN for oversampling the minority class\n",
            "   - Experiment with different classification thresholds\n",
            "   - Try focal loss or cost-sensitive learning\n",
            "\n",
            "2. **Model Enhancement**\n",
            "   - Hyperparameter tuning with cross-validation\n",
            "   - Ensemble methods (XGBoost, LightGBM)\n",
            "   - Neural network approaches for text features\n",
            "\n",
            "3. **Feature Engineering**\n",
            "   - Tune LDA parameters (number of topics, passes)\n",
            "   - Add more structured features (contract duration, vendor history)\n",
            "   - Experiment with word embeddings (BERT, Word2Vec)\n",
            "\n",
            "4. **Expand Target Variables**\n",
            "   - Predict schedule delays (`late` target)\n",
            "   - Multi-class classification for overrun severity levels\n",
            "   - Regression for continuous cost growth percentages\n",
            "\n",
            "### Long-term Goals\n",
            "\n",
            "1. **Full Dataset Analysis**: Scale to the complete 3.88M contract dataset\n",
            "2. **Temporal Validation**: Test on different time periods to ensure temporal generalization\n",
            "3. **Feature Importance**: Generate interpretable insights for contract management\n",
            "4. **Deployment**: Create a prediction tool for contract risk assessment\n",
            "\n",
            "### Technical Improvements\n",
            "\n",
            "- **Pipeline Optimization**: Parallel processing for faster execution\n",
            "- **Memory Management**: Sparse matrices and chunked processing for large datasets\n",
            "- **Model Interpretability**: SHAP values for feature importance analysis\n",
            "- **Error Analysis**: Investigate false positives/negatives for insights\n",
            "\n",
            "---\n",
            "\n",
            "**Note**: This notebook represents a comprehensive foundation for federal contract outcome prediction. The modular structure allows for easy extension and experimentation with advanced techniques as outlined above."
        ]
    }
]

# Append the new cells
notebook["cells"].extend(section12_cells)

# Write back
with open("step3_code_review.ipynb", "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=1, ensure_ascii=False)

print("✅ Section 12 (Next Steps) added - Notebook complete!")
