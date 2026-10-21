import pytest
import pandas as pd
import numpy as np

from smartimpute.similarity import SimilarityBasedImputer


def test_similarity_imputer_mixed_types_and_explainability():
    # Dataset with age, gender, salary
    df = pd.DataFrame({
        "age": [25.0, 26.0, 50.0, 52.0, 25.0],
        "gender": ["M", "M", "F", "F", "M"],
        "salary": [50000.0, 52000.0, 120000.0, 125000.0, np.nan],
    })

    imputer = SimilarityBasedImputer(n_neighbors=2, columns=["salary"]).fit(df)
    res = imputer.transform(df)

    # Missing salary at index 4 (young male, age 25) should be close to 50000-52000, not 120000!
    imputed_salary = res.loc[4, "salary"]
    assert 48000.0 <= imputed_salary <= 55000.0

    # Verify explainability records
    assert 4 in imputer.explanations_
    exp = imputer.explanations_[4]["salary"]
    assert "donor_indices" in exp
    assert "reliability_score" in exp
    assert exp["reliability_score"] > 0.5
