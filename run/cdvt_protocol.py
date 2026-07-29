"""Shared frozen protocol values and paper references for CDVT."""


FRAUDGT_PAPER_REFERENCE = {
    "title": (
        "FraudGT: A Simple, Effective, and Efficient Graph Transformer "
        "for Financial Fraud Detection"
    ),
    "doi": "10.1145/3677052.3698648",
    "table": 2,
    "selection_rule": "test F1 at the highest-validation checkpoint",
    "runs": 5,
}

# CDVT's account view uses ports and ego IDs without reverse message passing,
# so PE-FraudGT is the direct parent architecture in the original paper.
PE_FRAUDGT_PAPER = {
    "Small-LI": 0.4581,
    "Small-HI": 0.7641,
    "Medium-LI": 0.4353,
    "Medium-HI": 0.7422,
    "Large-LI": 0.3044,
    "Large-HI": 0.6864,
}

# Multi-FraudGT is the strongest overall FraudGT variant in Table 2 and is
# retained as a stricter published reference, not as CDVT's direct parent.
MULTI_FRAUDGT_PAPER = {
    "Small-LI": 0.4701,
    "Small-HI": 0.7613,
    "Medium-LI": 0.4406,
    "Medium-HI": 0.7593,
    "Large-LI": 0.3743,
    "Large-HI": 0.7334,
}


INITIAL_A2 = {
    "Small-LI": {
        "val_selected_test_f1": 0.46247,
        "raw_best_test_f1": 0.50667,
    },
    "Small-HI": {
        "val_selected_test_f1": 0.77984,
        "raw_best_test_f1": 0.79497,
    },
    "Medium-LI": {
        "val_selected_test_f1": 0.51163,
        "raw_best_test_f1": 0.59031,
    },
    "Medium-HI": {
        "val_selected_test_f1": 0.77574,
        "raw_best_test_f1": 0.78940,
    },
    "Large-LI": {
        "val_selected_test_f1": 0.30108,
        "raw_best_test_f1": 0.44720,
    },
    "Large-HI": {
        "val_selected_test_f1": 0.72897,
        "raw_best_test_f1": 0.76223,
    },
}
