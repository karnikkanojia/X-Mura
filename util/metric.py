from tensorflow_addons.metrics import CohenKappa, F1Score

chn_kappa = CohenKappa(num_classes=2, sparse_labels=True)
f1_score = F1Score(num_classes=1)

def get_metrics():
    return ['accuracy', f1_score, chn_kappa]