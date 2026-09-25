"""LightGBM pair classifier (MIT)."""
import lightgbm as lgb


def train(X_tr, y_tr, X_va, y_va, params, rounds=500, early=50):
    dtr = lgb.Dataset(X_tr, label=y_tr)
    dva = lgb.Dataset(X_va, label=y_va, reference=dtr)
    model = lgb.train(params, dtr, num_boost_round=rounds, valid_sets=[dva],
                      callbacks=[lgb.early_stopping(early, verbose=False)])
    return model


def predict(model, X):
    return model.predict(X)
