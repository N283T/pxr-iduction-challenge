-- Experiment tracking schema

CREATE TABLE experiments (
    id SERIAL PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,
    description TEXT,
    model_type TEXT NOT NULL,            -- e.g. 'lightgbm', 'chemprop', 'xgboost'
    feature_set TEXT NOT NULL,           -- e.g. 'rdkit_descriptors', 'morgan_fp', 'gnn'
    hyperparameters JSONB,               -- full config as JSON
    cv_folds INTEGER DEFAULT 5,
    submission_path TEXT,                 -- relative path to submission file
    created_at TIMESTAMPTZ DEFAULT now(),
    notes TEXT
);

CREATE TABLE experiment_cv_results (
    id SERIAL PRIMARY KEY,
    experiment_id INTEGER NOT NULL REFERENCES experiments(id) ON DELETE CASCADE,
    fold INTEGER NOT NULL,
    mae DOUBLE PRECISION,
    rae DOUBLE PRECISION,
    r2 DOUBLE PRECISION,
    spearman_r DOUBLE PRECISION,
    kendall_tau DOUBLE PRECISION,
    num_boost_rounds INTEGER,            -- or epochs for NN
    UNIQUE(experiment_id, fold)
);

CREATE INDEX idx_cv_results_experiment ON experiment_cv_results(experiment_id);

-- View for easy experiment comparison
CREATE VIEW experiment_summary AS
SELECT
    e.id,
    e.name,
    e.model_type,
    e.feature_set,
    e.cv_folds,
    round(avg(cv.rae)::numeric, 4) AS rae_mean,
    round(stddev(cv.rae)::numeric, 4) AS rae_std,
    round(avg(cv.mae)::numeric, 4) AS mae_mean,
    round(avg(cv.r2)::numeric, 4) AS r2_mean,
    round(avg(cv.spearman_r)::numeric, 4) AS spearman_mean,
    round(avg(cv.kendall_tau)::numeric, 4) AS kendall_mean,
    e.submission_path,
    e.created_at
FROM experiments e
LEFT JOIN experiment_cv_results cv ON cv.experiment_id = e.id
GROUP BY e.id
ORDER BY rae_mean ASC NULLS LAST;

-- OOF predictions for ensemble weight optimization
CREATE TABLE IF NOT EXISTS experiment_oof_predictions (
    experiment_id INTEGER NOT NULL REFERENCES experiments(id) ON DELETE CASCADE,
    train_idx INTEGER NOT NULL,
    oof_prediction DOUBLE PRECISION NOT NULL,
    PRIMARY KEY (experiment_id, train_idx)
);

CREATE INDEX IF NOT EXISTS idx_oof_experiment ON experiment_oof_predictions(experiment_id);
