# -*- coding: utf-8 -*-

"""
@created: 09/2023
@updated:
@author: quyen@marcaud.fr
"""
import pandas as pd
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from hyperopt.early_stop import no_progress_loss
from sklearn.metrics import accuracy_score
from typing import Any, Dict, Union, Sequence

from sklearn.model_selection import StratifiedKFold, cross_val_score


class HyperOpt:
    def __init__(self, estimator):
        self.estimator = estimator

    def __hyperparameter_tuning(self,
                                param_space: Dict[str, Union[float, int]],
                                X_train: pd.DataFrame, y_train: pd.Series,
                                X_test: pd.DataFrame, y_test: pd.Series,
                                metric: callable = accuracy_score) -> Dict[str, Any]:
        self.estimator.set_params(**param_space)
        self.estimator.fit(X_train, y_train)
        y_pred = self.estimator.predict(X_test)

        metric_eval = metric(y_test, y_pred)
        return {'loss': -metric_eval, 'status': STATUS_OK, 'model': self.estimator}

    def __hyperparameter_tuning_with_cv(self,
                                        param_space: Dict[str, Union[float, int]],
                                        X_train: pd.DataFrame, y_train: pd.Series,
                                        metric: str = 'accuracy') -> Dict[str, Any]:
        self.estimator.set_params(**param_space)
        cv = StratifiedKFold(n_splits=5, random_state=42, shuffle=True)

        score = cross_val_score(self.estimator, X=X_train, y=y_train, cv=cv, scoring=metric).mean()

        return {'loss': -score, 'status': STATUS_OK, 'model': self.estimator}

    def __hyperparameter_tuning_with_early_stopping(self,
                                                    param_space: Dict[str, Union[float, int]],
                                                    X_train: pd.DataFrame, y_train: pd.Series,
                                                    X_test: pd.DataFrame, y_test: pd.Series,
                                                    early_stopping_rounds: int = 50,
                                                    metric: str = 'accuracy') -> Dict[str, Any]:
        param_space['early_stopping_rounds'] = early_stopping_rounds
        self.estimator.set_params(**param_space)
        evaluation = [(X_train, y_train),
                      (X_test, y_test)]
        cv = StratifiedKFold(n_splits=5, random_state=42, shuffle=True)
        self.estimator.fit(X_train, y_train,
                           eval_set=evaluation,
                           cv=cv,
                           verbose=False)
        y_pred = self.estimator.predict(X_test)

        metric_eval = metric(y_test, y_pred)
        return {'loss': -metric_eval, 'status': STATUS_OK, 'model': self.estimator}

    @staticmethod
    def trial_to_dataframe(trial: Sequence[Dict[str, Any]]) -> pd.DataFrame:
        vals = []
        for t in trial:
            result = t['result']
            misc = t['misc']
            val = {k: (v[0] if isinstance(v, list) else v) for k, v in misc['vals'].items()}
            val['loss'] = result['loss']
            val['tid'] = t['tid']
            vals.append(val)
        return pd.DataFrame(vals)

    def optim(self,
              param_space: Dict[str, Union[float, int]],
              X_train: pd.DataFrame, y_train: pd.Series,
              X_test: pd.DataFrame, y_test: pd.Series,
              metric: str = 'accuracy',
              max_evals: int = 500,
              with_cross_validation: bool = True):
        trials = Trials()

        if with_cross_validation:
            best = fmin(fn=lambda space: self.__hyperparameter_tuning_with_cv(space,
                                                                              X_train=X_train, y_train=y_train,
                                                                              metric=metric),
                        space=param_space,
                        algo=tpe.suggest,
                        max_evals=max_evals,
                        trials=trials,
                        early_stop_fn=no_progress_loss(10))

        else:
            best = fmin(fn=lambda space: self.__hyperparameter_tuning(space,
                                                                      X_train=X_train, y_train=y_train,
                                                                      X_test=X_test, y_test=y_test,
                                                                      metric=metric),
                        space=param_space,
                        algo=tpe.suggest,
                        max_evals=max_evals,
                        trials=trials,
                        early_stop_fn=True)
        df_trial = self.trial_to_dataframe(trials)

        return {'best': best, 'trials': df_trial}

    def optim_with_early_stopping(self,
                                  param_space: Dict[str, Union[float, int]],
                                  X_train: pd.DataFrame, y_train: pd.Series,
                                  X_test: pd.DataFrame, y_test: pd.Series,
                                  early_stopping_rounds: int = 50,
                                  metric: callable = accuracy_score,
                                  max_evals: int = 100
                                  ):
        trials = Trials()
        best = fmin(fn=lambda space: self.__hyperparameter_tuning_with_early_stopping(space,
                                                                                      X_train=X_train, y_train=y_train,
                                                                                      X_test=X_test, y_test=y_test,
                                                                                      early_stopping_rounds=early_stopping_rounds,
                                                                                      metric=metric),
                    space=param_space,
                    algo=tpe.suggest,
                    max_evals=max_evals,
                    trials=trials)
        df_trial = self.trial_to_dataframe(trials)

        return {'best': best, 'trials': df_trial}
