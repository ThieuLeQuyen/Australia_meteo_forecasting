# -*- coding: utf-8 -*-

"""
@created: 09/2023
@updated:
@author: quyen@marcaud.fr
"""

import pandas as pd
from matplotlib import pyplot as plt
from yellowbrick import classifier, model_selection


class ClassificationEvaluation:
    def __init__(self, estimator, X_train, y_train, X_test, y_test):
        self.estimator = estimator
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

    def score(self, early_stopping_rounds: int = None):
        if not hasattr(self.estimator, "classes_"):
            print("The model is not fitted")
            if early_stopping_rounds is not None:
                evaluation = [(self.X_train, self.y_train),
                              (self.X_test, self.y_test)]
                self.estimator.fit(self.X_train, self.y_train, eval_set=evaluation)
            else:
                self.estimator.fit(self.X_train, self.y_train)
        return self.estimator.score(self.X_test, self.y_test)

    def viz_classification_report(self, file_out: str = None,
                                  classes=None, support: bool = False, title: str = None, ax=None,
                                  early_stopping_rounds: int = None):
        class_report = classifier.ClassificationReport(estimator=self.estimator,
                                                       classes=classes, support=support, cmap="YlGn",
                                                       title=title, is_fitted='auto', ax=ax)
        if early_stopping_rounds is not None:
            evaluation = [(self.X_train, self.y_train),
                          (self.X_test, self.y_test)]
            class_report.fit(self.X_train, self.y_train, eval_set=evaluation)
        else:
            class_report.fit(self.X_train, self.y_train)
        class_report.score(self.X_test, self.y_test)
        class_report.show(outpath=file_out)

    def viz_confusion_matrix(self, file_out: str = None, normalize: bool = False,
                             classes=None, title: str = None, ax=None,
                             early_stopping_rounds: int = None):
        cm = classifier.ConfusionMatrix(self.estimator, classes=classes, percent=normalize,
                                        title=title, is_fitted='auto', ax=ax)
        if early_stopping_rounds is not None:
            evaluation = [(self.X_train, self.y_train),
                          (self.X_test, self.y_test)]
            cm.fit(self.X_train, self.y_train, eval_set=evaluation)
        else:
            cm.fit(self.X_train, self.y_train)
        cm.score(self.X_test, self.y_test)
        cm.show(outpath=file_out)

    def viz_ROCAUC(self, file_out: str = None, classes=None, title: str = None, ax=None,
                   early_stopping_rounds: int = None):
        roc_curve = classifier.ROCAUC(self.estimator, classes=classes,
                                      title=title, is_fitted='auto', ax=ax)
        if early_stopping_rounds is not None:
            evaluation = [(self.X_train, self.y_train),
                          (self.X_test, self.y_test)]
            # @todo: TypeError: ROCAUC.fit() got an unexpected keyword argument 'eval_set'
            # roc_curve.fit(self.X_train, self.y_train, eval_set=evaluation)
            roc_curve.fit(self.X_train, self.y_train)
        else:
            roc_curve.fit(self.X_train, self.y_train)
        roc_curve.score(self.X_test, self.y_test)
        roc_curve.show(outpath=file_out)

    def feature_importances(self, file_out: str = None, title: str = None, ax=None,
                            early_stopping_rounds: int = None):
        viz = model_selection.FeatureImportances(self.estimator, title=title, ax=ax, is_fitted='auto')
        if early_stopping_rounds is not None:
            evaluation = [(self.X_train, self.y_train),
                          (self.X_test, self.y_test)]
            viz.fit(self.X_train, self.y_train, eval_set=evaluation)
        else:
            viz.fit(self.X_train, self.y_train)
        viz.show(outpath=file_out)

    def evaluate(self, classes=None, early_stopping_rounds: int = None,
                 file_confusion_matrix: str = None,
                 file_class_report: str = None,
                 file_rocauc: str = None,
                 file_feat_import: str = None,
                 title_confusion_matrix: str = None,
                 title_class_report: str = None,
                 title_rocauc: str = None,
                 title_feat_import: str = None,
                 confusion_matrix: bool = True,
                 class_report: bool = True,
                 rocauc: bool = True,
                 feature_importance: bool = True):
        # Matrice de confusion
        if confusion_matrix:
            fig_cm, ax = plt.subplots()
            self.viz_confusion_matrix(file_out=file_confusion_matrix, normalize=True, classes=classes,
                                      title=title_confusion_matrix, ax=ax, early_stopping_rounds=early_stopping_rounds)

        # Classification report
        if class_report:
            fig_report, ax = plt.subplots()
            self.viz_classification_report(file_out=file_class_report, support=True, classes=classes,
                                               title=title_class_report, ax=ax, early_stopping_rounds=early_stopping_rounds)

        # ROC curve
        if rocauc:
            fig_rocauc, ax = plt.subplots()
            self.viz_ROCAUC(file_out=file_rocauc, classes=classes,
                            title=title_rocauc, ax=ax, early_stopping_rounds=early_stopping_rounds)

        # Feature importances
        if feature_importance:
            fig_feat_import, ax = plt.subplots(figsize=(15, 20))
            self.feature_importances(file_out=file_feat_import,
                                     title=title_feat_import, ax=ax, early_stopping_rounds=early_stopping_rounds)

