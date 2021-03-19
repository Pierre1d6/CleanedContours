import warnings

import matplotlib as plt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sksurv.ensemble import RandomSurvivalForest
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.metrics import concordance_index_censored
import pingouin as pg



class RemoveHighlyCorrelatedFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, threshold=0.8):
        self.indices = None
        self.threshold = threshold

    def fit(self, X, y=None):
        corr = np.corrcoef(X.T)
        corr = (corr + corr.T) / 2
        np.fill_diagonal(corr, 1)
        indices_n, indices_k = np.where(np.abs(corr) > self.threshold)
        ind = [indices_k[i] for i, x in enumerate(indices_n) if x > indices_k[i]]
        ind = np.asarray(ind)
        ind = np.unique(ind)

        self.indices = ind
        return self

    def transform(self, X, y='deprecated', copy=None):
        # print(np.delete(X, self.indices, axis=1))
        return np.delete(X, self.indices, axis=1)


class StratifiedCustomKfold(StratifiedKFold):
    def split(self, X, y, groups=None):
        y = pd.DataFrame.from_records(y)
        y = y.iloc[:, 0]
        return super().split(X, y, groups)


scaler = StandardScaler()
alphas = np.linspace(0, 0, 1)
reg = CoxPHSurvivalAnalysis(n_iter=1000)

rsf = RandomSurvivalForest(n_estimators=100,
                           min_samples_split=10,
                           min_samples_leaf=15,
                           max_features="sqrt",
                           n_jobs=50,
                           random_state=0)
pipe = Pipeline(steps=[
    ('normalization', scaler),
    ('drop_features', RemoveHighlyCorrelatedFeatures()),
    # ('feature_selection', selector),
    # ('pca',PCA()),
    ('regressor', reg),
])
T_OPTIONS = [0.6, 0.65, 0.70, 0.75, 0.80]
param_grid = [
    {
        'drop_features__threshold': T_OPTIONS
    },
]

filePath = '/media/pierre/HDD/PHD/Hecktor/VincentData/hecktor_gtvt_Features274fts.csv'
filePath2 = '/media/pierre/HDD/PHD/Hecktor/VincentData/original_gtvt_Features274fts.csv'
outputFileFS = "SurvivalAnalysis/FeatureselectionBasedCindex.csv"
outputFileCindex = "SurvivalAnalysis/CindexRadiothyVsRadiomics.csv"
outcome = 'DFS'
time = "Time-DFS"

inputFile = pd.read_csv(filePath, index_col=0, sep=";")
inputFile2 = pd.read_csv(filePath2, index_col=0, sep=";")

Xhecktor = inputFile
y = Xhecktor.loc[:, [outcome, time]]
Xhecktor = Xhecktor.iloc[:, 0:-2]

Xoriginal = inputFile2
Xoriginal = Xoriginal.iloc[:, 0:-2]


n_iter = 1
iteration = 1
computeKMcurve = False
saveCindex = False
rskf = StratifiedShuffleSplit(n_splits=n_iter, train_size=0.8, random_state=42)
cv = StratifiedCustomKfold(n_splits=5, shuffle=True, random_state=42)

warnings.simplefilter("ignore", ConvergenceWarning)
warnings.simplefilter("ignore", UserWarning)
CindexTrainO = []
CindexTrainH = []
CindexTestO = []
CindexTestH = []
Best_paramsH = []
Best_paramsO = []
pvalH = []
pvalO = []
countRadiotherapy = np.zeros_like(Xhecktor.columns.to_list(), dtype=np.int8)
countRadiomics = np.zeros_like(Xhecktor.columns.to_list(), dtype=np.int8)
listiteration = []
Testsamples = pd.DataFrame()


for train_index, test_index in rskf.split(Xhecktor, y.loc[:, outcome]):
    print("Iteration n°: " + str(iteration))
    y[outcome] = y[outcome].astype(bool)
    X_trainH, X_testH = Xhecktor.iloc[train_index, :], Xhecktor.iloc[test_index, :]

    # print(X_trainH)
    listiteration.append("iteration n° " + str(iteration))
    Testsamplestemp = pd.DataFrame((X_testH.index.values).T)
    Testsamples = pd.concat([Testsamples, Testsamplestemp], axis=1)
    # print(Testsamples)

    X_trainO, X_testO = Xoriginal.iloc[train_index, :], Xoriginal.iloc[test_index, :]
    y_train, y_test = y.iloc[train_index, :], y.iloc[test_index, :]
    y_traint = y_train.to_records(index=False)
    y_testt = y_test.to_records(index=False)

    featurelist = []
    cindexlistoriginal = []
    Cindexoriginal_shift = []
    Cindexhecktor_shift = []
    cindexlisthecktor = []
    ICC = []
    pval = []
    CI95 = []

    for feature in X_trainH.columns.to_list():

        Cindexoriginal = concordance_index_censored(y_train.loc[:, outcome].astype('bool'), y_train.loc[:, time],
                                                    X_trainO.loc[:, feature])
        Cindexhecktor = concordance_index_censored(y_train.loc[:, outcome].astype('bool'), y_train.loc[:, time],
                                                   X_trainH.loc[:, feature])
        featurelist.append(feature)
        cindexlistoriginal.append(Cindexoriginal[0])
        cindexlisthecktor.append(Cindexhecktor[0])

        #featureOrignal = X_trainO.loc[:,feature].reset_index().rename(columns={'index': 'Patient', feature:'Feature'})
        #featureHecktor = X_trainH.loc[:,feature].reset_index().rename(columns = {'index': 'Patient',feature:'Feature'})
        #featureOrignal.insert(0, 'Annotations', 'Radiotherapy')
        #featureHecktor.insert(0, 'Annotations', 'Radiomics')
        #data = pd.concat([featureOrignal,featureHecktor],ignore_index=True)
        #icc = pg.intraclass_corr(data=data, targets='Patient', raters='Annotations',ratings='Feature')#.round(3)
        #icc.set_index("Type",inplace=True)
        #ICC.append(icc.loc['ICC3',"ICC"])
        #pval.append(icc.loc['ICC3',"pval"])
        #CI95.append(icc.loc['ICC3',"CI95%"])

        Cindexoriginal_shift.append(abs(Cindexoriginal[0] - 0.5))
        Cindexhecktor_shift.append(abs(Cindexhecktor[0] - 0.5))

    #results = pd.DataFrame({"ICC3": ICC,"Cindex Radiotherapy": cindexlistoriginal, "Cindex Radiomics": cindexlisthecktor, "CindexRadiotherapyShift": Cindexoriginal_shift, "CindexRadiomicsShift": Cindexhecktor_shift},index=featurelist)
    #results.to_csv("ICC3_274fts.csv")

    resultsheadFeatureRadiotherapy = pd.DataFrame(
        {"Cindex Radiotherapy shift": Cindexoriginal_shift, "Feature": featurelist})
    resultsheadFeatureRadiomics = pd.DataFrame({"Cindex Radiomics shift": Cindexhecktor_shift, "Feature": featurelist})

    resultsheadFeatureRadiotherapy.sort_values("Cindex Radiotherapy shift", inplace=True, ascending=False)
    resultsheadFeatureRadiomics.sort_values("Cindex Radiomics shift", inplace=True, ascending=False)
    head20Radiotherapy = resultsheadFeatureRadiotherapy.head(20)  # 20
    head20Radiomics = resultsheadFeatureRadiomics.head(20)  # 20
    ListFeatureRadiotherapy = head20Radiotherapy['Feature'].tolist()
    ListFeatureRadiomics = head20Radiomics['Feature'].tolist()

    searchO = GridSearchCV(pipe, param_grid, cv=cv,
                           n_jobs=7, refit=True, error_score=0.5)
    searchH = GridSearchCV(pipe, param_grid, cv=cv,
                           n_jobs=7, refit=True, error_score=0.5)

    X_trainO = X_trainO[ListFeatureRadiotherapy]
    X_testO = X_testO[ListFeatureRadiotherapy]

    X_trainH = X_trainH[ListFeatureRadiomics]
    X_testH = X_testH[ListFeatureRadiomics]

    searchO.fit(X_trainO, y_traint)
    searchH.fit(X_trainH, y_traint)

    CindexTrainO.append(searchO.score(X_trainO, y_traint))
    CindexTrainH.append(searchH.score(X_trainH, y_traint))

    c_cindexTestO = searchO.score(X_testO, y_testt)
    c_cindexTestH = searchH.score(X_testH, y_testt)

    Best_paramsO.append(searchO.best_params_)
    Best_paramsH.append(searchH.best_params_)

    CindexTestO.append(c_cindexTestO)
    CindexTestH.append(c_cindexTestH)

    Columns = ["Cindex_TrainVal", "Cindex_Test", "Best_params"]

    predictedscoreH = searchH.predict(X_testH)
    predictedscoreO = searchO.predict(X_testO)

    y_testH = y_test.copy(deep=False)
    y_testO = y_test.copy(deep=False)

    y_testH['predicted score'] = predictedscoreH
    y_testH['group'] = pd.cut(y_testH['predicted score'],
                              [np.min(y_testH["predicted score"].values),
                               np.quantile(y_testH["predicted score"].values, 0.5),
                               np.max(y_testH["predicted score"].values)], labels=["infmed", "supmed"], right=True,
                              include_lowest=True)
    y_testO['predicted score'] = predictedscoreO
    y_testO['group'] = pd.cut(y_testO['predicted score'],
                              [np.min(y_testO["predicted score"].values),
                               np.quantile(y_testO["predicted score"].values, 0.5),
                               np.max(y_testO["predicted score"].values)], labels=["infmed", "supmed"], right=True,
                              include_lowest=True)

    groupsH = y_testH['group']
    groupsO = y_testO['group']
    ixH = (groupsH == 'infmed')
    ixO = (groupsO == 'infmed')

    resultsH = logrank_test(y_testH["Time-DFS"][~ixH], y_testH["Time-DFS"][ixH], event_observed_A=y_testH["DFS"][~ixH],
                            event_observed_B=y_testH["DFS"][ixH])
    pvalH.append(resultsH.p_value)
    resultsO = logrank_test(y_testO["Time-DFS"][~ixO], y_testO["Time-DFS"][ixO], event_observed_A=y_testO["DFS"][~ixO],
                            event_observed_B=y_testO["DFS"][ixO])
    pvalO.append((resultsO.p_value))

    if computeKMcurve:
        if iteration == 64:  # 63
            kmf = KaplanMeierFitter()
            kmf.fit(y_testH["Time-DFS"][~ixH], y_testH["DFS"][~ixH], label='SupMed')
            ax = kmf.plot_survival_function(ci_show=True, show_censors=True)

            kmf.fit(y_testH["Time-DFS"][ixH], y_testH["DFS"][ixH], label='InfMed')
            ax = kmf.plot_survival_function(ax=ax, ci_show=True, show_censors=True)
            ax.set_ylim(0.45, 1)
            ax.text(0, 0.5, "p-value = " + str(round(resultsH.p_value, 2)))
            print(c_cindexTestH)
            plt.pyplot.show()
        elif iteration == 81:
            kmf = KaplanMeierFitter()
            kmf.fit(y_testO["Time-DFS"][~ixO], y_testO["DFS"][~ixO], label='SupMed')
            ax = kmf.plot_survival_function(ci_show=True, show_censors=True)

            kmf.fit(y_testO["Time-DFS"][ixO], y_testO["DFS"][ixO], label='InfMed')
            ax = kmf.plot_survival_function(ax=ax, ci_show=True, show_censors=True)
            ax.set_ylim(0.45, 1)
            ax.text(0, 0.5, "p-value = " + str(round(resultsO.p_value, 2)))
            print(c_cindexTestO)
            plt.pyplot.show()

    iteration += 1

dfResults = pd.DataFrame({"Cindex_TrainVal_Radiotherapy": CindexTrainO, "Cindex_Test_Radiotherapy": CindexTestO,
                          "Cindex_TrainVal_Radiomics": CindexTrainH, "Cindex_Test_Radiomics": CindexTestH,
                          "p-valueTest_Radiotherapy": pvalO, "PvalueTest_Radiomics": pvalH})
dfBestparamsO = pd.DataFrame(Best_paramsO)
dfBestparamsH = pd.DataFrame(Best_paramsH)

print(np.mean(CindexTestO))
print(np.mean(CindexTestH))
dfResults = pd.concat([dfResults, dfBestparamsO, dfBestparamsH], axis=1)
if saveCindex:
    dfResults.to_csv(outputFileCindex)



