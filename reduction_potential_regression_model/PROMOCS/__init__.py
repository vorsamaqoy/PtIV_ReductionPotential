import random
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import statistics
import sys
random.seed(0)
np.random.seed(0)

from collections import defaultdict
from pathlib import Path
from copy import deepcopy
from numpy import *
from sklearn.preprocessing import StandardScaler
import matplotlib.colors as mcolors

from ipywidgets import interact, fixed, widgets

from rdkit import Chem
from rdkit.Chem import AllChem, Draw, rdFMCS, PandasTools, rdFingerprintGenerator

import xgboost as xgb

from sklearn import preprocessing
from sklearn.model_selection import train_test_split, cross_val_score, KFold, LeaveOneOut
from sklearn.metrics import *
from sklearn.inspection import permutation_importance
from sklearn.feature_selection import VarianceThreshold, SelectFromModel
from sklearn.linear_model import *
from sklearn.neighbors import *
from sklearn.tree import *
from sklearn.ensemble import *
from sklearn.decomposition import PCA

def load_smiles(file):
    """
    It loads the list of smiles from an .smi file
    which is transformed into an list of RDKit objects.

    Parameters
    ----------
    file : str
        name of the file with the .smi extension containing the list of molecular smiles
        "name.smi" string
    Returns
    -------
    list
        list of molecules in RDKit object form.
    
    Example usage:
    molecules = load_smiles('smiles.smi')
    """
    with open(file, 'r') as f:
        smiles_list = f.read().splitlines()
    
    molecules = []
    
    for smiles in smiles_list:
        molecule = Chem.MolFromSmiles(smiles,sanitize=False)
        molecules.append(molecule)
    
    return molecules

def is_transition_metal(at):
    n = at.GetAtomicNum()
    return (n>=22 and n<=29) or (n>=40 and n<=47) or (n>=72 and n<=79)
def set_dative_bonds(mol, fromAtoms=(7,8)):
    """ convert some bonds to dative

    Replaces some single bonds between metals and atoms with atomic numbers in fomAtoms
    with dative bonds. The replacement is only done if the atom has "too many" bonds.

    Returns the modified molecule.

    """
    pt = Chem.GetPeriodicTable()
    rwmol = Chem.RWMol(mol)
    rwmol.UpdatePropertyCache(strict=False)
    metals = [at for at in rwmol.GetAtoms() if is_transition_metal(at)]
    for metal in metals:
        for nbr in metal.GetNeighbors():
            if nbr.GetAtomicNum() in fromAtoms and \
               nbr.GetExplicitValence()>pt.GetDefaultValence(nbr.GetAtomicNum()) and \
               rwmol.GetBondBetweenAtoms(nbr.GetIdx(),metal.GetIdx()).GetBondType() == Chem.BondType.SINGLE:
                rwmol.RemoveBond(nbr.GetIdx(),metal.GetIdx())
                rwmol.AddBond(nbr.GetIdx(),metal.GetIdx(),Chem.BondType.DATIVE)
    return rwmol

def generate_ECFP(molecules_list, radius = 6, nbit = 2048):
    """
    Given a list of RDKit objects, returns a pandas dataframe with the generated ECFPs.

    Parameters
    ----------
    molecules_list : list
    list of molecules in RDKit mol object form.
    radius: int
    max radius for substructures generation.
    default is 6
    nbit: int
    fingerprint bit lenght
    default is 2048
    
    Returns
    -------
    dataframe
        Dataframe pandas with calculated ECFP.
    
    Example usage:
    ECFP_df = generate_ECFP(molecules_list, radius = 6, nbit = 2048)
    """
    fp_list = []
    
    for mol in molecules_list:
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nbit, useFeatures=True)
        fp_dict = {f'ECFP{i+1}': bit for i, bit in enumerate(fp.ToBitString())}
        fp_list.append(fp_dict)
    
    ECFP_df = pd.DataFrame(fp_list)
    ECFP_df = ECFP_df.astype(int)
    return ECFP_df

def OHE(alva_df):
    """
    One Hot Encoding of categorical features.

    Parameters
    ----------
    alva_df : pandas.DataFrame
    Dataframe with categorical descriptors.

    Returns
    -------
    dataframe
        Dataframe with categorical descriptors encoded.
    
    Example usage:
    alva_df_OHE = OHE(alva_df)
    """
    for col in alva_df.columns:
        dataset = pd.get_dummies(alva_df[col], prefix=col, drop_first=True)
        alva_df = alva_df.join(dataset)
    
    alva_df_OHE = alva_df.drop(alva_df.columns[:len(alva_df.columns)//2], axis=1)
    return alva_df_OHE

def detect_high_corr(dataset, threshold=0.9):
    """
    Detect high correlated columns

    Parameters
    ----------
    dataset : pandas.DataFrame
    Dataframe with descriptors.
    threshold : float
    correlation ratio threshold
    default is 0.9

    Returns
    -------
    set
        set with high correlated features' name.
    
    Example usage:
    corr_features = correlation(X, 0.9)
    """
    col_corr = set()
    corr_matrix = dataset.corr()
    for i in range(len(corr_matrix.columns)):
        if i % 100 == 0:
            a=len(corr_matrix.columns)
            b=int((i/a)*100)
            sys.stdout.write("\rResearch in progress: {}%".format(b))
            sys.stdout.flush()
        for j in range(i):
            if abs(corr_matrix.iloc[i, j]) > threshold:
                colname = corr_matrix.columns[i]
                col_corr.add(colname)
    return col_corr

def benchmark(X, y, models):
    """
    Benchmark various models to find the best one

    Parameters
    ----------
    X : pandas.DataFrame
    Dataframe with descriptors.
    y : pd.Series
    target values.
    models: list
    list of models instances.

    Returns
    -------
    pd.DataFrame
        dataframe containing the results calculated with LOOCV for each model.
    
    Example usage:
    df_results = benchmark(X, y, models)
    """
    cv = LeaveOneOut()
    results = []

    for model in models:
        mse_scores = []
        mae_scores = []
        y_pred_list = []

        for train_index, test_index in cv.split(X):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            mse_scores.append(mean_squared_error(y_test, y_pred))
            mae_scores.append(mean_absolute_error(y_test, y_pred))
            y_pred_list.append(y_pred)
        
        r2 = r2_score(y, np.concatenate(y_pred_list))
        result_dict = {
            "Model": type(model).__name__,
            "mse": np.mean(mse_scores),
            "rmse": np.sqrt(np.mean(mse_scores)),
            "mae": np.mean(mae_scores),
            "r2": np.mean(r2)
        }
        results.append(result_dict)

    df_results = pd.DataFrame(results)
    df_results.set_index('Model', inplace=True)
    df_results = df_results.round(2)
    return df_results

def select_features(model, X, y, thr='mean'):
    """
    Given a dataframe pandas containing features,
    returns a dataframe pandas after selecting the
    most important features second SelectFromModel of Scikit-Learn.

    Parameters
    ----------
    model: ML model
    ML model.
    X : pandas.DataFrame
    Dataframe with descriptors.
    y : pd.Series
    target values.
    thr: str
    threshold to use with SelectFromModel by scikit-learn.
    default is "mean"

    Returns
    -------
    pd.DataFrame
        Dataframe with selected descriptors.
    
    Example usage:
    X_sel_RFR = select_features(models[0], X, y)
    """
    sfm = SelectFromModel(model, threshold=thr)
    selection = sfm.fit_transform(X, y)
    selected_features = sfm.get_support()
    selected_columns = X.columns[selected_features]
    X_sel = X[selected_columns]
    
    return X_sel

def evaluate_model(X, y, model):
    """
    Evaluate only one model.

    Parameters
    ----------
    X : pandas.DataFrame
    Dataframe with descriptors.
    y : pd.Series
    target values.
    model: ML model
    ML model.
    
    Returns
    -------
    pd.DataFrame, pd.DataFrame, pd.Series
        Dataframe with results, dataframe with feature importance, target predictions.
    
    Example usage:
    df_results_RFR, feat_imp_df_RFR, y_pred_list_RFR = evaluate_model(X_sel_RFR, y, models[0])
    """
    cv = LeaveOneOut()
    mse_scores = []
    mae_scores = []
    y_pred_list = []
    feature_importances = np.zeros(X.shape[1])
    
    for train_index, test_index in cv.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        model.fit(X_train, y_train)
        feature_importances += model.feature_importances_
        y_pred = model.predict(X_test)
        mse_scores.append(mean_squared_error(y_test, y_pred))
        mae_scores.append(absolute(mean_absolute_error(y_test, y_pred)))
        y_pred_list.append(y_pred)
        
    mean_feature_importances = feature_importances / cv.get_n_splits(X)
    
    r2 = r2_score(y, y_pred_list)
    df_results_dict = {
        "mse": [mean(mse_scores)],
        "rmse": [np.sqrt(mean(mse_scores))],
        "mae": [mean(mae_scores)],
        "r2": [mean(r2)]
    }
    
    feat_imp_df = pd.DataFrame({'Feature': X.columns, 'Importance': mean_feature_importances})
    feat_imp_df = feat_imp_df.sort_values('Importance', ascending=False)
    
    df_results = pd.DataFrame(df_results_dict)
    
    return df_results, feat_imp_df, y_pred_list

def RFE(features_list, X, y, model):
    """
    Performs Recursive Features Elimination.

    Parameters
    ----------
    features_list : list
    Features' name.
    X : pandas.DataFrame
    Dataframe with descriptors.
    model: ML model
    ML model.
    
    Returns
    -------
    list
        list of metrics considered according to the number of features selected.
    
    Example usage:
    RFE_results = RFE(feat_imp_df_ETR.Feature, X, models[3])
    """
    RFE_results = []
    
    X_RFE = X[features_list]
    
    cv = LeaveOneOut()
    
    while len(X_RFE.columns) > 0:
        score = cross_val_score(model, X_RFE, y, scoring='neg_mean_squared_error', cv=cv, n_jobs=-1)
        RFE_results.append(mean(absolute(score)))
        X_RFE = X_RFE.drop(X_RFE.columns[-1], axis=1)
        
    RFE_results.reverse()
    
    return RFE_results

def plot_scatter(y_true, y_pred, model_name, ref=1.1):
    """
    x and y scatterplot.

    Parameters
    ----------
    y_true : list
    true target values list.
    y_pred : list
    predcted target values list.
    ref: float
    reference
    default is 1.1
    model: model
    ML model
    
    Returns
    -------
    list
        scatteplot with fit line and 1st and 2nd standard deviation.
    
    Example usage:
    plot_scatter(y, y_pred_list_ETR)
    """
    y_array = np.array(y_true)
    
    y_pred_array = np.array(y_pred)
    
    residuals = y_array - y_pred_array
    
    std_dev = np.std(residuals)
    
    plt.scatter(y_array, y_pred_array, c="#045275", s=15)
    plt.plot([-ref, ref], [-ref, ref], 'green', linestyle='--')
    
    plt.plot([-ref, ref], [-ref + std_dev, ref + std_dev], '--', label='σ', color="orange")
    plt.plot([-ref, ref], [-ref - std_dev, ref - std_dev], '--', color="orange")
    plt.plot([-ref, ref], [-ref + 2*std_dev, ref + 2*std_dev], '--', label='2σ', color="red")
    plt.plot([-ref, ref], [-ref - 2*std_dev, ref - 2*std_dev], '--', color="red")
    
    plt.xlabel('Experimental E(V)')
    plt.ylabel('Predicted E(V)')
    plt.title(type(model_name).__name__)
    plt.grid(True, linestyle=':', linewidth='0.5', color='gray')
    plt.legend()
    
    plt.xlim(-ref, ref)
    plt.ylim(-ref, ref)
    plt.show()
    
def plot2D(data, hue_data=None, palette='viridis', figsize=(5,5), dot = 4):
    """
    Perform PCA on data and create a scatter plot of the results.
    Parameters
    ----------
    data : pd.DataFrame
        Data to perform PCA on.
    hue_data : pd.Series, optional
        Data to use for the hue of the scatter plot points. Default is None.
    pc_x : str, optional
        Principal component to plot on the x-axis. Default is 'PC1'.
    pc_y : str, optional
        Principal component to plot on the y-axis. Default is 'PC2'.
    palette : str, optional
        Palette to use for the scatter plot points. Default is 'viridis'.
    figsize : tuple, optional
        Size of the figure. Default is (5,5).
    dot : int
        Size of dots. Default i
    Returns
    -------
    df_pca : pd.Dataframe
    PC1/PC2 results dataframe.
    
    Example usage:
    pca_components = plot2D(X_no_outlier, hue_data=y_no_outlier, palette='viridis', figsize=(6,5), dot = 30)
    """
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(data)
    pca = PCA(n_components=2)
    principal_components = pca.fit_transform(features_scaled)
    explained_variance = pca.explained_variance_ratio_
    pca_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])
    plt.figure(figsize=figsize)
    plt.scatter(pca_df['PC1'], pca_df['PC2'], c=hue_data, cmap=palette, s=dot)
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title('Explained Variance: PC1 = {:.2f}%, PC2 = {:.2f}%'.format(explained_variance[0]*100,
                                                                        explained_variance[1]*100))
    plt.colorbar(label='Target')
    plt.show()
    
    return pca.components_
                                             
def plot_loadings(pca_components, data, figsize=(5,5)):
    loadings = pca_components
    
    plt.figure(figsize=figsize)
    for i, feature in enumerate(data.columns):
        plt.arrow(0, 0, loadings[0, i]*2, loadings[1, i]*2, color='r',
                  alpha=0.5, head_width=0.02, head_length=0.02)
        plt.text(loadings[0, i]*2.2, loadings[1, i]*2.2, feature,
                 color='black', ha='center', va='center_baseline')
    
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title('PCA - Frecce dei Loadings sul Piano PC1/PC2')
    plt.show()

def corr_matrix(data, color1, color2):
    """
    Builds and shows correlation matrix between features.
    Parameters
    ----------
    data : pd.DataFrame
        Data with features.
    color1 : str
        Hex code of first color.
    color2 : str
        Hex code of second color

    Returns
    -------
    Example usage:
    corr_matrix(data, "#045275", "#7CCBA2")
    """
    
    corr_matrix = data.corr()

    cmap = mcolors.LinearSegmentedColormap.from_list("Custom Colormap", [color1, color2])

    sns.heatmap(corr_matrix, cmap=cmap)
    
    
    
   