import os
import glob
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from openpyxl import Workbook
from matplotlib.lines import Line2D

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score
from imblearn.over_sampling import SMOTE
import shap
import warnings

warnings.filterwarnings('ignore')


class DEMATELSolver:
    def __init__(self):
        self.experts = []
        self.factors = []
        self.matrix = []
        self.Z = None   # aggregated direct influence
        self.X = None   # normalized direct influence
        self.T = None   # total influence
        self.R = None   # row sums
        self.C = None   # column sums
        self.prominence = None  # R + C
        self.relation = None    # R - C
        self.result = {"cause": [], "effect": []}

    def read_factors(self, path_txt):
        """Load factor names, one per line."""
        with open(path_txt) as f:
            self.factors = [l.strip() for l in f if l.strip()]
        print(f"Loaded {len(self.factors)} factors.")

    def load_expert_matrices(self, folder_csv):
        """Read each CSV in folder as one expert's direct‐influence matrix."""
        for csvfile in glob.glob(os.path.join(folder_csv,"*.csv")):
            with open(csvfile) as f:
                M = np.loadtxt(f, delimiter=',')
            self.matrix.append(M)
            self.experts.append(os.path.splitext(os.path.basename(csvfile))[0])
        E = len(self.matrix)
        if E>0:
            F = self.matrix[0].shape[0]
            print(f"Loaded {E} expert matrices, each {F}×{F}")

    def step1_aggregate(self):
        """Z = average of expert matrices (equal weights)."""
        self.Z = sum(self.matrix)/len(self.matrix)
        print("Step1: aggregated Z")

    def step2_normalize(self):
        """X = Z / max(sum rows, sum cols)."""
        S = max(self.Z.sum(axis=1).max(), self.Z.sum(axis=0).max())
        self.X = self.Z / S
        print("Step2: normalized X")

    def step3_total(self):
        """T = X · (I−X)^−1."""
        I = np.eye(len(self.factors))
        self.T = self.X.dot(np.linalg.inv(I-self.X))
        print("Step3: total T")

    def step4_identify(self):
        """Compute R, C, prominence, relation → cause/effect."""
        self.R = self.T.sum(axis=1)
        self.C = self.T.sum(axis=0)
        self.prominence = self.R + self.C
        self.relation   = self.R - self.C
        for i, f in enumerate(self.factors):
            if self.relation[i]>0:
                self.result["cause"].append(f)
            else:
                self.result["effect"].append(f)
        print(f"Causes: {self.result['cause']}")
        print(f"Effects: {self.result['effect']}")

    def plot_irm(self):
        """Scatter R+C vs R−C, green=cause, red=effect."""
        colors = ['green' if r>0 else 'red' for r in self.relation]
        plt.figure(figsize=(8,6))
        plt.scatter(self.prominence, self.relation, c=colors, s=80)
        for f,p,r in zip(self.factors,self.prominence,self.relation):
            plt.text(p,r,f,ha='center',va='bottom')
        plt.axhline(0, color='k', linestyle='--')
        plt.axvline(0, color='k', linestyle='--')
        plt.xlabel('Prominence (R + C)')
        plt.ylabel('Relation  (R − C)')
        plt.title('DEMATEL IRM')
        legend = [
            Line2D([0],[0],marker='o',color='w',label='Cause',markerfacecolor='green',markersize=10),
            Line2D([0],[0],marker='o',color='w',label='Effect',markerfacecolor='red',markersize=10),
        ]
        plt.legend(handles=legend)
        plt.tight_layout()
        plt.show()

    def save_excel(self, out_folder):
        """Save Z, X, T and R/C results to DEMATELAnalysis.xlsx."""
        os.makedirs(out_folder, exist_ok=True)
        wb = Workbook()
        # Sheet1: R/C
        ws = wb.active; ws.title="R-C Metrics"
        ws.append(["Factor","R","C","R+C","R-C"])
        for f,r,c in zip(self.factors,self.R,self.C):
            ws.append([f,float(r),float(c),float(r+c),float(r-c)])
        # Z, X, T each in its own sheet
        def dump_mat(sheet_name, M):
            ws2=wb.create_sheet(sheet_name)
            ws2.append([""]+self.factors)
            for i,row in enumerate(M):
                ws2.append([self.factors[i]]+list(row))
        dump_mat("Z", self.Z)
        dump_mat("X", self.X)
        dump_mat("T", self.T)
        wb.save(os.path.join(out_folder,"DEMATELAnalysis.xlsx"))
        print("Excel saved.")

    def prepare_dataset(self):
        """
        Build DataFrame where each row=factor, columns=avg influence profile
        plus label 'Cluster' {1:Cause,0:Effect}.
        """
        avg_profiles = self.Z  # use aggregated direct matrix
        data = []
        for i,f in enumerate(self.factors):
            row = list(avg_profiles[i,:])
            label = 1 if f in self.result["cause"] else 0
            data.append(row+[label])
        cols = self.factors + ["Cluster"]
        df = pd.DataFrame(data,columns=cols)
        return df

    def train_random_forest(self, df, n_estimators=100):
        """SMOTE → scale → cross_val_score → final fit → feature importances → SHAP."""
        X = df[self.factors]
        y = df["Cluster"]
        sm = SMOTE(random_state=0)
        Xr, yr = sm.fit_resample(X,y)
        scaler = StandardScaler()
        Xs = scaler.fit_transform(Xr)
        # CV
        cv = StratifiedKFold(5,shuffle=True,random_state=0)
        rf = RandomForestClassifier(n_estimators=n_estimators,random_state=0)
        scores = cross_val_score(rf,Xs,yr,cv=cv,scoring="accuracy")
        print(f"RF CV accuracy: {scores.mean():.3f} ± {scores.std():.3f}")
        # final fit
        rf.fit(Xs,yr)
        imp = pd.Series(rf.feature_importances_,index=self.factors)
        print("Feature importances:")
        print(imp.sort_values(ascending=False))
        # SHAP
        expl = shap.TreeExplainer(rf)
        shap_vals = expl.shap_values(Xs)
        shap.summary_plot(shap_vals, pd.DataFrame(Xs,columns=self.factors), plot_type="bar")
        return rf

def main():
    solver = DEMATELSolver()
    # adjust paths as needed
    solver.read_factors("inputs/factors.txt")
    solver.load_expert_matrices("inputs/matrices")
    if not solver.experts:
        print("No expert data—exiting."); return

    solver.step1_aggregate()
    solver.step2_normalize()
    solver.step3_total()
    solver.step4_identify()
    solver.plot_irm()
    solver.save_excel("outputs")

    df = solver.prepare_dataset()
    rf_model = solver.train_random_forest(df)

if __name__ == "__main__":
    main()
