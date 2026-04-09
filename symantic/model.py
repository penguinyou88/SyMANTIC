
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 17 09:22:50 2023

@author: muthyala.7
"""


from .feature_expansion import nondimensional as fcc

from .feature_expansion import dimensional as dfcc

from .results import FitResult

from .validation import validate_dataframe, validate_operators, validate_dimensions

import sys

import time

import torch

import numpy as np

import pandas as pd

from sympy import symbols

import matplotlib.pyplot as plt

import matplotlib

class SymanticModel:

  def __init__(self,df,operators=None,multi_task = None,n_expansion=None,n_term=None,sis_features=20,device=None,relational_units = None,initial_screening = None,dimensionality=None,output_dim = None,metrics=[0.06,0.995],disp=False,pareto=False,max_features=None):
    """Initialize SymanticModel.

    Parameters
    ----------
    df : pd.DataFrame
        Input data. First column is the target, rest are features.
    operators : list of str
        Mathematical operators for feature expansion (e.g. ['+', '*', 'exp']).
    multi_task : tuple or None
        (target_indices, feature_indices) for multi-task regression.
    n_expansion : int or None
        Number of expansion levels. None enables auto-depth mode.
        Note: uses range(1, n_expansion), so n_expansion=2 gives 1 level.
    n_term : int or None
        Max terms per equation. Default 3.
    sis_features : int
        Number of features to keep via SIS screening. Default 20.
    device : str or None
        'cpu', 'cuda', or None (auto-detect). Default None selects
        'cuda' when a GPU is available, otherwise 'cpu'.
    relational_units : list or None
        Unit relationships for dimensional regression.
    initial_screening : tuple or None
        (n_features, quantile) for initial feature screening.
    dimensionality : list or None
        Sympy dimension expressions for dimensional regression.
    output_dim : sympy expr or None
        Dimension of the target variable.
    metrics : list
        [rmse_threshold, r2_threshold] for auto-depth convergence.
    disp : bool
        Print progress information. Default False.
    pareto : bool
        Compute Pareto front. Default False.
    max_features : int or None
        Maximum features before stopping expansion in auto-depth mode.
        Default: 2000 for non-dimensional, 10000 for dimensional.
        Increase to search for more complex equations (at higher memory cost).
    """
    # Validate inputs
    validate_dataframe(df)
    if operators is not None:
        validate_operators(operators)
    if dimensionality is not None:
        validate_dimensions(dimensionality, df)

    self.operators = operators

    self.df=df

    self.no_of_operators = n_expansion

    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    self.device = device

    if n_term == None: self.dimension = 3

    else: self.dimension = n_term

    if sis_features == None: self.sis_features = 10

    else: self.sis_features = sis_features

    self.relational_units = relational_units

    self.initial_screening = initial_screening

    self.dimensionality = dimensionality

    self.output_dim = output_dim


    self.metrics   = metrics

    self.multi_task = multi_task

    self.disp=disp

    self.pareto=pareto

    # Set max_features with sensible defaults per mode
    if max_features is not None:
        self.max_features = max_features
    elif dimensionality is not None:
        self.max_features = 10000
    else:
        self.max_features = 2000

    if multi_task!=None:

        self.multi_task_target = multi_task[0]

        self.multi_task_features = multi_task[1]
    self.final_df = None
    

  def combine_equation(self,row):
      
      terms = row['Equations']
      
      coeffs = row['Coefficients']
      
      intercept = row['Intercepts']
      
      if isinstance(terms, str):
      
          terms = [terms]
      
      equation_parts = []
      
      for term, coeff in zip(terms, coeffs):
          if pd.isna(coeff):
              continue
          if coeff == 1:
              equation_parts.append(term)
          elif coeff == -1:
              equation_parts.append(f"-{term}")
          else:
              equation_parts.append(f"{coeff:.4f}*{term}")
      
      equation = " + ".join(equation_parts)
      
      if intercept != 0:
          
          if intercept > 0:
              
              equation = f"{equation} + {intercept:.4f}"
          
          else:
              
              equation = f"{equation} - {abs(intercept):.4f}"
      
      return equation   

    
  def _build_pareto_result(self, final):
      """Post-process auto-depth Pareto front into FitResult."""
      self.final_df = final

      final['Normalized_Loss'] = (final['Loss'] - final['Loss'].min()) / (final['Loss'].max() - final['Loss'].min())
      final['Normalized_Complexity'] = (final['Complexity'] - final['Complexity'].min()) / (final['Complexity'].max() - final['Complexity'].min())
      final['Distance_to_Utopia'] = np.sqrt(final['Normalized_Loss']**2 + final['Normalized_Complexity']**2)

      utopia_row = final['Distance_to_Utopia'].idxmin()

      final_edited = pd.DataFrame()
      final_edited['Loss'] = final['Loss']
      final_edited['Complexity'] = final['Complexity']
      final_edited['R2'] = final['Score']
      final_edited['Equation'] = final.apply(self.combine_equation, axis=1)

      if self.disp:
          print('Pareto set generated. Access via result.pareto_front')

      return FitResult(
          rmse=float(final_edited.Loss[utopia_row]),
          equation=str(final_edited.Equation[utopia_row]),
          r2=float(final_edited.R2[utopia_row]),
          complexity=float(final_edited.Complexity[utopia_row]),
          pareto_front=final_edited,
      )

  def fit(self):
      """Run symbolic regression and return a FitResult.

      Returns
      -------
      FitResult
          Unified result object. Supports backward-compatible tuple unpacking:
          - Auto-depth: ``res, pareto_df = model.fit()``
          - Fixed-depth: ``rmse, equation, r2 = model.fit()``
          - Multi-task: ``rmse, equation, r2, equations = model.fit()``
      """
      if self.dimensionality == None:

        if self.operators==None: sys.exit('Please provide the operators set for the non dimensional Regression!!')

        if self.multi_task!=None:

            if self.disp: print('************************************* Performing MultiTask Symbolic regression!!..**************************************************************** \n')

            equations =[]

            for i in range(len(self.multi_task_target)):

                if self.disp: print('***************************************** Performing symbolic regression of',i+1,'Target variables******************************************** \n')

                list1 =[]
                list1.extend([self.multi_task_target[i]]+self.multi_task_features[i])
                df1 = self.df.iloc[:,list1]

                if self.no_of_operators==None:

                    st = time.time()
                    rmse,equation,r2,_ = fcc.feature_space_construction(self.operators,df1,self.no_of_operators,self.device,self.initial_screening,self.metrics,dimension=self.dimension,sis_features=self.sis_features,disp=self.disp,pareto=self.pareto,max_features=self.max_features).feature_space()
                    if self.disp: print('************************************************ Autodepth regression completed in::', time.time()-st,'seconds ************************************************ \n')

                    equations.append(equation)
                    if i+1 == len(self.multi_task_target):
                        if self.disp: print('Equations found::',equations)
                        return FitResult(rmse=rmse, equation=equation, r2=r2, all_equations=equations)
                    else:continue

                else:

                    x,y,names,complexity = fcc.feature_space_construction(self.operators,df1,self.no_of_operators,self.device,self.initial_screening,disp=self.disp,pareto=self.pareto,max_features=self.max_features).feature_space()
                    from .regression.l0_greedy import Regressor
                    rmse, equation,r2,r,c,n,intercepts,coeffs,_ =  Regressor(x,y,names,complexity,self.dimension,self.sis_features,self.device).regressor_fit()

                    equations.append(equation)
                    if i+1 == len(self.multi_task_target):
                        if self.disp: print('Equations found::',equations)
                        return FitResult(rmse=rmse, equation=equation, r2=r2, all_equations=equations)
                    else: continue

        elif self.no_of_operators==None:

            st = time.time()
            rmse,equation,r2,final = fcc.feature_space_construction(self.operators,self.df,self.no_of_operators,self.device,self.initial_screening,self.metrics,dimension=self.dimension,sis_features=self.sis_features,disp=self.disp,pareto=self.pareto,max_features=self.max_features).feature_space()
            if self.disp: print('************************************************ Autodepth regression completed in::', time.time()-st,'seconds ************************************************ \n')

            return self._build_pareto_result(final)

        else:

            x,y,names,complexity = fcc.feature_space_construction(self.operators,self.df,self.no_of_operators,self.device,self.initial_screening,disp=self.disp,max_features=self.max_features).feature_space()
            from .regression.l0_greedy import Regressor
            rmse, equation,r2,r,c,n,intercepts,coeffs,_ =  Regressor(x,y,names,complexity,self.dimension,self.sis_features,self.device).regressor_fit()

            return FitResult(rmse=rmse, equation=equation, r2=r2)

      else:

        if self.multi_task!=None:

            if self.disp: print('************************************************ Performing MultiTask Symbolic regression!!..************************************************ \n')

            equations =[]

            for i in range(len(self.multi_task_target)):

                if self.disp: print('************************************************ Performing symbolic regression of',i+1,'Target variables....************************************************ \n')

                list1 =[]
                list1.extend([self.multi_task_target[i]]+self.multi_task_features[i])
                df1 = self.df.iloc[:,list1]

                if self.no_of_operators==None:

                    st = time.time()
                    rmse,equation,r2,final = dfcc.feature_space_construction(df1,self.operators,self.relational_units,self.initial_screening,self.no_of_operators,self.device,self.dimensionality,self.metrics,self.output_dim,disp=self.disp,pareto=self.pareto,max_features=self.max_features).feature_expansion()
                    if self.disp: print('************************************************ Autodepth regression completed in::', time.time()-st,'seconds ************************************************ \n')

                    equations.append(equation)
                    if i+1 == len(self.multi_task_target):
                        if self.disp: print('Equations found::',equations)
                        return FitResult(rmse=rmse, equation=equation, r2=r2, all_equations=equations)
                    else:continue

                else:

                    x,y,names,dim,complexity = dfcc.feature_space_construction(df1,self.operators,self.relational_units,self.initial_screening,self.no_of_operators,self.device,self.dimensionality,disp=self.disp,pareto=self.pareto,max_features=self.max_features).feature_expansion()
                    from .regression.l0_greedy_dimensional import Regressor
                    rmse,equation,r2,_,_,_,_,_,_ = Regressor(x,y,names,dim,complexity,self.dimension,self.sis_features,self.device,self.output_dim,disp=self.disp,pareto=self.pareto).regressor_fit()

                    equations.append(equation)
                    if i+1 == len(self.multi_task_target):
                        if self.disp: print('Equations found::',equations)
                        return FitResult(rmse=rmse, equation=equation, r2=r2, all_equations=equations)
                    else: continue

        if self.no_of_operators==None:

            st = time.time()
            rmse,equation,r2,final = dfcc.feature_space_construction(self.df,self.operators,self.relational_units,self.initial_screening,self.no_of_operators,self.device,self.dimensionality,self.metrics,self.output_dim,disp=self.disp,pareto=self.pareto,max_features=self.max_features).feature_expansion()
            if self.disp: print('************************************************ Autodepth regression completed in::', time.time()-st,'seconds ************************************************ \n')

            return self._build_pareto_result(final)

        else:

            x,y,names,dim,complexity = dfcc.feature_space_construction(self.df,self.operators,self.relational_units,self.initial_screening,self.no_of_operators,self.device,self.dimensionality,disp=self.disp,pareto=self.pareto,max_features=self.max_features).feature_expansion()
            from .regression.l0_greedy_dimensional import Regressor
            rmse,equation,r2,_,_,_,_,_,_ = Regressor(x,y,names,dim,complexity,self.dimension,self.sis_features,self.device,self.output_dim,disp=self.disp,pareto=self.pareto).regressor_fit()

            return FitResult(rmse=rmse, equation=equation, r2=r2)
        
  def plot_pareto_front(self):
    
    import matplotlib.pyplot as plt

    import matplotlib  
    plt.figure(figsize=(10, 8))
    
    plt.scatter(self.final_df['Complexity'],self.final_df['Loss'], c='red', label='Pareto front')
    
    plt.step(self.final_df['Complexity'],self.final_df['Loss'], 'r-', where='post', label='Pareto Line')
    
    plt.scatter(self.final_df['Complexity'].min(),self.final_df['Loss'].min(), c='green', label='Utopia',marker='*',s=100)
    
    plt.xlabel(r'Complexity = $k \log n$ (bits)',weight='bold') 
    
    plt.ylabel('Accuracy (RMSE)',weight='bold')
    
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.grid(True)
    
    plt.title('Pareto Frontier')
    
    plt.show()


