#!/usr/bin/env python3

###########################################################################################
# This script implements error propagation for statistical uncertainties which are        #
# uncorrelated (for the purposes of basic error handling in an undergraduate lab setting) #
###########################################################################################

import sympy as sp
import pint

ureg = pint.UnitRegistry()
Q_   = ureg.Quantity

def get_symbolic_error(expr,variables):
    '''
    expr: give the mathematical expression as a string that can be converted into sympy
    variables: give a list of strings for the variables which will be made sympy symbols
    '''
    symbol_vars = {_: sp.Symbol(_) for _ in variables}
    error_vars  = {_: sp.Symbol('\delta %s'%_) for _ in variables}
     
    expr = sp.sympify(expr).subs([(_,symbol_vars[_]) for _ in variables])
    
    err = sp.S(0)
    for _ in variables:
        err += (expr.diff(symbol_vars[_])*error_vars[_])**2
    err = sp.sqrt(err)

    return err

def get_error(expr,variables):
    '''
    expr: give the mathematical expression as a string that can be converted into sympy
    variables: give a dictionary with variables as strings for keys and a list where the
        first entry is an array of measurements and the second entry is an array of errors
        or a single value if constant error for each measurement
    '''
    # need to implement units with pint
    variable_list = list(variables.keys())
    symbol_list   = []
    param_list    = []
    for _ in variable_list:
        symbol_list.append(sp.Symbol(_))
        param_list.append(Q_(variables[_][0],variables[_][-1]))
        
        symbol_list.append(sp.Symbol('\delta %s'%_))
        param_list.append(Q_(variables[_][1],variables[_][-1]))        
        
    err_sym_expr = get_symbolic_error(expr,variable_list)
    err_num_expr = sp.lambdify(symbol_list,err_sym_expr,'numpy')
    
    return err_num_expr(*param_list)

def get_nominal(expr,variables):
    # need to implement units with pint
    variable_list = list(variables.keys())
    symbol_list   = []
    param_list    = []
    for _ in variable_list:
        symbol_list.append(sp.Symbol(_))
        param_list.append(Q_(variables[_][0],variables[_][-1]))   
        
    nom_sym_expr = sp.sympify(expr).subs([(var_,sym_) for var_,sym_ in zip(variable_list,symbol_list)])
    nom_num_expr = sp.lambdify(symbol_list,nom_sym_expr,'numpy')
    
    return nom_num_expr(*param_list)

def get_nominal_error(expr,variables):
    nom = get_nominal(expr,variables)
    err = get_error(expr,variables)
    return nom,err

def print_err_report(expr,variables,desired_units=''):
    nom,err = get_nominal_error(expr,variables)
    print('{:P} +- {:P}'.format(nom,err))