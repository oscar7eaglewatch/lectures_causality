import numpy as np
import graphviz
import pandas as pd
import lingam as gs_cd

from sklearn.linear_model import LassoCV

def make_graph(adjacency_matrix, labels=None):
    idx = np.abs(adjacency_matrix) > 0.01
    dirs = np.where(idx)
    dag = graphviz.Digraph(engine='dot')
    names = labels if labels else [f'x{i}' for i in range(len(adjacency_matrix))]
    for to, from_, coef in zip(dirs[0], dirs[1], adjacency_matrix[idx]):
        dag.edge(names[from_], names[to], label=f'{coef:.2f}')
    return dag

def get_total_effect(model, X, cause_index, result_index):
    #cause = 0
    #result = 3
    te = model.estimate_total_effect(X, cause_index, result_index)
    print(f'total effect: {te:.3f}')


def get_intervention_effect(model, X, target_index):
    features = [i for i in range(X.shape[1]) if i != target_index]
    
    reg = LassoCV(cv=7, random_state=0)
    reg.fit(X.iloc[:, features], X.iloc[:, target_index])
    
    ce = gs_cd.CausalEffect(model)
    effects = ce.estimate_effects_on_prediction(X, target_index, reg)

    df_effects = pd.DataFrame()
    df_effects['feature'] = X.columns
    df_effects['effect_plus'] = effects[:, 0]
    df_effects['effect_minus'] = effects[:, 1]
    
    max_index = np.unravel_index(np.argmax(effects), effects.shape)
    max_intervention_element_name = X.columns[max_index[0]]
    
    intervention_data = {"reg_model": reg, 
                         "causal_effect": ce, 
                         "intervention_effect": df_effects, 
                         "max_intervention_element_name": max_intervention_element_name}
    
    return intervention_data                          


def get_bootstrapping_prediction(X, cause_index, target_index, n_sampling=100, min_causal_effect=0.01):
    model = gs_cd.DirectLiNGAM()
    result = model.bootstrap(X, n_sampling=n_sampling)
    
    causal_effects = result.get_total_causal_effects(min_causal_effect=min_causal_effect)

    dfc = pd.DataFrame(causal_effects)
    
    labels = list(X)
    dfc['from'] = dfc['from'].apply(lambda x : labels[x])
    dfc['to'] = dfc['to'].apply(lambda x : labels[x])
    
    df_interested = dfc[dfc['to']==labels[target_index]]
    
    causal_path = pd.DataFrame(result.get_paths(cause_index, target_index))    
    
    boostrapping_result = {"from-to-df": dfc,
                           "top-70%": dfc[dfc['probability'] > 0.7].sort_values('probability', ascending=False),
                           "effect-on-target": df_interested,
                           "causal-path": causal_path}
    
    return boostrapping_result    


def get_optimal_intervention(X, intervention_data, intervention_index, target_index, target_value):
    return intervention_data["causal_effect"].estimate_optimal_intervention(X, 
                                                                            target_index, 
                                                                            intervention_data["reg_model"], 
                                                                            intervention_index, 
                                                                            target_value)


def get_label_processing(X):
    initial_labels = list(X)
    labels = []
    for i, each in enumerate(initial_labels):
        labels.append(str(i) + '. ' + each)

        if i == 0:
            df2 = X.rename(columns={each: str(i) + '. ' + each})
        else:
            df2.rename(columns={each: str(i) + '. ' + each}, inplace=True)
            
    return labels, df2   


def get_causal_model():
    return gs_cd.DirectLiNGAM()