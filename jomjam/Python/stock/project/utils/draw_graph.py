import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def draw_corr_heatmap(df, map_opt, figsize=(10,10)):
    corr_df = df.corr()
    if map_opt:
        fig, ax = plt.subplots( figsize=figsize )
        mask = np.zeros_like(corr_df, dtype=np.bool)
        mask[np.triu_indices_from(mask)] = True
        sns.heatmap(corr_df, 
                    cmap = 'RdYlBu_r',
                    #annot=True,
                    mask=mask, 
                    linewidths=.5,  
                    cbar_kws={"shrink": .5},
                    vmin = -1,vmax = 1,
                    #fmt=".3f"
                )  
        plt.show()
    else:
        print(corr_df)

