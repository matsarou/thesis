import seaborn as sns

def pairplot(df, x_vars=[], y_vars=[]):
    if len(x_vars)==0 or len(y_vars)==0:
        sns.pairplot(df)
    else:
        sns.pairplot(data=df,
                 x_vars=x_vars,
                 y_vars=y_vars)

def box_plot(df):
    df.plot.box()