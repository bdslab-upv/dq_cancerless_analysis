import pandas as pd
import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, IterativeImputer
from sklearn.neighbors import KernelDensity
from sklearn.manifold import Isomap
from sklearn.neighbors import LocalOutlierFactor
from itertools import product
from collections import Counter
# Alternative PCA version with more info
from pca import pca
import plotly.express as px
import plotly.graph_objects as go

def merge_questions(df: pd.DataFrame) -> pd.DataFrame:
    """
    The datasets downloaded in this version have different columns per translations
    the idea is to merge this columns
    """
    new_data = []
    # iterate over the dataframe as a list of dicts
    for row in df.to_dict('records'):
        new_row = {}
        for col, val in row.items():
            # if the value is missing, ignore it
            if pd.isnull(val):
                continue
            # check if the column name starts with a number
            if col.strip()[0].isnumeric():
                question_number = int(col.split(')')[0].strip())
                new_row[question_number] = val
            # this is not a question, therefore just copy what was already there
            else:
                new_row[col] = val
        new_data.append(new_row)
    new_df = pd.DataFrame(new_data)
    nums = sorted([el for el in new_df.columns.values if isinstance(el, int)])
    sorted_headers = ['id', 'internal_code', 'pilot'] + nums
    return new_df.reindex(sorted_headers, axis=1)

def split_per_pilot(df, pilot_col):
    # create a determinist order!
    pilot_set = sorted(list(set(pilot_col)))
    output = []
    for pilot in pilot_set:
        output.append(df[pilot_col == pilot])
    return output

def filter_site_outliers(arr):
    lof = LocalOutlierFactor()
    outlier_labels = lof.fit_predict(arr)
    # filter based on the labels
    filtered_arr = arr[outlier_labels == 1]
    return filtered_arr


def preprare_data_DQ_count(sites_df, dim_reduction='PCA', missing_treatment=-1,
                           n_bins=10, outlier_filtering=False):
    if dim_reduction == 'PCA':
        dim_reductor = pca(n_components=3, normalize=True, verbose=0)
    elif dim_reduction == 'Isomap':
        dim_reductor = Isomap(n_components=3)
    else:
        raise ValueError(f'Unvalid dimensionality reduction technique: {dim_reduction}')

    headers = sites_df[0].columns.values
    filled_sites_df = []
    # Deal with the missings - modifying the inner state of the DFs, maybe working with a copy(?)
    if missing_treatment == 'impute':
        # imputer = SimpleImputer(strategy="median")
        imputer = IterativeImputer()
        imputer.fit(pd.concat(sites_df))
        # Critical - Since python 3.6 dicts are ordered deterministicly, therefore this should be ok
        for df in sites_df:
            imputed_matrix_ = imputer.transform(df)
            filled_sites_df.append(pd.DataFrame(data=imputed_matrix_, columns=headers))

    elif isinstance(missing_treatment, (int, float)):
        # Avoid the SettingWithCopyWarning
        for df in sites_df:
            filled_sites_df.append(df.fillna(missing_treatment))
    else:
        raise ValueError(f'Missing treatment value error: {missing_treatment}')

    # Concat and apply PCA // IMPORTANT - the pca library does not have fit as standalone method
    _ = dim_reductor.fit_transform(pd.concat(filled_sites_df))
    # apply the transformation to the different sites
    site_vector = []
    for df in filled_sites_df:
        low_dim_df_ = np.array(dim_reductor.transform(df))
        # IMPORTANT - outlier filtering PER SITE
        if outlier_filtering:
            low_dim_df_ = filter_site_outliers(low_dim_df_)
        flatten_histogram = np.histogramdd(low_dim_df_, bins=n_bins)[0].reshape(1, -1)
        site_vector.append((flatten_histogram / flatten_histogram.sum())[0])
    return site_vector


def plot_scatter_3d(df, pilot_col, outlier_filtering=False, width=800, height=800):
    if df.shape[1] < 3:
        print(f"Dimensions of the DF should at least be 3. Current dimension: {df.shape[1]}")
        return
    colnames = df.columns.values
    imputer = IterativeImputer()
    imputed_df = pd.DataFrame(data=imputer.fit_transform(df), columns=colnames)
    pca_output = pca(n_components=3, normalize=True, verbose=0).fit_transform(imputed_df)
    pca_data = pca_output.get('PC')
    print(f'Explained variance: {sum(pca_output.get("explained_var"))}')
    pca_data['pilot'] = pilot_col
    if outlier_filtering:
        non_outliers_dfs = []
        dataframes = split_per_pilot(pca_data, pca_data.pilot)
        # For every site dataframe, left only the x, y and z variables to calculate outliers
        for df in dataframes:
            actual_pilot = df.pilot.iloc[0]
            df_not_outlier = filter_site_outliers(df.drop('pilot', axis=1))
            # after that, add again the pilot information and merge all the datasets
            df_not_outlier['pilot'] = actual_pilot
            non_outliers_dfs.append(df_not_outlier)
        pca_data = pd.concat(non_outliers_dfs)
    # pca_data['identifier'] = df.index
    fig = px.scatter_3d(pca_data, x="PC1", y="PC2", z="PC3", color="pilot",
                        title="PCA 3D data", hover_data=["PC1", "PC2", "pilot"], width=width, height=height)

    return fig


def plot_scatter_2d(df, pilot_col_, outlier_filtering=False, width=800, height=800):
    if df.shape[1] < 2:
        print(f"Dimensions of the DF should at least be 2. Current dimension: {df.shape[1]}")
        return
    colnames = df.columns.values
    imputer = IterativeImputer()
    imputed_df = pd.DataFrame(data=imputer.fit_transform(df), columns=colnames)
    pca_output = pca(n_components=2, normalize=True, verbose=0).fit_transform(imputed_df)
    pca_data = pca_output.get('PC')
    #print(f'Explained variance: {sum(pca_output.get("explained_var"))}')
    pca_data['pilot'] = pilot_col_
    if outlier_filtering:
        non_outliers_dfs = []
        dataframes = split_per_pilot(pca_data, pca_data.pilot)
        # For every site dataframe, left only the x, y and z variables to calculate outliers
        for df in dataframes:
            actual_pilot = df.pilot.iloc[0]
            df_not_outlier = filter_site_outliers(df.drop('pilot', axis=1))
            # after that, add again the pilot information and merge all the datasets
            df_not_outlier['pilot'] = actual_pilot
            non_outliers_dfs.append(df_not_outlier)
        pca_data = pd.concat(non_outliers_dfs)


    fig = px.scatter(pca_data, x="PC1", y="PC2", color="pilot",
                     title="PCA 2D data", labels={"PC1": "1st comp", "PC2": "2nd comp"}, hover_data=["PC1", "PC2", "pilot"], width=width, height=height)

    fig.update_layout(title=f"PCA projected data in 2D",
                      title_x=0.5,
                      font=dict(
                          family="Courier New, monospace",
                          size=14,
                      ),
                      legend=dict(
                          title="Pilot"
                      ))

    return fig

def plotMSV(msvMetrics: dict, n_by_source: list[int], label_sources: list[str], title: str, height=800, width=800):
    sphere_max_size = 100
    scale_factor = sphere_max_size / max(n_by_source)
    fig = px.scatter_3d(x=msvMetrics['Vertices'][:, 0],
                        y=msvMetrics['Vertices'][:, 1],
                        z=msvMetrics['Vertices'][:, 2],
                        color=label_sources,
                        size=[n * scale_factor for n in n_by_source],
                        size_max=sphere_max_size,
                        text=label_sources,
                        height=height,
                        width=width
                        )
    fig.update_layout(title=title,
                      title_x=0.5,
                      font=dict(
                          family="Courier New, monospace",
                          size=14,
                      ),
                      legend=dict(
                          title="Pilot"
                      )
                      )
    return fig


def plotMSV2d(msvMetrics: dict, n_by_source: list[int], label_sources: list[str], title: str, height=800, width=800):
    sphere_max_size = 100
    scale_factor = sphere_max_size / max(n_by_source)
    fig = px.scatter(x=msvMetrics['Vertices'][:, 2],#x=msvMetrics['Vertices'][:, 0],
                     y=msvMetrics['Vertices'][:, 1],

                     color=label_sources,
                     size=[n * scale_factor for n in n_by_source],
                     size_max=sphere_max_size,
                     text=label_sources,
                     height=height,
                     width=width,
                     labels={'x': '1st comp',
                             'y': '2nd comp'}
                     )
    fig.update_layout(title=title,
                      title_x=0.5,
                      font=dict(
                          family="Courier New, monospace",
                          size=14,
                      ),
                      legend=dict(
                          title="Pilot"
                      ),
                      template="seaborn"
                      )
    return fig

def plotMSV2d_questionnaire(msvMetrics: list, n_by_source: list[int], label_sources: list[str], title: str, height=800, width=800):
    sphere_max_size = 100
    scale_factor = sphere_max_size / max(n_by_source)
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'cyan', 'magenta', 'yellow', 'grey']

    questionnaires = ['Socio-demographic','Health Literacy','Health Data', 'Risk behaviours and healthy lifestyles',
                      'Psychological distress', 'Quality of life', 'Health Care Empowerment','Interpersonal Communication','Use of health care services']
    label_sources_reduction = ['AUS', 'GRE', 'SPA', 'UK']
    fig = go.Figure()
    for i in range(len(msvMetrics)):
        scatter = go.Scatter(
            x=msvMetrics[i]['Vertices'][:, 2],
            y=msvMetrics[i]['Vertices'][:, 1],
            mode='markers+text',
            marker=dict(
                size=50,#[n * scale_factor for n in n_by_source],
                sizemode='diameter',
                #sizeref=2. * max(n_by_source) / (sphere_max_size ** 2),
                #sizeref=(2.*max(n_by_source)) / (sphere_max_size),
                #sizemin=4,
                #sizemin=4,
                color=colors[i % len(colors)],
                colorscale='Viridis',
                showscale=False
            ),
            text=label_sources_reduction,
            textposition='middle center',
            textfont=dict(
                family="sans serif",
                size=10,
                color="Black"
            ),
            name=f'{questionnaires[i]}'
        )
        fig.add_trace(scatter)
        fig.update_layout(
            xaxis_title='1st comp',
            yaxis_title='2nd comp',
            height=height,
            width=width,

            title=dict(
                text=title,
                x=0.5,
                xanchor='center',
                yanchor='top',
                font=dict(
                    family="Arial, sans-serif",
                    size=24,
                    color='black'
                )
            ),
            #legend_title="Questionnaires",
            legend = dict(
                title="Questionnaires",
                font=dict(
                    family="Arial, sans-serif",
                    size=14,
                    color='black'
                ),
                #bgcolor='rgba(255, 255, 255, 0.5)',
                bgcolor='rgba(0,0,0,0)',
                bordercolor='black',
                borderwidth=1
            ),
            template="seaborn"
        )

    return fig

import plotly.graph_objects as go

def charts(title: str, height=800, width=800, missing=False):
    colors = ['#636EFA', '#EF553B', '#00CC96', '#AB63FA']
    questionnaires = ['Complete','Socio-demographic', 'Health Literacy', 'Health Data', 'Risk behaviours and healthy lifestyles',
                      'Psychological distress', 'Quality of life', 'Health Care Empowerment',
                      'Interpersonal Communication', 'Use of health care services']
    if missing:
        gpds = [0.933, 0.876, 0, 0, 0.877, 0, 0.593, 0, 0.603, 0.888]

        spos = [0.75,	0.771,	0.787,	0.839 ,0.877, 0.601, 0.877, 0.601,0, 0, 0, 0, 0,0,0,0, 0.647, 0.753, 0.879, 0.679,0,0,0,0,0.333,0.333,1,0.333,0,0,0,0,0.35,
                0.339, 0.997,0.349,0.652,0.824,0.846,0.673]
    else:
        gpds = [0.833, 0.882, 0.897, 0.452, 0.789, 0.854, 0.838, 0.905, 0.911]
        spos = [0.692,	0.701,	0.706,	0.711, 0.791, 0.714, 0.785, 0.727, 0.854, 0.839, 0.663, 0.671, 0.473, 0.291, 0.273, 0.488, 0.691, 0.649, 0.682,
                0.638, 0.782, 0.674, 0.724, 0.701, 0.746, 0.662, 0.701, 0.717, 0.778, 0.721, 0.776, 0.778, 0.798, 0.721,
                0.836, 0.718, 0.801, 0.732, 0.849, 0.703]

    # Seleccionar 4 elementos de spos para cada cuestionario
    spos_per_questionnaire = [spos[i * 4:(i + 1) * 4] for i in range(len(questionnaires))]

    fig = go.Figure()
    pilots = ['Austria','Greece', 'Spain', 'United Kingdom']
    # Agregar la barra de GPD
    fig.add_trace(go.Bar(
        x=questionnaires,
        y=gpds[:len(questionnaires)],
        name='GPD',
        marker_color='black'
    ))

    # Agregar las barras de SPO
    for i in range(4):
        fig.add_trace(go.Bar(
            x=questionnaires,
            y=[spos_per_questionnaire[j][i] if i < len(spos_per_questionnaire[j]) else None for j in range(len(questionnaires))],
            name=f'SPO {pilots[i]}',
            marker_color=colors[i % len(colors)]
        ))

    fig.update_layout(
            height=height,
            width=width,
            title=dict(
                text=title,
                x=0.5,
                xanchor='center',
                yanchor='top',
                font=dict(
                    family="Arial, sans-serif",
                    size=24,
                    color='black'
                )
            ),
            legend = dict(
                title="Pilots",
                font=dict(
                    family="Arial, sans-serif",
                    size=14,
                    color='black'
                ),
                #bgcolor='rgba(255, 255, 255, 0.5)',
                bgcolor='rgba(0,0,0,0)',
                bordercolor='black',
                borderwidth=1
            ),
            template="seaborn"
        )

    return fig


def spider (title: str, height=800, width=800, missing=False):
    colors = ['#636EFA', '#EF553B', '#00CC96', '#AB63FA']
    questionnaires = ['Socio-demographic', 'Health Literacy', 'Health Data', 'Risk behaviours and healthy lifestyles',
                      'Psychological distress', 'Quality of life', 'Health Care Empowerment',
                      'Interpersonal Communication', 'Use of health care services']
    fig = go.Figure()

    if missing:
        gpds = [0.876, 0, 0, 0.877, 0, 0.593, 0, 0.603, 0.888]

        spos = [0.877, 0.601, 0.877, 0.601,0, 0, 0, 0, 0,0,0,0, 0.647, 0.753, 0.879, 0.679,0,0,0,0,0.333,0.333,1,0.333,0,0,0,0,0.35,
                0.339, 0.997,0.349,0.652,0.824,0.846,0.673]
    else:
        gpds = [0.882, 0.897, 0.452, 0.789, 0.854, 0.838, 0.905, 0.911]
        spos = [0.791, 0.714, 0.785, 0.727, 0.854, 0.839, 0.663, 0.671, 0.473, 0.291, 0.273, 0.488, 0.691, 0.649, 0.682,
                0.638, 0.782, 0.674, 0.724, 0.701, 0.746, 0.662, 0.701, 0.717, 0.778, 0.721, 0.776, 0.778, 0.798, 0.721,
                0.836, 0.718, 0.801, 0.732, 0.849, 0.703]

    # Seleccionar 4 elementos de spos para cada cuestionario
    spos_per_questionnaire = [spos[i * 4:(i + 1) * 4] for i in range(len(questionnaires))]

    pilots = ['Austria','Greece', 'Spain', 'United Kingdom']
    fig.add_trace(go.Scatterpolar(
        # x=questionnaires,
        # y=gpds[:len(questionnaires)],
        r=gpds[:len(questionnaires)],
        theta=questionnaires,
        name='GPD',
        marker_color='black'
    ))

    # Agregar las barras de SPO
    for i in range(4):
        fig.add_trace(go.Scatterpolar(
            r=[spos_per_questionnaire[j][i] if i < len(spos_per_questionnaire[j]) else None for j in range(len(questionnaires))],
            theta=questionnaires,
            name=f'SPO {pilots[i]}',
            #fill='toself',

            #marker_color=colors[i % len(colors)]
        ))

    # Layout simplificado
    fig.update_layout(
            height=height,
            width=width,
            title=dict(
                text=title,
                x=0.5,
                xanchor='center',
                yanchor='top',
                font=dict(
                    family="Arial, sans-serif",
                    size=24,
                    color='black'
                )
            ),
            legend = dict(
                title="Pilots",
                font=dict(
                    family="Arial, sans-serif",
                    size=14,
                    color='black'
                ),
                #bgcolor='rgba(255, 255, 255, 0.5)',
                bgcolor='rgba(0,0,0,0)',
                bordercolor='black',
                borderwidth=1
            ),
            margin=dict(
                l=200,
                # r=50,
                # b=100,
                # t=100,
                # pad=4
            ),
            template="seaborn"
        )

    return fig


def complete_chart(title: str, height=800, width=800, missing=False):
    colors = ['#636EFA', '#EF553B', '#00CC96', '#AB63FA']
    questionnaires = ['Austria', 'Greece', 'Spain', 'United Kingdom']

    if missing:
        gpds = [0.933]

        spos = [0.75, 0.771, 0.787, 0.839]
    else:
        gpds = [0.833]
        spos = [0.692, 0.701, 0.706, 0.711]

    # Seleccionar 4 elementos de spos para cada cuestionario
    #spos_per_questionnaire = [spos[i * 4:(i + 1) * 4] for i in range(len(questionnaires))]

    fig = go.Figure()
    pilots = ['Austria', 'Greece', 'Spain', 'United Kingdom']
    # Agregar la barra de GPD
    fig.add_trace(go.Bar(
        x=questionnaires,
        y=gpds[:len(questionnaires)],
        name='GPD',
        marker_color='black'
    ))

    # Agregar las barras de SPO
    for i in range(4):
        fig.add_trace(go.Bar(
            x=questionnaires,
            #y=[spos_per_questionnaire[j][i] if i < len(spos_per_questionnaire[j]) else None for j in range(len(questionnaires))],
            y = spos[:len(questionnaires)],
            name=f'SPO {pilots[i]}',
            marker_color=colors[i % len(colors)]
        ))

    # Layout simplificado
    fig.update_layout(
        height=height,
        width=width,
        title=dict(
            text=title,
            x=0.5,
            xanchor='center',
            yanchor='top',
            font=dict(
                family="Arial, sans-serif",
                size=24,
                color='black'
            )
        ),
        legend=dict(
            title="Pilots",
            font=dict(
                family="Arial, sans-serif",
                size=14,
                color='black'
            ),
            # bgcolor='rgba(255, 255, 255, 0.5)',
            bgcolor='rgba(0,0,0,0)',
            bordercolor='black',
            borderwidth=1
        ),
        template="seaborn"
    )

    return fig



def show_better_results(gpd_res: dict) -> None:
    countries_ = ['Austria', 'Greece', 'Spain', 'United Kingdom']
    print(f'GPD: {round(gpd_res["GPD"], 3)}')
    for c, spo in zip(countries_, gpd_res['SPOs']):
        print(f'{c}: {round(spo, 3)}')


# if __name__ == '__main__':
#
#     # Load the data to run the experiment
#     extraction_date = '2024_01_02'
#     t0_path = f'/home/ravn/datasets/cidma/{extraction_date}/export_T0_All.csv'
#     df_t0 = pd.read_csv(t0_path, sep=';')
#     questions_t0 = pd.read_csv('questionnaires/t0.csv', sep=';')
#     complete_df = merge_questions(df_t0)
#
#     discard_questions = []
#     for row in questions_t0.itertuples():
#         if row.type in {'multiple', 'text', 'longtext'}:
#             discard_questions.append(row.number)
#
#     clean_complete_df = complete_df.drop(discard_questions + ['id', 'pilot', 'internal_code'], axis=1)
#
#     countries = set(complete_df.pilot)
#     dfs_country = {}
#     for country in countries:
#         dfs_country[country] = clean_complete_df[complete_df.pilot == country]
#
#     prepared_data_count = preprare_data_DQ_count(list(dfs_country.copy().values()), dim_reduction='PCA',
#                                                  missing_treatment='impute', outlier_filtering=True)
#     msv_results_count = estimateMSVmetrics(np.column_stack(prepared_data_count))