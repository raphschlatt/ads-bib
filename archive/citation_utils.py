import pandas as pd
from collections import defaultdict, Counter
import itertools
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sqlite3
import os

def filter_nodes(df_nodes, df_edges, edge_columns):
    """
    This function filters the nodes DataFrame to include only those nodes that are present in the edges DataFrame.

    Args:
    df_nodes (pandas.DataFrame): DataFrame containing node data.
    df_edges (pandas.DataFrame): DataFrame containing edge data.
    edge_columns (list): List of columns in edges DataFrame to check for unique nodes.

    Returns:
    df_filtered_nodes (pandas.DataFrame): Filtered DataFrame containing only the nodes present in the edges DataFrame.
    """
    # Get unique nodes from edges
    unique_nodes = pd.unique(df_edges[edge_columns].values.ravel('K'))

    # Filter nodes DataFrame
    df_filtered_nodes = df_nodes[df_nodes['id'].isin(unique_nodes)]

    return df_filtered_nodes

def process_db(db_name, edge_table_cols, df_edges, df_nodes):
    """
    This function processes the database, creating the required tables and inserting data from DataFrames.

    Args:
    db_name (str): Name of the SQLite database file.
    edge_table_cols (str): Column definitions for the 'edges' table.
    df_edges (pandas.DataFrame): DataFrame containing edge data.
    df_nodes (pandas.DataFrame): DataFrame containing node data.
    """
    # Create a connection to the SQLite database file.
    conn = sqlite3.connect(db_name)
    c = conn.cursor()

    # Create two tables in the database, if they don't already exist.
    c.execute(f'CREATE TABLE IF NOT EXISTS edges ({edge_table_cols})')
    c.execute('CREATE TABLE IF NOT EXISTS nodes (id text, Author text, Title text, Year number, Journal text, Issue number, Volume number, Category text, Abstract text, Keywords text)')

    # Commit the changes to the database.
    conn.commit()

    # Remove unnecessary columns from df_nodes
    if 'References' in df_nodes.columns:
        df_nodes = df_nodes.drop(columns=['References'])
    if 'tokens' in df_nodes.columns:
        df_nodes = df_nodes.drop(columns=['tokens'])

    # Write the DataFrames to their corresponding tables in the database.
    df_edges.to_sql('edges', conn, if_exists='replace', index=False)
    df_nodes.to_sql('nodes', conn, if_exists='replace', index=False)

    # Execute SELECT queries on both tables to verify that the data has been written correctly.
    c.execute('SELECT * FROM edges')
    c.execute('SELECT * FROM nodes')

    conn.close()

def save_to_csv(df_edges, df_nodes, directory):
    """
    This function saves the DataFrames to CSV files in the specified directory.

    Args:
    df_edges (pandas.DataFrame): DataFrame containing edge data.
    df_nodes (pandas.DataFrame): DataFrame containing node data.
    directory (str): Directory where the CSV files will be saved.
    """
    # Create the folder if it doesn't already exist.
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Write the DataFrames to their corresponding CSV files in the specified directory.
    df_edges.to_csv(f"{directory}/edges.csv", index=False)
    df_nodes.to_csv(f"{directory}/nodes.csv", index=False)


