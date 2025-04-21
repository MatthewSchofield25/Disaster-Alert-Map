import dash
import dash_bootstrap_components.themes
import dash_bootstrap_components as dbc
from click import style
from dash import dcc, html
import plotly.graph_objs as go
import pandas as pd
import pyodbc
import os
from sklearn.neighbors import NearestNeighbors
import numpy as np
import pyodbc
from dotenv import load_dotenv
import sys

#Initialize Database Connection
driver = '{ODBC Driver 18 for SQL Server}'
server = os.getenv("DATABASE_SERVER")
database = os.getenv("DATABASE_NAME")
username = os.getenv("DATABASE_USERNAME")
password = os.getenv("DATABASE_PASSWORD")

# Initialize the Dash app
app = dash.Dash(__name__, 
                external_stylesheets=[dash_bootstrap_components.themes.BOOTSTRAP],
                meta_tags = [
                    {"name": "viewport", "content": "width=device-width, initial-scale=1"},
                    {"name": "description", "content": "Natural Disaster Sentiment Analysis"},
                    {"name": "keywords", "content": "disaster, sentiment, analysis"},
                    # Add this for smooth scrolling:
                    {"name": "scroll-behavior", "content": "smooth"}
                ]
)

register_page(
    __name__,
    "/",
    title="Home",
    name="Home"
)

valid_categories = {'Flood', 'Drought', 'Landslide', 'Volcano', 'Blizzard',
                        'Earthquake', 'Tsunami', 'Wildfire', 'Hurricane', 'Tornado', 'Other'}


# # CODE TO USE FOR LATER ONCE DATABASE IS SET UP
try:
    db = pyodbc.connect(f'DRIVER={driver};SERVER={server};DATABASE={database};UID={username};PWD={password}')
except Exception as e:
    print(f"Error connecting to SQL Server: {e}")
    sys.quit()
print("SQL Server Connection Successful")


# # LATER DEVELOPMENT: UTILIZE THE TIMEPOSTED TO ONLY GET THE LAST 31 DAYS OF DATA
cursor = db.cursor()
cursor.execute("SELECT * FROM LSTM_Posts")
results = cursor.fetchall()
columns = [column[0] for column in cursor.description]
df = pd.DataFrame.from_records(results, columns=columns)
df['location'] = df['location'].replace({'None': np.nan})

# # Define emergency thresholds
# emergency_sentiment_threshold = 0  # Adjust as needed
# emergency_post_count_threshold = 0  # Adjust as needed

# # Add emergency flag
# location_stats['emergency'] = (
#     (location_stats['avg_sentiment'] <= emergency_sentiment_threshold) |
#     (location_stats['post_count'] >= emergency_post_count_threshold)
# )

def get_most_frequent_location(group):
    mode_series = group['location'].mode()
    most_frequent_location = mode_series.iloc[0] if not mode_series.empty else None
    avg_sentiment = group['sentiment_score'].mean()
    post_count = group['location'].count()
    return pd.Series([most_frequent_location, avg_sentiment, post_count], index=['location', 'avg_sentiment', 'post_count'])

def applyMostFrequentLocationCoords(df): # Accept the DataFrame and the frequent locations
    frequent_location_data = pd.read_csv("data/Generalized_UnitedStatesOfAmerica.csv")
    frequent_location_data['location'] = frequent_location_data.apply(
        lambda row: row['name'] if pd.notna(row['name']) else row['state'],
        axis=1
    )
    merged_name = pd.merge(df, frequent_location_data[['name', 'LATITUDE', 'LONGITUDE']],
                           left_on='location', right_on='name', how='left', suffixes=('', '_name'))

    # Merge on 'state' (e.g., "AZ")
    merged_state = pd.merge(df, frequent_location_data[['state', 'LATITUDE', 'LONGITUDE']],
                            left_on='location', right_on='state', how='left', suffixes=('', '_state'))

    # Add both match columns back into df
    df['LATITUDE'] = merged_name['LATITUDE'].combine_first(merged_state['LATITUDE'])
    df['LONGITUDE'] = merged_name['LONGITUDE'].combine_first(merged_state['LONGITUDE'])
    return df

grouped_before_fillna = df.groupby('category')

most_frequent_locations = grouped_before_fillna.apply(get_most_frequent_location, include_groups=False).reset_index()

posts_generalized = applyMostFrequentLocationCoords(most_frequent_locations.copy()) # Pass copies

posts_generalized_cleaned = posts_generalized.dropna()
posts_generalized_cleaned = posts_generalized_cleaned.reset_index(drop=True)

# Create a category-to-location mapping from df2
category_location_map = posts_generalized_cleaned.set_index('category')['location'].to_dict()

# Update df1's location based on the mapping
df['location'] = df['category'].map(category_location_map)
df.dropna()

# Mapbox token
mapbox_access_token = os.getenv("MAP_ACCESSKEY")

def generate_geo_map(geo_data):
    if geo_data.empty:
        return {"data": [], "layout": {}}

# Modified generate_geo_map function
def generate_geo_map(geo_data):
    if geo_data.empty:
        return {"data": [], "layout": {}}

    lat = geo_data["LATITUDE"].tolist()
    lon = geo_data["LONGITUDE"].tolist()
    avg_sentiment = geo_data["avg_sentiment"].tolist()
    post_count = geo_data["post_count"].tolist()
    location = geo_data['location'].tolist()
    disaster_type = geo_data['category'].tolist()

    disasters = []

    for i in range(len(lat)):
        size = 10 * (1 - avg_sentiment[i]) + post_count[i]
        color = "#21c7ef"

        disaster = go.Scattermap(
            lat=[lat[i]],
            lon=[lon[i]],
            mode="markers",
            marker=dict(
                color=color,
                size=size,
            ),
            opacity=0.8,
            customdata=[location[i], disaster_type[i]],
            hoverinfo="text",
            text=f"{location[i]} ({disaster_type[i]})<br>Avg Sentiment: {avg_sentiment[i]:.2f}<br>Post Count: {post_count[i]}"
        )
        disasters.append(disaster)

    layout = go.Layout(
        margin=dict(l=10, r=10, t=20, b=10, pad=5),
        geo = dict(
            scope = 'usa',
            landcolor = 'rgb(217, 217, 217)',
        ),
        plot_bgcolor="#171b26",
        paper_bgcolor="#171b26",
        clickmode="event+select",
        hovermode="closest",
        showlegend=False,
        mapbox=go.layout.Mapbox(
            accesstoken=mapbox_access_token,
            bearing=0,
            center=go.layout.mapbox.Center(
                lat = 37.0902,
                lon = -95.7129
            ),
            pitch=0,
            zoom=3,
            style="mapbox://styles/plotlymapbox/cjvppq1jl1ips1co3j12b9hex",
            bounds=dict(west=-125, east=-66, south=24, north=50),  # Continental US
        ),
    )
    return {"data": disasters, "layout": layout}
def create_sentiment_graph(filtered_df, group_by='category'):
    """
    Creates a sentiment trend graph for a specific location and disaster type

    Args:
        filtered_df: DataFrame containing columns 'timeposted', 'sentiment_score',
                    'location', and 'disaster_type'

    Returns:
        plotly.graph_objects.Figure: The generated sentiment graph
    """
    fig = go.Figure()
    if group_by == 'category':
        loc_df = filtered_df.groupby(['timeposted', 'location'])['sentiment_score'].mean().reset_index()
        for location in loc_df['location'].unique():
            temp_df = loc_df[loc_df['location'] == location]
            fig.add_trace(
                go.Scatter(
                    x=temp_df['timeposted'],
                    y=temp_df['sentiment_score'],
                    mode='lines',
                    line=dict(width=1, color='rgba(150,150,150,0.2)'),  # Light gray, semi-transparent
                    name=location,
                    showlegend=False,
                    hoverinfo='skip'  # Skip hover for individual locations
                )
            )
        # Then add the category average line (bold and prominent)
        avg_df = filtered_df.groupby(['timeposted', 'category'])['sentiment_score'].mean().reset_index()
        fig.add_trace(
            go.Scatter(
                x=avg_df['timeposted'],
                y=avg_df['sentiment_score'],
                mode='lines+markers',
                line=dict(width=3, color='#1f77b4'),
                marker=dict(size=8),
                name='Category Average',
                hovertemplate=(
                    '<b>Date</b>: %{x|%Y-%m-%d}<br>'
                    '<b>Avg Sentiment</b>: %{y:.2f}<extra></extra>'
                )
            )
        )
        #add .iloc[0] to avg_df['category'] to get the first value
        title = f"Sentiment Trend with '{avg_df['category'].iloc[0]}' Average"


    # for the top 4 disaster part of
    elif group_by == 'location':
        # Aggregate by location and date
        agg_df = filtered_df.groupby(['timeposted', 'location'])['sentiment_score'].mean().reset_index()

        # Add a trace for each location
        for location in agg_df['location'].unique():
            loc_df = agg_df[agg_df['location'] == location]
            fig.add_trace(
                go.Scatter(
                    x=loc_df['timeposted'],
                    y=loc_df['sentiment_score'],
                    mode='lines+markers',
                    line=dict(width=2),
                    marker=dict(size=6),
                    name=location,
                    hovertemplate=(
                        '<b>Date</b>: %{x|%Y-%m-%d}<br>'
                        '<b>Sentiment</b>: %{y:.2f}<extra></extra>'
                    )
                )
            )

        title = "Sentiment Trend by Location"
    # for sentiment over time by disaster
    else:
        # Original behavior for specific location/disaster
        fig.add_trace(
            go.Scatter(
                x=filtered_df['timeposted'],
                y=filtered_df['sentiment_score'],
                mode='lines+markers',
                line=dict(color='#1f77b4', width=2),
                marker=dict(size=6),
                name='Sentiment',
                hovertemplate=(
                    '<b>Date</b>: %{x|%Y-%m-%d}<br>'
                    '<b>Sentiment</b>: %{y:.2f}<extra></extra>'
                )
            )
        )

        if not filtered_df.empty:
            location = filtered_df['location'].iloc[0]
            disaster_type = filtered_df['category'].iloc[0]
            title = f'{location} {disaster_type} Sentiment Trend'
        else:
            title = "Sentiment Trend"

    # Add horizontal reference lines
    fig.add_hline(
        y=0,
        line_dash='dash',
        line_color='gray',
        annotation_text="Neutral",
        annotation_position="bottom right"
    )
    fig.add_hline(
        y=1,
        line_dash='dot',
        line_color='green',
        opacity=0.3
    )
    fig.add_hline(
        y=-1,
        line_dash='dot',
        line_color='red',
        opacity=0.3
    )

    fig.update_layout(
        title=title,
        xaxis_title='Date',
        yaxis_title='Sentiment Score',
        yaxis=dict(range=[-1, 1]),
        hovermode='x unified',
        plot_bgcolor='white',
        margin=dict(l=40, r=40, t=60, b=40),
        height=300,
        xaxis=dict(
            # allows for seeing specific time frames
            rangeselector=dict(
                buttons=list([
                    dict(count=7, label="1w", step="day", stepmode="backward"),
                    dict(count=14, label="2w", step="day", stepmode="backward"),
                    dict(step="all")
                ])
            ),
            rangeslider=dict(visible=True),
            type="date"
        )
    )
    return fig

#top 4 sidebar creation

def create_top4_sidebar(top_disasters):
    if not top_disasters:
        return html.Div("No current disasters")

    top_disasters = sorted(top_disasters, key=lambda x: x['rank'])

    return html.Div(
        style={"padding": "20px", "maxWidth": "350px", "margin": "right"},
        children=[
            dbc.Stack(
                [
                    html.H4("Current Disasters:"),
                    *[
                        dbc.Card(
                            [
                                dbc.CardHeader(
                                    html.A(
                                        f"{disaster['location']} {disaster['disaster']}",
                                        href=f"#tab-{disaster['rank']}",
                                        id=f"sidebar-link-{disaster['rank']}",
                                        style={
                                            "color": "inherit",
                                            "text-decoration": "none",
                                            "display": "block",
                                            "width": "200px",
                                            "height": "50px",
                                            "text-align": "left",
                                            "font-size": "clamp(12px,2vw,20px)",
                                            "white-space": "nowrap",
                                            "overflow": "hidden",
                                            "text-overflow": "ellipsis",
                                            "font-weight": "bold",
                                            "padding": "0.5rem"
                                        }
                                    ),
                                    style={"padding": 0}
                                ),
                            ],
                            style={"margin-bottom": "1rem"},
                        )
                        for disaster in top_disasters
                    ]
                ]
            )
        ]
    )

def calculate_top_disasters(df):
    df['timeposted'] = pd.to_datetime(df['timeposted'], errors='coerce')
    df = df.dropna(subset=['timeposted'])

    if df.empty:
        print("No valid data after date cleaning")
        return []

    # Get most recent date in data
    most_recent_date = df['timeposted'].max()

    # Define time windows - comparing recent 10 days vs previous 31 days
    recent_10_days = most_recent_date - pd.Timedelta(days=10)
    comparison_31_days = most_recent_date - pd.Timedelta(days=31)  # 31 days ending 10 days ago

    try:
        # Filter data for time periods
        recent_data = df[df['timeposted'] > recent_10_days]
        comparison_data = df[(df['timeposted'] > comparison_31_days) &
                             (df['timeposted'] <= recent_10_days)]

        # Calculate average sentiment for each period
        avg_recent = recent_data.groupby(['location', 'category'])['sentiment_score'] \
            .mean().reset_index() \
            .rename(columns={'sentiment_score': 'recent_10day_avg'})

        avg_comparison = comparison_data.groupby(['location', 'category'])['sentiment_score'] \
            .mean().reset_index() \
            .rename(columns={'sentiment_score': 'previous_31day_avg'})

        # Merge the averages
        merged = avg_comparison.merge(
            avg_recent,
            on=['location', 'category'],
            how='inner'  # Only include disasters present in both periods
        ).fillna({'recent_10day_avg': 0, 'previous_31day_avg': 0})

        # Calculate metrics
        merged['sentiment_change'] = merged['recent_10day_avg'] - merged['previous_31day_avg']

        # New ranking formula that considers:
        # 1. Current sentiment level (more negative = worse)
        # 2. Magnitude of negative change
        # 3. Recent activity level
        merged['severity_score'] = (
                (1 - merged['recent_10day_avg']) *  # Base severity (0-2 range)
                (1 + (merged['previous_31day_avg'] - merged['recent_10day_avg'])) *  # Change impact
                20  # Scaling factor
        ).round(2)

        # Add mention counts
        mention_counts = recent_data.groupby(['location', 'category']).size()
        merged = merged.join(mention_counts.rename('recent_mentions'), on=['location', 'category'])

        # Filter and rank - require minimum 5 mentions in recent period
        ranked_disasters = merged[merged['recent_mentions'] >= 5] \
            .nlargest(4, 'severity_score')

        return [
            {
                "rank": i + 1,
                "location": row['location'],
                "disaster": row['category'],
                "severity": row['severity_score'],
                "current_sentiment": round(row['recent_10day_avg'], 2),
                "previous_sentiment": round(row['previous_31day_avg'], 2),
                "trend": "↑" if row['sentiment_change'] > 0 else "↓",
                "mention_count": int(row['recent_mentions']),
                "change_magnitude": abs(row['sentiment_change'])
            }
            for i, (_, row) in enumerate(ranked_disasters.iterrows())
        ]

    except Exception as e:
        print(f"Error calculating top disasters: {str(e)}")
        return []

sidebar = create_top4_sidebar(calculate_top_disasters(df))

app.layout = html.Div(
    style={'backgroundColor': '#f8f9fa'},  # Light background for the page
    children=[
        html.Div(
            className="container",  # Bootstrap container for proper width control
            children=[
                html.H1(
                    children='Natural Disaster Sentiment Analysis',
                    style={'textAlign': 'left', 'marginTop': '5px'}
                ),
                # Flexbox row for sidebar and graph
                html.Div(
                    style={
                        'display': 'flex',  # Flexbox for horizontal alignment
                        'flexDirection': 'row',  # Row direction
                        'justifyContent': 'space-between',  # Space between items
                        'alignItems': 'flex-start',  # Align items at the top
                        'marginTop': '20px',
                    },
                    children=[
                        # Map on the left
                        html.Div(
                            style={
                                'width': '70%',  # Graph width (adjust as needed)
                                'marginLeft': '20px',  # Space between sidebar and graph
                            },
                            children=[
                                dcc.Graph(
                                    id='disaster-map',
                                    style={
                                        'height': '500px',
                                        'width': '100%',
                                        'borderRadius': '10px',
                                        'boxShadow': '0 4px 8px rgba(0,0,0,0.1)',
                                        'margin-bottom': '30px',
                                    },
                                    config={'displayModeBar': True},
                                ),

                            ]
                        ),
                        html.Div(
                            id='sidebar',
                            style={
                                'width': '25%',  # Sidebar width (adjust as needed)
                                'backgroundColor': '#ffffff',
                                'padding': '15px',
                                'borderRadius': '10px',
                                'boxShadow': '0 4px 8px rgba(0,0,0,0.1)',
                            },
                            children=create_top4_sidebar(calculate_top_disasters(df)),  # Replace with your sidebar content
                        ),

                    ]
                ),

                print("Final top disasters data:", calculate_top_disasters(df)),
                html.H1(
                    children='Top 4 Disasters Happening Now',
                    style={'textAlign': 'left', 'marginTop': '30px', 'marginBottom': '15px'}
                ),

                dbc.Tabs(
                    [
                        dbc.Tab(
                            id= f"tab-{disaster['rank']}", # individual tab id
                            label=f"#{disaster['rank']} {disaster['location']} {disaster['disaster']} {disaster['trend']}",
                            tab_id = f"tab-{disaster['rank']}" ,
                            tab_style={"margin": "5px", "padding": "10px"},
                            active_label_style={
                                "color": "white",
                                "backgroundColor": "#007BFF",
                                "fontWeight": "bold"
                            },
                            children=[
                                html.Div(
                                    style={'padding': '20px'},
                                    children=[
                                        html.H3(f"{disaster['location']} {disaster['disaster']}"),
                                        html.P([
                                            html.Strong("Current Severity: "),
                                            f"{disaster['severity']}",
                                            html.Br(),
                                            html.Strong("Current Sentiment: "),
                                            f"{disaster['current_sentiment']}",
                                            html.Br(),
                                            html.Strong("Trend: "),
                                            html.Span(
                                                disaster['trend'],
                                                style={
                                                    'color': 'green' if disaster['trend'] == '↑' else 'red',
                                                    'fontWeight': 'bold'
                                                }
                                            )
                                        ]),
                                        dcc.Graph(
                                            figure=create_sentiment_graph(
                                                df[(df['location'] == disaster['location']) &
                                                   (df['category'] == disaster['disaster'])]
                                            ),
                                            style={'height': '300px'}
                                        )
                                    ]
                                )
                            ]
                        )
                        for disaster in calculate_top_disasters(df)  # Make sure this returns data
                    ],
                    style={'marginBottom': '30px'}
                ),
                html.H1(
                    children='Sentiment Over Time',
                    style={'textAlign': 'left', 'marginTop': '5px'}
                ),

                html.Div(
                    dbc.Tabs([
                        dbc.Tab(
                            dcc.Graph(
                                id=f'graph-{category}',
                                figure = create_sentiment_graph(
                                    df[df['category'] == category]
                                ),
                                style={'height': '300px'}

                            ),
                            label = category
                        )
                        for category in valid_categories

                    ])
                )
            ]
        )      
    ]
)


#@app.callback(
#    dash.dependencies.Output("navbar-collapse", "is_open"),
#    dash.dependencies.Input("navbar-toggler", "n_clicks"),
#    dash.dependencies.State("navbar-collapse", "is_open"),
#)
#@app.callback(
#    dash.dependencies.Output("disaster-tabs", "active_tab"),
#    [dash.dependencies.Input(f"sidebar-link-{i}", "n_clicks") for i in range(1, 5)],
#    prevent_initial_call=True
#)
def switch_tab(*args):
    ctx = dash.callback_context
    if not ctx.triggered:
        return dash.no_update

    # Extract the rank from the clicked link
    clicked_id = ctx.triggered[0]['prop_id'].split('.')[0]
    rank = int(clicked_id.split('-')[-1])

    # Return the corresponding tab_id
    return f"tab-{rank}"


def toggle_navbar_collapse(n_clicks, is_open):
    if n_clicks:
        return not is_open
    return is_open

disaster_map_figure = generate_geo_map(posts_generalized_cleaned)

# Update the layout with the generated figure
app.layout['disaster-map'].figure = disaster_map_figure
# Run the app
#if __name__ == '__main__':
#    app.run(debug=False, host='127.0.0.1', port=8050)
