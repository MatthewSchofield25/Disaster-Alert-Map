import dash
import dash_bootstrap_components.themes
import dash_bootstrap_components as dbc
from click import style
from dash import dcc, html
import plotly.graph_objs as go
import pandas as pd
import pyodbc
import os
<<<<<<< HEAD
import numpy as np
=======
from sklearn.neighbors import NearestNeighbors
import numpy as np

#Initialize Database Connection
driver = '{ODBC Driver 18 for SQL Server}'
server = os.getenv("DATABASE_SERVER")
database = os.getenv("DATABASE_ONENAME")
username = os.getenv("DATABASE_USERNAME")
password = os.getenv("DATABASE_PASSWORD")
>>>>>>> a76d33d87b081b1fab94217d736839c02fc11ec1

# Initialize the Dash app
app = dash.Dash(__name__, external_stylesheets=[dash_bootstrap_components.themes.BOOTSTRAP])

<<<<<<< HEAD
# Load the datasets for
df = pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/master/gapminder_unfiltered.csv')
=======
# CODE TO USE FOR LATER ONCE DATABASE IS SET UP
>>>>>>> a76d33d87b081b1fab94217d736839c02fc11ec1

# try:
#     db = pyodbc.connect(f'DRIVER={driver};SERVER={server};DATABASE={database};UID={username};PWD={password}')
# except Exception as e:
#     print(f"Error connecting to SQL Server: {e}")
# print("SQL Server Connection Successful")

<<<<<<< HEAD
valid_categories = {'Flood', 'Drought', 'Landslide', 'Volcano', 'Blizzard',
                        'Earthquake', 'Tsunami', 'Wildfire', 'Hurricane', 'Tornado', 'Other'}

try:
    # Load your CSV file (adjust path if needed)
    df = pd.read_csv('sentiment_data.csv')
    # Ensure date is in datetime format
    df['date'] = pd.to_datetime(df['date'])
except FileNotFoundError:
    print("Error: sentiment_data.csv not found. Using mock data instead.")
    # Fallback to mock data if file not found
    dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
    categories = list(valid_categories)
    mock_data = {
        'date': np.random.choice(dates, size=1000),
        'sentiment': np.random.uniform(-1, 1, size=1000),
        'category': np.random.choice(categories, size=1000)
    }
    df = pd.DataFrame(mock_data)

state_list = list(state_map.keys())
=======
>>>>>>> a76d33d87b081b1fab94217d736839c02fc11ec1

# # LATER DEVELOPMENT: UTILIZE THE TIMEPOSTED TO ONLY GET THE LAST 31 DAYS OF DATA
# cursor = db.cursor()
# cursor.execute("SELECT sentiment_score, Predicted_Disaster_Type, location FROM LSTM_Posts")
# results = cursor.fetchall()
# columns = [column[0] for column in cursor.description]
# df = pd.DataFrame.from_records(results, columns=columns)

# CURRENTLY READING IN THROUGH CSV FILE MADE BY VANESSA
# LATER DEVELOPMENT: ONCE DATABASE IS SET UP, USE THE CODE ABOVE TO GET THE DATA FROM THE DATABASE
df = pd.read_csv('Bluesky_Disaster_Predictions_With_Relevance.csv', usecols=['sentiment_score', 'Predicted_Disaster_Type', 'Location'])

# Aggregate sentiment and count posts per location
# Can see problem occuring here where if the location is blank those will all be grouped together
# and will not be able to be plotted on the map. Need to filter out blank locations before this step.
# Or we can just not plot them on the map.
# location_stats = df.groupby('Predicted_Disaster_Type').agg(
#     avg_sentiment=('sentiment_score', 'mean'),
#     post_count=('Predicted_Disaster_Type', 'count')
# ).reset_index()

# print(location_stats)

# # Define emergency thresholds
# emergency_sentiment_threshold = 0  # Adjust as needed
# emergency_post_count_threshold = 0  # Adjust as needed

# # Add emergency flag
# location_stats['emergency'] = (
#     (location_stats['avg_sentiment'] <= emergency_sentiment_threshold) |
#     (location_stats['post_count'] >= emergency_post_count_threshold)
# )

grouped = df.groupby('Predicted_Disaster_Type')

def get_most_frequent_location(group):
    if group.empty:
        return pd.Series([None, None, 0], index=['location', 'avg_sentiment', 'post_count']) #Handle empty groups.

    mode_series = group['Location'].mode()
    most_frequent_location = mode_series.iloc[0] if not mode_series.empty else None #Handle empty modes.
    avg_sentiment = group['sentiment_score'].mean()
    post_count = group['Location'].count() # or group['Location'].count()
    return pd.Series([most_frequent_location, avg_sentiment, post_count], index=['Location', 'avg_sentiment', 'post_count'])

result = grouped.apply(get_most_frequent_location, include_groups=False).reset_index()
print(result)

# Load location data
location_data = pd.read_csv("data/US_cities_lat_lon.csv")

# Create a 'location' column in location_data
location_data['location'] = location_data.apply(
    lambda row: row['CITY'] if pd.notna(row['CITY']) else (row['STATE_NAME'] if pd.notna(row['STATE_NAME']) else row['COUNTY']),
    axis=1
)

result['Location'] = result['Location'].astype(str)
location_data['location'] = location_data['location'].astype(str)


# Merge with location data
merged_data = pd.merge(result, location_data, left_on='Location', right_on='location')


# Mapbox token, this is a public key, there are no
# issues posting this on the internet
mapbox_access_token = "pk.eyJ1IjoicGxvdGx5bWFwYm94IiwiYSI6ImNrOWJqb2F4djBnMjEzbG50amg0dnJieG4ifQ.Zme1-Uzoi75IaFbieBDl3A"


# Modified generate_geo_map function
def generate_geo_map(geo_data):
    if geo_data.empty:
        return {"data": [], "layout": {}}

    lat = geo_data["LATITUDE"].tolist()
    lon = geo_data["LONGITUDE"].tolist()
    avg_sentiment = geo_data["avg_sentiment"].tolist()
    post_count = geo_data["post_count"].tolist()
    location = geo_data['Location'].tolist()
    disaster_type = geo_data['Predicted_Disaster_Type'].tolist()
    
    hospitals = []

    for i in range(len(lat)):
        size = 10 * (1 - avg_sentiment[i]) + post_count[i]
        color = "#21c7ef"

        hospital = go.Scattermap(
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
        hospitals.append(hospital)

    layout = go.Layout(
        margin=dict(l=10, r=10, t=20, b=10, pad=5),
        geo = dict(
            scope = 'usa',
            landcolor = 'rgb(217, 217, 217)',
        ),
        plot_bgcolor="#171b26",
        paper_bgcolor="rgba(0,0,0,0)",
        clickmode="event+select",
        hovermode="closest",
        showlegend=False,
        mapbox=go.layout.Mapbox(
            accesstoken=mapbox_access_token,
            bearing=0,
            center=go.layout.mapbox.Center(
<<<<<<< HEAD
                lat = 37.0902,
                lon = -95.7129
=======
                lat=geo_data.LATITUDE.mean(), lon=geo_data.LONGITUDE.mean()
>>>>>>> a76d33d87b081b1fab94217d736839c02fc11ec1
            ),
            pitch=0,
            zoom=3,
            style="mapbox://styles/plotlymapbox/cjvppq1jl1ips1co3j12b9hex",
            bounds=dict(west=-125, east=-66, south=24, north=50),  # Continental US

        )
    )

    return {"data": hospitals, "layout": layout}

<<<<<<< HEAD

def create_sentiment_graph(filtered_df):
    """
    Creates a sentiment trend graph for a specific location and disaster type

    Args:
        filtered_df: DataFrame containing columns 'date', 'sentiment_score',
                    'location', and 'disaster_type'

    Returns:
        plotly.graph_objects.Figure: The generated sentiment graph
    """
    fig = go.Figure()

    #fig.add_trace adds a new data series to the figure
    fig.add_trace(
        go.Scatter( #uses scatter plot but will act as a line graph
        x=filtered_df['date'],
        y=filtered_df['sentiment_score'],
        mode='lines+markers', #creates lines between points and at each point add a marker
        line=dict(color='#1f77b4', width=2), # blue line with a width of 2 pixels
        marker=dict(size=6), #configures the marker portion of the mode line, makes the diameter 6 pixels
        name='Sentiment',
        hovertemplate=(
            '<b>Date</b>: %{x|%Y-%m-%d}<br>' #bolds Date and formats x into YYYY-MM-DD, <br> = line break
            '<b>Sentiment</b>: %{y:.2f}<extra></extra>' # bold sentiment, formats y values w percision of 2
        ) # hovering over a marker displays data, <extra></extra> gets rid displaying the name component
    ))

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

    # Get location and disaster type for title
    location = filtered_df['location'].iloc[0] #iloc = integer location, grabs first unique value
    disaster_type = filtered_df['disaster_type'].iloc[0]

    fig.update_layout(
        title=f'{location} {disaster_type} Sentiment Trend',
        xaxis_title='Date',
        yaxis_title='Sentiment Score',
        yaxis=dict(range=[-1, 1]),
        hovermode='x unified',
        plot_bgcolor='white',
        margin=dict(l=40, r=40, t=60, b=40),
        height=300,
        xaxis=dict(
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


def calculate_top_disasters(df):
    """
    Calculate top disasters based on sentiment change between:
    - 31-day average sentiment (days 1-31 ago)
    - 10-day average sentiment (most recent 10 days)

    Returns: List of dictionaries containing ranked disasters
    """
    # Convert date and filter out invalid dates
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.dropna(subset=['date'])

    if df.empty:
        print("No valid data after date cleaning")
        return []

    # Get most recent date in data
    most_recent_date = df['date'].max()

    # Define time windows - comparing recent 10 days vs previous 31 days
    recent_10_days = most_recent_date - pd.Timedelta(days=10)
    comparison_31_days = most_recent_date - pd.Timedelta(days=31)  # 31 days ending 10 days ago

    try:
        # Filter data for time periods
        recent_data = df[df['date'] > recent_10_days]
        comparison_data = df[(df['date'] > comparison_31_days) &
                             (df['date'] <= recent_10_days)]

        # Calculate average sentiment for each period
        avg_recent = recent_data.groupby(['location', 'disaster_type'])['sentiment_score'] \
            .mean().reset_index() \
            .rename(columns={'sentiment_score': 'recent_10day_avg'})

        avg_comparison = comparison_data.groupby(['location', 'disaster_type'])['sentiment_score'] \
            .mean().reset_index() \
            .rename(columns={'sentiment_score': 'previous_31day_avg'})

        # Merge the averages
        merged = avg_comparison.merge(
            avg_recent,
            on=['location', 'disaster_type'],
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
        mention_counts = recent_data.groupby(['location', 'disaster_type']).size()
        merged = merged.join(mention_counts.rename('recent_mentions'), on=['location', 'disaster_type'])

        # Filter and rank - require minimum 5 mentions in recent period
        ranked_disasters = merged[merged['recent_mentions'] >= 5] \
            .nlargest(4, 'severity_score')

        return [
            {
                "rank": i + 1,
                "location": row['location'],
                "disaster": row['disaster_type'],
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

=======
>>>>>>> a76d33d87b081b1fab94217d736839c02fc11ec1
navbar = dbc.Navbar(
    dbc.Container(
        [
            dbc.NavbarBrand("CS Project", href="#"),  # Navbar brand
            dbc.NavbarToggler(id="navbar-toggler"),  # Navbar toggler button

            dbc.Collapse(
                dbc.Row(
                    [
                        # Left-aligned navigation links
                        dbc.Col(
                            dbc.Nav(
                                [
                                    dbc.NavItem(dbc.NavLink("Home", href="#", active=True)),
                                    dbc.NavItem(dbc.NavLink("Current Events", href="#")),
                                    dbc.NavItem(dbc.NavLink("About", href="#")),
                                    dbc.DropdownMenu(
                                        [
                                            dbc.DropdownMenuItem("Action", href="#"),
                                            dbc.DropdownMenuItem("Another action", href="#"),
                                            dbc.DropdownMenuItem("Something else here", href="#"),
                                            dbc.DropdownMenuItem(divider=True),
                                            dbc.DropdownMenuItem("Separated link", href="#"),
                                        ],
                                        label="Dropdown",
                                        nav=True,
                                    ),
                                ],
                                className="me-auto",  # Align links to the left
                            ),
                            width="auto",
                        ),

                        # Right-aligned search bar
                        dbc.Col(
                            dbc.InputGroup(
                                [
                                    dbc.Input(type="search", placeholder="Search"),
                                    dbc.Button("Search", color="secondary", className="ms-2"),
                                ],
                            ),
                            width="auto",
                        ),
                    ],
                    justify="between",  # Distribute space between links and search
                    className="g-0 w-100",  # Remove gutters and take full width
                ),
                id="navbar-collapse",
                is_open=False,
                navbar=True,
            ),
        ],
        fluid=True,  # Make the container fluid
    ),
    color="primary",
    dark=True,
)

sidebar = html.Div(
    style={"padding": "20px", "maxWidth": "350px", "margin": "right"},
    children=[
        dbc.Stack(
            [
                dbc.InputGroup(
                    [
                        dbc.Input(type="search", placeholder="Search"),
                        dbc.Button("Search", color="secondary", className="ms-2"),
                    ],
                    className="mb-3",

                ),
                dbc.Card(
                    [
                        dbc.CardHeader(
                            html.A(
                                "LA Wildfire",
                                href = "#",
                                style = {
                                    "color": "inherit",
                                    "text-decoration": "none",
                                    "display": "block",
                                    "width": "200px",
                                    "height": "50px",
                                    # font and text aligment
                                    "text-align": "left",
                                    "font-size": "clamp(12px,2vw,20px)",
                                    "white-space": "nowrap",
                                    "overflow": "hidden",
                                    "text-overflow": "ellipsis",
                                    "font-weight": "bold",
                                    "padding": "0.5rem"

                                }

                            ),
                            style = {"padding":0}
                        ),
                        #dbc.CardBody("Lorem ipsum"),

                    ],
                    style={"margin-bottom": "1rem"},
                ),
                dbc.Card(
                    [
                        dbc.CardHeader(
                            html.A(
                                "Florida Hurricane",
                                href="#",
                                style={
                                    "color": "inherit",
                                    "text-decoration": "none",
                                    "display": "block",
                                    "width": "200px",
                                    "height": "50px",
                                    # font and text aligment
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
                        # dbc.CardBody("Lorem ipsum"),

                    ],
                    style={"margin-bottom": "1rem"},
                ),
                dbc.Card(
                    [
                        dbc.CardHeader(
                            html.A(
                                "North Texas Freeze",
                                href="#",
                                style={
                                    "color": "inherit",
                                    "text-decoration": "none",
                                    "display": "block",
                                    "width": "200px",
                                    "height": "50px",
                                    # font and text aligment
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
                        # dbc.CardBody("Lorem ipsum"),

                    ],
                    style={"margin-bottom": "1rem"},
                ),
                dbc.Card(
                    [
                        dbc.CardHeader(
                            html.A(
                                "South Carolina Flood",
                                href="#",
                                style={
                                    "color": "inherit",
                                    "text-decoration": "none",
                                    "display": "block",
                                    "width": "200px",
                                    "height": "50px",
                                    # font and text aligment
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
                        # dbc.CardBody("Lorem ipsum"),

                    ],
                    style={"margin-bottom": "1rem"},
                ),
            ]
        )
    ]
)


    # Define the layout
app.layout = html.Div(
    style={'backgroundColor': '#f8f9fa'},  # Light background for the page
    children=[
        navbar,
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
                                    id='hospital-map',
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
                            children=sidebar,  # Replace with your sidebar content
                        ),
                        # Graph on the right

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
                            label=f"#{disaster['rank']} {disaster['location']} {disaster['disaster']} {disaster['trend']}",
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
                                                   (df['disaster_type'] == disaster['disaster'])]
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
                                    df[df['disaster_type'] == category]
                                ),
                                style={'height': '300px'}

                            ),
                            label = category
                        )
                        for category in valid_categories

                    ])
                ),
            ]
        )
    ]
)


@app.callback(
    dash.dependencies.Output("navbar-collapse", "is_open"),
    dash.dependencies.Input("navbar-toggler", "n_clicks"),
    dash.dependencies.State("navbar-collapse", "is_open"),
)

def toggle_navbar_collapse(n_clicks, is_open):
    if n_clicks:
        return not is_open
    return is_open

hospital_map_figure = generate_geo_map(merged_data)

# Update the layout with the generated figure
app.layout['hospital-map'].figure = hospital_map_figure
# Run the app
if __name__ == '__main__':
<<<<<<< HEAD
    app.run(debug=True)
=======
    app.run(debug=True, host='127.0.0.1', port = 8050)
>>>>>>> a76d33d87b081b1fab94217d736839c02fc11ec1
