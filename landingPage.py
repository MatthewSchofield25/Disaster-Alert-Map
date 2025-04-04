import dash
import dash_bootstrap_components.themes
import dash_bootstrap_components as dbc
from dash import dcc, html
import plotly.graph_objs as go
import pandas as pd
import pyodbc
import os
from sklearn.neighbors import NearestNeighbors
import numpy as np

#Initialize Database Connection
driver = '{ODBC Driver 18 for SQL Server}'
server = os.getenv("DATABASE_SERVER")
database = os.getenv("DATABASE_ONENAME")
username = os.getenv("DATABASE_USERNAME")
password = os.getenv("DATABASE_PASSWORD")

# Initialize the Dash app
app = dash.Dash(__name__, external_stylesheets=[dash_bootstrap_components.themes.BOOTSTRAP])

# CODE TO USE FOR LATER ONCE DATABASE IS SET UP

# try:
#     db = pyodbc.connect(f'DRIVER={driver};SERVER={server};DATABASE={database};UID={username};PWD={password}')
# except Exception as e:
#     print(f"Error connecting to SQL Server: {e}")
# print("SQL Server Connection Successful")


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


# Mapbox token
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
        paper_bgcolor="#171b26",
        clickmode="event+select",
        hovermode="closest",
        showlegend=False,
        mapbox=go.layout.Mapbox(
            accesstoken=mapbox_access_token,
            bearing=10,
            center=go.layout.mapbox.Center(
                lat=geo_data.LATITUDE.mean(), lon=geo_data.LONGITUDE.mean()
            ),
            pitch=5,
            zoom=5,
            style="mapbox://styles/plotlymapbox/cjvppq1jl1ips1co3j12b9hex",
        ),
    )

    return {"data": hospitals, "layout": layout}

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

# Define the layout
app.layout = html.Div([
    navbar,
    html.H1(children='Natural Disaster Sentiment Analysis', style={'textAlign': 'left'}),
    html.H2(children='Current Events', style={'textAlign': 'center'}),
    dcc.Graph(id='hospital-map'),
])

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
    app.run(debug=True, host='127.0.0.1', port = 8050)