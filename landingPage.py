import dash
import dash_bootstrap_components.themes
import dash_bootstrap_components as dbc
from click import style
from dash import dcc, html
import plotly.graph_objs as go
import pandas as pd
import os
import numpy as np

# Initialize the Dash app
app = dash.Dash(__name__, external_stylesheets=[dash_bootstrap_components.themes.BOOTSTRAP])

# Load the datasets
df = pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/master/gapminder_unfiltered.csv')

# State map and list
state_map = {
    "AK": "Alaska", "AL": "Alabama", "AR": "Arkansas", "AZ": "Arizona", "CA": "California",
    "CO": "Colorado", "CT": "Connecticut", "DC": "District of Columbia", "DE": "Delaware",
    "FL": "Florida", "GA": "Georgia", "HI": "Hawaii", "IA": "Iowa", "ID": "Idaho", "IL": "Illinois",
    "IN": "Indiana", "KS": "Kansas", "KY": "Kentucky", "LA": "Louisiana", "MA": "Massachusetts",
    "MD": "Maryland", "ME": "Maine", "MI": "Michigan", "MN": "Minnesota", "MO": "Missouri",
    "MS": "Mississippi", "MT": "Montana", "NC": "North Carolina", "ND": "North Dakota",
    "NE": "Nebraska", "NH": "New Hampshire", "NJ": "New Jersey", "NM": "New Mexico", "NV": "Nevada",
    "NY": "New York", "OH": "Ohio", "OK": "Oklahoma", "OR": "Oregon", "PA": "Pennsylvania",
    "RI": "Rhode Island", "SC": "South Carolina", "SD": "South Dakota", "TN": "Tennessee",
    "TX": "Texas", "UT": "Utah", "VA": "Virginia", "VT": "Vermont", "WA": "Washington",
    "WI": "Wisconsin", "WV": "West Virginia", "WY": "Wyoming",
}

valid_categories = {'Flood', 'Drought', 'Landslide', 'Volcano', 'Blizzard',
                        'Earthquake', 'Tsunami', 'Wildfire', 'Hurricane', 'Tornado', 'Other'}

# Create mock sentiment data if 'category' column doesn't exist
if 'category' not in df.columns:
    dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
    categories = list(valid_categories)

    mock_data = {
        'date': np.random.choice(dates, size=1000),
        'sentiment': np.random.uniform(-1, 1, size=1000),
        'category': np.random.choice(categories, size=1000)
    }
    df = pd.DataFrame(mock_data)

state_list = list(state_map.keys())

# Load hospital data
data_dict = {}
for state in state_list:
    csv_path = f"data/processed/df_{state}_lat_lon.csv"
    if os.path.exists(csv_path):  # Check if the file exists
        state_data = pd.read_csv(csv_path)
        data_dict[state] = state_data
    else:
        print(f"Warning: File not found for state {state} at {csv_path}")

def get_lat_lon_add(df, name):
    return [
        df.groupby(["Provider Name"]).get_group((name,))["lat"].tolist()[0],
        df.groupby(["Provider Name"]).get_group((name,))["lon"].tolist()[0],
        df.groupby(["Provider Name"])
        .get_group((name,))["Provider Street Address"]
        .tolist()[0],
    ]

def generate_aggregation(df, metric):
    aggregation = {
        metric[0]: ["min", "mean", "max"],
        metric[1]: ["min", "mean", "max"],
        metric[2]: ["min", "mean", "max"],
    }
    grouped = (
        df.groupby(["Hospital Referral Region (HRR) Description", "Provider Name"])
        .agg(aggregation)
        .reset_index()
    )

    grouped["lat"] = grouped["lon"] = grouped["Provider Street Address"] = grouped[
        "Provider Name"
    ]
    grouped["lat"] = grouped["lat"].apply(lambda x: get_lat_lon_add(df, x)[0])
    grouped["lon"] = grouped["lon"].apply(lambda x: get_lat_lon_add(df, x)[1])
    grouped["Provider Street Address"] = grouped["Provider Street Address"].apply(
        lambda x: get_lat_lon_add(df, x)[2]
    )

    return grouped

# Debug: Print loaded states
print("Loaded data for states:", data_dict.keys())

# Cost Metric
cost_metric = [
    "Average Covered Charges",
    "Average Total Payments",
    "Average Medicare Payments",
]

# Mapbox token
mapbox_access_token = "pk.eyJ1IjoicGxvdGx5bWFwYm94IiwiYSI6ImNrOWJqb2F4djBnMjEzbG50amg0dnJieG4ifQ.Zme1-Uzoi75IaFbieBDl3A"

# Function to generate the hospital map
def generate_geo_map(geo_data, selected_metric, region_select, procedure_select):
    if geo_data.empty:
        return {"data": [], "layout": {}}

    filtered_data = geo_data[
        geo_data["Hospital Referral Region (HRR) Description"].isin(region_select)
    ]

    # Debug: Print columns to verify structure
    print("Filtered Data Columns:", filtered_data.columns)

    colors = ["#21c7ef", "#76f2ff", "#ff6969", "#ff1717"]
    hospitals = []

    lat = filtered_data["lat"].tolist()
    lon = filtered_data["lon"].tolist()

    # Access the mean value correctly
    if isinstance(filtered_data.columns, pd.MultiIndex):
        # If the DataFrame has hierarchical columns
        average_covered_charges_mean = filtered_data[(selected_metric, "mean")].tolist()
    else:
        # If the DataFrame has flat columns
        average_covered_charges_mean = filtered_data[selected_metric].tolist()

    regions = filtered_data["Hospital Referral Region (HRR) Description"].tolist()
    provider_name = filtered_data["Provider Name"].tolist()

    # Cost metric mapping from aggregated data
    cost_metric_data = {
        "min": min(average_covered_charges_mean),
        "max": max(average_covered_charges_mean),
        "mid": (min(average_covered_charges_mean) + max(average_covered_charges_mean)) / 2,
        "low_mid": (min(average_covered_charges_mean) + max(average_covered_charges_mean)) / 4,
        "high_mid": 3 * (min(average_covered_charges_mean) + max(average_covered_charges_mean)) / 4,
    }

    for i in range(len(lat)):
        val = average_covered_charges_mean[i]
        region = regions[i]
        provider = provider_name[i]

        if val <= cost_metric_data["low_mid"]:
            color = colors[0]
        elif cost_metric_data["low_mid"] < val <= cost_metric_data["mid"]:
            color = colors[1]
        elif cost_metric_data["mid"] < val <= cost_metric_data["high_mid"]:
            color = colors[2]
        else:
            color = colors[3]

        selected_index = []
        if provider in procedure_select["hospital"]:
            selected_index = [0]

        hospital = go.Scattermap(
            lat=[lat[i]],
            lon=[lon[i]],
            mode="markers",
            marker=dict(
                color=color,
                showscale=True,
                colorscale=[
                    [0, "#21c7ef"],
                    [0.33, "#76f2ff"],
                    [0.66, "#ff6969"],
                    [1, "#ff1717"],
                ],
                cmin=cost_metric_data["min"],
                cmax=cost_metric_data["max"],
                size=10 * (1 + (val + cost_metric_data["min"]) / cost_metric_data["mid"]),
                colorbar=dict(
                    x=0.9,
                    len=0.7,
                    title=dict(
                        text="Happening now",
                        font={"color": "#737a8d", "family": "Open Sans"},
                        side="top",
                    ),
                    #tickmode="array",
                   # tickvals=[cost_metric_data["min"], cost_metric_data["max"]],
                   # ticktext=[
                   #     "${:,.2f}".format(cost_metric_data["min"]),
                    #    "${:,.2f}".format(cost_metric_data["max"]),
                   # ],
                    ticks="outside",
                    thickness=15,
                    tickfont={"family": "Open Sans", "color": "#737a8d"},
                ),
            ),
            opacity=0.8,
            selectedpoints=selected_index,
            selected=dict(marker={"color": "#ffff00"}),
            customdata=[(provider, region)],
            hoverinfo="text",
            text=provider + "<br>" + region + "<br>Average Procedure Cost:" + " ${:,.2f}".format(val),
        )
        hospitals.append(hospital)

    layout = go.Layout(
        margin=dict(l=10, r=10, t=20, b=10, pad=5),
        plot_bgcolor="#171b26",
        paper_bgcolor="rgba(0,0,0,0)",
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

        )
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


def create_sentiment_graph(df, category):
    category_df = df[df['category'] == category]

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=category_df['date'],
        y=category_df['sentiment'],
        mode='lines+markers',
        name=category
    ))

    fig.update_layout(
        title=f'Sentiment Over Time: {category}',
        yaxis=dict(range=[-1, 1], title='Sentiment Score'),
        xaxis=dict(title='Date'),
        hovermode='x unified',
        height=300,
        margin=dict(l=40, r=40, t=60, b=40)
    )

    # Add neutral reference line
    fig.add_hline(y=0, line_dash='dash', line_color='gray')

    return fig


    # Define the layout
'''
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
                html.Div(
                    style={
                        #controls map box size
                        'width': '100%',
                        'marginLeft': '0',
                        'marginRight': 'auto',
                    },
                    children=[
                        dcc.Graph(
                            id='hospital-map',
                            style = {
                                'height': '500px',
                                #controls width
                                'width': '75%',
                                'borderRadius': '10px',
                                'boxShadow': '0 4px 8px rgba(0,0,0,0.1)',
                            },
                            config={'displayModeBar': True},
                        )
                    ]
                )
            ]
         ),
        sidebar,
    ]
)
'''
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
                html.H1(
                    children='Sentiment Over Time',
                    style={'textAlign': 'left', 'marginTop': '5px'}
                ),

                html.Div(
                    dbc.Tabs([
                        dbc.Tab(
                            dcc.Graph(
                                id=f'graph-{category}',
                                figure = create_sentiment_graph(df, category),

                            ),
                            label = category
                        )
                        for category in sorted(valid_categories)

                    ])
                ),

                '''
                html.Div(
                    dcc.Graph(
                        id = 'sentiment timeline',
                        figure = generate_sentiment_timeline(),
                        style={
                            'margin-top': '30px',
                            'margin-bottom': '30px',
                            'height': '500px',
                            'width': '100%',
                            'borderRadius': '10px',
                            'boxShadow': '0 4px 8px rgba(0,0,0,0.1)',
                            'backgroundColor': 'white',
                        }
                    )
                )
                '''
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


# Generate the hospital map directly
if not data_dict:
    hospital_map_figure = {"data": [], "layout": {}}
else:
    # Use the first state's data for demonstration
    state = state_list[0]
    if state not in data_dict:
        hospital_map_figure = {"data": [], "layout": {}}
    else:
        state_data = data_dict[state]
        aggregated_data = generate_aggregation(state_data, cost_metric)
        regions = state_data["Hospital Referral Region (HRR) Description"].unique()
        hospital_map_figure = generate_geo_map(aggregated_data, cost_metric[0], regions, {"hospital": []})

# Update the layout with the generated figure
app.layout['hospital-map'].figure = hospital_map_figure

# Run the app
if __name__ == '__main__':
    app.run(debug=True)