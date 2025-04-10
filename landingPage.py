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

# Load the datasets for
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

# Mapbox token, this is a public key, there are no
# issues posting this on the internet
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