import dash
import dash_bootstrap_components.themes
import dash_bootstrap_components as dbc
from dash import dcc, html
import plotly.graph_objs as go
import pandas as pd
import os

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
        paper_bgcolor="#171b26",
        clickmode="event+select",
        hovermode="closest",
        showlegend=False,
        mapbox=go.layout.Mapbox(
            accesstoken=mapbox_access_token,
            bearing=10,
            center=go.layout.mapbox.Center(
                lat=filtered_data.lat.mean(), lon=filtered_data.lon.mean()
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
    app.run_server(debug=True)