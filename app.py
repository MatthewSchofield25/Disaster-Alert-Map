#main application file
from dash import Dash, html, page_container
import dash_bootstrap_components as dbc
from dash import page_registry

# Initialize the Dash app
app = Dash(__name__, use_pages=True, external_stylesheets=[dbc.themes.BOOTSTRAP])

# Define the navbar with links to both pages
navbar = dbc.NavbarSimple(
    children=[
        dbc.NavItem(dbc.NavLink("Home", href="/")),
        dbc.NavItem(dbc.NavLink("About Us", href="/AboutUsPage")),
    ],
    brand="Natural Disaster Sentiment Analysis",
    brand_href="/",
    color="primary",
    dark=True,
)

# App layout
app.layout = html.Div([
    navbar,
    page_container
])

if __name__ == "__main__":
    app.run(debug=True)