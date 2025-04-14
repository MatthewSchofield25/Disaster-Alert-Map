import dash
import dash_bootstrap_components.themes
import dash_bootstrap_components as dbc
from dash import dcc, html
import os

from nltk import data

#Initialize Database Connection
driver = '{ODBC Driver 18 for SQL Server}'
server = os.getenv("DATABASE_SERVER")
database = os.getenv("DATABASE_NAME")
username = os.getenv("DATABASE_USERNAME")
password = os.getenv("DATABASE_PASSWORD")

# Initialize the Dash app
app = dash.Dash(__name__, external_stylesheets=[dash_bootstrap_components.themes.BOOTSTRAP])

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
                                    dbc.NavItem(dbc.NavLink("Home", href="#", active=False)),
                                    dbc.NavItem(dbc.NavLink("About", href="#", active=True)),
                                    #dbc.DropdownMenu(
                                   #     [
                                   #         dbc.DropdownMenuItem("Action", href="#"),
                                   #         dbc.DropdownMenuItem("Another action", href="#"),
                                   #         dbc.DropdownMenuItem("Something else here", href="#"),
                                    #        dbc.DropdownMenuItem(divider=True),
                                    #        dbc.DropdownMenuItem("Separated link", href="#"),
                                    #    ],
                                  #      label="Dropdown",
                                   #     nav=True,
                                   # ),
                                ],
                                className="me-auto",  # Align links to the left
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

container_data = [
    {"label" : "Flood", "text" : "Preparation Checklist", "url" : "https://www.redcross.org/content/dam/redcross/atg/PDF_s/Preparedness___Disaster_Recovery/Disaster_Preparedness/Flood/Flood.pdf?srsltid=AfmBOooYb5XqIU4q2rpe4A2Hp28Bzj9dahoGHyrIcco_GJSxpU-J7rCo"},
    {"label" : "Drought", "text" : "Preparation", "url" : "https://www.redcross.org/get-help/how-to-prepare-for-emergencies/types-of-emergencies/drought.html#:~:text=Take%20shorter%20showers.,excess%20water%20for%20watering%20plants."},
    {"label" : "Landslide", "text" : "Preparation Checklist", "url" : "https://www.redcross.org/content/dam/redcross/atg/PDF_s/Preparedness___Disaster_Recovery/Disaster_Preparedness/Landslide/Landslide.pdf?srsltid=AfmBOop3gTFdYz2qGqZs2h9R_kA05dedcyzQv45cRGthIBBQUNvBCqUQ"},
    {"label" : "Volcano", "text" : "Preparation Checklist", "url" : "https://www.redcross.org/content/dam/redcross/get-help/pdfs/volcano/EN-Volcano-Preparedness-Checklist.pdf?srsltid=AfmBOopXz8YX7B7U2B--Kz-4sWxsCVmX7WMja4Uty4HRAmuuHoW4huRs"},
    {"label" : "Blizzard", "text" : "Preparation Checklist", "url" : "https://www.redcross.org/content/dam/redcross/get-help/pdfs/winter-storm/EN_Winter-Storm-Preparedness-Checklist.pdf?srsltid=AfmBOor7GyFezrLptgyv0dhYblX37Q4gFgAgAEocwDTFnIv-qlNtm8OV"},
    {"label" : "Earthquake", "text" : "Preparation Checklist", "url" : "https://www.redcross.org/content/dam/redcross/atg/PDF_s/Preparedness___Disaster_Recovery/Disaster_Preparedness/Earthquake/Earthquake.pdf?srsltid=AfmBOorWLUzIWi0f9PTdupBli10qhxqndYXV8ja3pjrktRW9LT36SP2Q"},
    {"label" : "Tsunami", "text" : "Preparation Checklist", "url" : "https://www.redcross.org/content/dam/redcross/atg/PDF_s/Preparedness___Disaster_Recovery/Disaster_Preparedness/Tsunami/Tsunami.pdf?srsltid=AfmBOooKXPhZz3pWkWDFr-fd0Yh09y8_NSx7Jlk57pfGAIEc-1c5_gNG"},
    {"label" : "Wildfire", "text" : "Preparation Checklist", "url" : "https://www.redcross.org/content/dam/redcross/get-help/pdfs/wildfire/EN_Wildfire-Safety-Checklist.pdf?srsltid=AfmBOooxaNVvAw1i67Te2XtaaFySkWh_zG5Jymc0-6TSk98HCWSVUJSS"},
    {"label" : "Hurricane", "text" : "Preparation Checklist", "url" : "https://www.redcross.org/content/dam/redcross/get-help/pdfs/hurricane/EN_Hurricane-Safety-Checklist.pdf?srsltid=AfmBOooxmu63WM4R56wSqLWsMhAy_R1khKx99umgvAz3u1aWiTwMytK9"},
    {"label" : "Tornado", "text" : "Preparation Checklist", "url" : "https://www.redcross.org/content/dam/redcross/get-help/pdfs/tornado/EN_Tornado-Safety-Checklist.pdf?srsltid=AfmBOorBQ3VJoHMyjfyCXQgXFAxITmey36YT6TFW0r4w6POiBPjM8tX7"}
]

navbar = dbc.Navbar(
    dbc.Container(
        [
            dbc.NavbarBrand("CS Project", href="#"),
            dbc.NavbarToggler(id="navbar-toggler"),

            dbc.Collapse(
                dbc.Row(
                    [
                        dbc.Col(
                            dbc.Nav(
                                [
                                    dbc.NavItem(dbc.NavLink("Home", href="#")),
                                    dbc.NavItem(dbc.NavLink("About", href="#", active=True)),
                                ],
                                className="me-auto",
                            ),
                            width="auto",
                        ),
                    ],
                    justify="between",
                    className="g-0 w-100",
                ),
                id="navbar-collapse",
                is_open=False,
                navbar=True,
            ),
        ],
        fluid=True,
    ),
    color="primary",
    dark=True,
)

info_blurb = dbc.Container(
    dbc.Row([
        dbc.Col([
            html.H2("About Us", className="text-center mb-4"),
            html.P(
                "The team : Matthew Schofield, Liz Cadungog, Kenny Rodriguez, Vanessa Alvarez, Adam Mondragon, Gail Hernandez"
            ),
            html.P(
                "We are a group of students at the University of Texas at Dallas (there's gonna be more later)"
            )
        ])
    ])
)

# Create the app layout
app.layout = html.Div(
    children=[
        navbar,
        info_blurb,
            dbc.Container([
                # Heading paragraph
                dbc.Row([
                    dbc.Col([
                        html.H2("External Resources", className="text-center mb-4"),
                        html.P(
                            "The American Red Cross provides many resources for information"
                            "about natural disasters and the steps you can take to prepare for them."
                            "Please see the links below to inform yourself and your loved ones in the"
                            "case that you need to prepare for any of these disasters.",
                            className="lead text-center mb-5"
                        )
                    ], width=12)
                ]),

                # Container for the information boxes
                dbc.Row([
                    dbc.Col([
                        # Generate the 10 containers dynamically
                        *[
                            dbc.Card(
                                dbc.CardBody([
                                    html.H4(data["label"], className="card-title"),
                                    html.P(
                                        html.A(
                                            data["text"],
                                            href=data["url"],
                                            target="_blank",
                                            className="card-link"
                                        ),
                                        className="card-text"
                                    )
                                ]),
                                className="mb-4 shadow-sm"
                            )
                            for data in container_data
                        ]
                    ], width=12)
                ])
            ], fluid=True)
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

# Run the app
if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port = 8050)