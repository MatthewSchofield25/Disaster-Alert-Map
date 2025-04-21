import dash
import dash_bootstrap_components.themes
import dash_bootstrap_components as dbc
from dash import dcc, html, register_page
import os

# Initialize the Dash app
app = dash.Dash(__name__, external_stylesheets=[dash_bootstrap_components.themes.BOOTSTRAP])

register_page(
    __name__,
    "/AboutUsPage",
    title="About Us",
    name="About Us"
)

postofnote = html.Div(
    className="bluesky-embed",
    **{
        "data-bluesky-uri": "at://did:plc:k5nskatzhyxersjilvtnz4lh/app.bsky.feed.post/3ln445ahyoc2f",
        "data-bluesky-cid": "bafyreicrmal7jdvkdj7kxxy46epzlwxzp3tfxplpjr7aw5gmnh4bwtrr7a",
        "data-bluesky-embed-color-mode": "system"
    },
    children=[
        html.P(
            "Exclusive: Immigrants falsely labeled dead by the Social Security Administration are showing up at field offices with documents proving they are alive, leading staff to reinstate nearly three dozen people over the past week, according to records obtained by The Post.",
            lang="en"
        ),
        html.Br(),
        html.Br(),
        html.A(
            "[image or embed]",
            href="https://bsky.app/profile/did:plc:k5nskatzhyxersjilvtnz4lh/post/3ln445ahyoc2f?ref_src=embed"
        ),
        html.P(
            [
                "— The Washington Post (",
                html.A(
                    "@washingtonpost.com",
                    href="https://bsky.app/profile/did:plc:k5nskatzhyxersjilvtnz4lh?ref_src=embed"
                ),
                ") ",
                html.A(
                    "April 18, 2025 at 12:05 PM",
                    href="https://bsky.app/profile/did:plc:k5nskatzhyxersjilvtnz4lh/post/3ln445ahyoc2f?ref_src=embed"
                )
            ]
        )
    ]
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

info_blurb = dbc.Container(
    dbc.Row([
        dbc.Col([
            html.H2("About Us", className="text-center mb-4"),
            html.H4(
                "The team : "
            ),
            dbc.Card(
                dbc.CardBody([
                    html.H4("Team Leader"),
                    html.P("Matthew Schofield")
                ])
            ),
            dbc.Card(
                dbc.CardBody([
                    html.H4("Scrum Master"),
                    html.P("Liz Cadungog")
                ])
            ),
            dbc.Card(
                dbc.CardBody([
                    html.H4("Data Scientists"),
                    html.P("Vanessa Alvarez, Kenny Rodriguez")
                ])
            ),
            dbc.Card(
                dbc.CardBody([
                    html.H4("Front-End Developers"),
                    html.P("Gail Hernandez, Adam Mondragon")
                ])
            ),
            html.P(
                "We are a group of students at the University of Texas at Dallas who have worked to create a dashboard to show the public sentiment about current and recent natural disasters by using data collected from Bluesky posts. We want our dashboard to be an accessible and reliable source of information regarding the sentiment of these events from the perspectives of the people, without the biases and filters of traditional news sources. "
            )
        ])
    ])
)

layout = html.Div(
        children=[
            #postofnote,
            info_blurb,
                dbc.Container([
                    # Heading paragraph
                    dbc.Row([
                        dbc.Col([
                            html.H2("External Resources", className="text-center mb-4"),
                            html.P(
                                "The American Red Cross provides many resources for information "
                                "about natural disasters and the steps you can take to prepare for them. "
                                "Please see the links below to inform yourself and your loved ones in the "
                                "event that you need to prepare for any of these disasters.",
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


#@app.callback(
#    dash.dependencies.Output("navbar-collapse", "is_open"),
#    dash.dependencies.Input("navbar-toggler", "n_clicks"),
#    dash.dependencies.State("navbar-collapse", "is_open"),
#)

def toggle_navbar_collapse(n_clicks, is_open):
    if n_clicks:
        return not is_open
    return is_open

# Run the app
#if __name__ == '__main__':
#    app.run(debug=True, host='127.0.0.1', port = 8050)
