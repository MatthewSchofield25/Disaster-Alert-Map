import dash
import dash_bootstrap_components as dbc
from dash import html

# Initialize the Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# Define the layout of the webpage
app.layout = dbc.Container([
    # Header row with title and button
    dbc.Row([
        dbc.Col(html.H1("<Location> <Disaster type>", className="text-left"), width=6),
        dbc.Col(dbc.Button("Return to Map", color="primary", className="float-end"), width=6)
    ], className="mb-4"),

    # Row with two side-by-side cards
    dbc.Row([
        dbc.Col(dbc.Card([
            dbc.CardHeader("Sentiment Analysis Summary"),
            dbc.CardBody("Here we would have a short paragraph describing the sentiment analysis of bluesky posts regarding the disaster. These would be updated regularly, maybe with time indirectly proportional to the rate of posts being made about the disaster. I'm not yet sure how our data scientists will produce these paragraphs, but I need to know if a paragraph will be produced every time a user enters or reloads this page, or if we will store paragraphs that will show up the same for every user. Much to discuss...")
        ]), width=6),
        dbc.Col(dbc.Card([
            dbc.CardHeader("Assistance Resources"),
            dbc.CardBody("We want to make it clear that we cannot provide immediate help, as our primary purpose is to provide analysis of social media user's sentiment of these disasters. But we can have a list of links to sources like the Red Cross or NOAA for prevention, preparations, and information on how to deal with these disasters. Could have 2-3 sources for each type of disaster. Maybe add a disclaimer that if the viewer is in immediate danger, they should consider evacuation or contacting first responders.")
        ]), width=6)
    ], className="mb-4"),

    # Row with three segments stacked vertically
    dbc.Row([
        dbc.Col(html.Div([
            html.H4("Posts of Note"),
            html.P("Here we would embed several posts with high relevance and engagement. How to measure these? idk. But also, do we want to censor posts for profanity?")
        ]), width=12, className="mb-3"),
        dbc.Col(html.Div([
            html.H4("Mentions Over Time"),
            html.P("line graph, x axis : number of posts, y axis : time(ideally scales with passage of time)")
        ]), width=12, className="mb-3"),
        dbc.Col(html.Div([
            html.H4("Frequently Used Words"),
            html.Img(src="", alt="word cloud here, separate tab in this section for graph of words used?")
        ]), width=12, className="mb-3")
    ])
], fluid=True)

# Run the app
if __name__ == "__main__":
    app.run(debug=True)