import plotly.plotly as py
py.plotly.tools.set_credentials_file(username='lpegram', api_key='wc3S5UG8FWOPiq5ZgyAA')


import pandas as pd
from IPython.display import display

df = pd.read_csv(r'C:\Users\Lisa.Pegram\Downloads\ABA_Opp.csv')
df['text']= df['Lat'].astype(str) + ', ' + df['Longitude'].astype(str) + ''

# Success! Display the first 5 rows of the dataset
display(df.head(n=5))
display(df.info())

# # # Store our latitude and longitude
latitudes = df["Lat"]
longitudes = df["Longitude"]
#
scl = [ [0,"rgb(5, 10, 172)"],[0.35,"rgb(40, 60, 190)"],[0.5,"rgb(70, 100, 245)"],\
    [0.6,"rgb(90, 120, 245)"],[0.7,"rgb(106, 137, 247)"],[1,"rgb(220, 220, 220)"] ]

data = [ dict(
        type = 'scattergeo',
        locationmode = 'USA-states',
        lon = longitudes,
        lat = latitudes,
        text = df['text'],
        mode = 'markers',
        marker = dict(
            size = 8,
            opacity = 0.8,
            reversescale = True,
            autocolorscale = False,
            symbol = 'circle',
            line = dict(
                width=1,
                color='rgba(102, 102, 102)'
            ),
            colorscale = scl,
            cmin = 0,
            color = df['Count'],
            cmax = df['Count'].max(),
            colorbar=dict(
                title="ABA Originating Prospects"
            )
        ))]
# this is where you would apply the additional data in the data array I would change the color
# https://plot.ly/python/line-and-scatter/

layout = dict(
        title = 'ABA Originating Prospects',
        #colorbar = True,
        colorbar=False,
        geo = dict(
            scope='usa',
            projection=dict( type='albers usa' ),
            showland = True,
            landcolor = "rgb(250, 250, 250)",
            subunitcolor = "rgb(105, 105, 105)",
            countrycolor = "rgb(105, 105, 105)",
            countrywidth = 0.5,
            subunitwidth = 0.5
        ),
    )

fig = dict( data=data, layout=layout )
py.iplot( fig, validate=False, filename='ABA_geosegment' )