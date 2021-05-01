import streamlit as st
import base64
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from streamlit_folium import folium_static
import geopandas as gpd
import folium
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
import plotly.graph_objects as go
import pydeck as pdk
import urllib.request as request
import json
#from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score





st.title('Find the best house investment in Madrid')

st.image('https://www.unicainmobiliaria.com/fileuploads/user/viviendas%20de%20lujo%20en%20Madrid_UNICA%20Inmobiliaria.jpg')



st.header('Analize the housing market in Madrid and find the ***best home*** for you')
st.write('')
st.write('')
st.write('Please select the area to show you the top 10 houses by estimated profit now')

#Loading data

@st.cache
def load_data():
    path_res = 'https://drive.google.com/uc?export=download&id=1eudftZPB8pt49_hEkR1destT7P0FRFTR'
    results_mh = pd.read_csv(path_res)
    results_mh.drop(columns=['Unnamed: 0'], inplace=True)
    return results_mh

results_mh=load_data()


#Area selection
unique_zone=sorted(results_mh.zona.unique())
selected_zone= st.multiselect('Area', unique_zone, unique_zone)

df_selected_zone= results_mh[(results_mh.zona.isin(selected_zone))]
df_selected_zone.sort_values(by=['profit_now'], axis=0, ascending= False, inplace=True)

#Table

figtable= go.Figure(data=[go.Table(
    columnwidth=[200,300],
    header=dict(values=['Post Code', 'Area', 'District', 'Size (m2)', 'Rooms', 'Bathrooms', 'Price (€)', '€ per m2', 'Inmediate Profit (%)', 'Future Profit (%)'],
                fill_color='darkgrey',
                align='center',
                font=dict(color='white', size=12)),
    cells=dict(values=[df_selected_zone.post_code, df_selected_zone.zona, df_selected_zone.poblacion, df_selected_zone['size'], df_selected_zone.rooms,
    df_selected_zone.bathrooms, df_selected_zone.prices, df_selected_zone.market_per_m2.astype(int),
    (df_selected_zone.profit_now*100).round(2), (df_selected_zone.profit_future*100).round(2)],
               fill_color='lightgrey',
               align='center',
               font_size=10))])
figtable.update_layout(height=250, width=800, margin=dict(l=0,r=0,b=0,t=0))
st.plotly_chart(figtable)


#Boxplot by area
st.write('The boxplot shows how each zone compares to each other showing averages and quartiles')

fig1= plt.figure(figsize=(11,6))
ax1= sns.boxplot(x= 'zona',y= 'prices',data= df_selected_zone, palette= 'Set2').set_title('Prices By Area')
st.pyplot(fig1)


#Boxplot by number of rooms
figroom= plt.figure(figsize=(11,6))
axroom= sns.boxplot(x= 'rooms',y= 'market_per_m2',data= df_selected_zone, palette='Blues').set_title('Price/m2 By Number of Rooms')
st.pyplot(figroom)


#Line chart of prices by floor size
st.write('Now analize how prices and prices/m2 vary depending on size of the house')
df_sizes_index=df_selected_zone.groupby(by='size').median()
st.line_chart(df_sizes_index['prices'], width=1000, height=350)

df_sizes_index=df_selected_zone.groupby(by='size').median()
st.line_chart(df_sizes_index['market_per_m2'], width=1000, height=350)


#Correlation graph
st.write('We can see in the correlation graph that centre area is highly correlated with the prices. Also the size is obviously highly correlated with the number of rooms and bathrooms as we can expect')

onehot_enc_zones = pd.get_dummies(results_mh['zona'])
results_mh_corr = pd.concat([results_mh, onehot_enc_zones], axis=1, join="inner")
fig2, ax = plt.subplots(figsize=(12,10))
corr = results_mh_corr.corr()
ax2= sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap="RdBu", linewidths = 0.5,
            square=True, ax=ax)
st.pyplot(fig2)


#Interactive map by area using Cloropleth
st.write('Map heatmap to see the prices distribution using the market price and the two predictions')
selected_prediction= st.select_slider('Slide to select the prediction on the map', ['market_per_m2', 'pred_2020_market', 'price_pred_2030'])

mad_map= gpd.read_file('https://raw.githubusercontent.com/inigoflores/ds-codigos-postales/d3036e99d124582b1c5c69660bd8d1c6bd0b7af0/data/MADRID.geojson')
mad_map['COD_POSTAL']=mad_map['COD_POSTAL'].astype(int)

df_4_maps=results_mh.groupby(by='post_code').median()
df_4_maps.reset_index(inplace=True)

df_4_maps['post_code']=df_4_maps['post_code'].astype(int)
df_4_maps['rooms']=df_4_maps['rooms'].astype(int)
df_4_maps['bathrooms']=df_4_maps['bathrooms'].astype(int)

graphmap_mad= folium.Map(location=[40.4881, -3.6683], zoom_start=9)

bins_mkt = list(df_4_maps[selected_prediction].quantile([0, 0.15, 0.3, 0.45, 0.6, 0.75, 0.9, 1]))

folium.Choropleth(
    geo_data=mad_map,
    name="prices",
    data=df_4_maps,
    columns=["post_code", selected_prediction],
    key_on="feature.properties.COD_POSTAL",
    fill_color="YlOrRd",
    nan_fill_color='#fff7bc',
    hover_name='Post Code',
    hover_data=[selected_prediction],
    fill_opacity=0.7,
    line_opacity=0.6,
    line_color='white',
    bins=bins_mkt,
    highlight=True,
    reset=True,
    show=False,
    control=True
).add_to(graphmap_mad)

folium_static(graphmap_mad)



# Dynamic 3D graph showing prices per m2 in Madrid by post code

st.write(' In this dynamic 3D map we can see prices are more under their estimation in the outskirts which means the centre is not an opportunity at the moment but it´s a more secure investment because are going to keep growing in the next years')

map=st.empty()

st.cache(allow_output_mutation=True)
def pdk_data():
    with request.urlopen('https://raw.githubusercontent.com/palvgoya/house_investment_finder/main/pdk_madhouse_df.geojson') as response:
            if response.getcode() == 200:
                source = response.read()
                pydeck_data = json.loads(source)
            else:
                print('An error occurred while attempting to retrieve data from the API.')
    return pydeck_data

pydeck_data1= pdk_data()

geojson_layer= pdk.Layer(
'GeoJsonLayer',
data= pydeck_data1,
opacity=0.8,
stroked=False,
filled=True,
extruded=True,
wireframe=True,
get_elevation= 'properties.profit_now * 100000',
get_fill_color= [240, 237,55],
get_line_color=[195,195,195],
pickable=False)

onsale_pdk= pdk.Deck(
map_style='mapbox://styles/mapbox/light-v9',
     initial_view_state=pdk.ViewState(
         latitude=40.4081,
         longitude=-3.6683,
         zoom=8,
         pitch=50),
     layers=[geojson_layer])

map.pydeck_chart(onsale_pdk)
#onsale_pdk.to_html()
#st.pydeck_chart(onsale_pdk)


#Lets build a house price estimator
st.header('Please introduce the information below to know the price of your house now and the prediction we have for 2030')
st.write('')
st.write('')

@st.cache
def load_x():
    x_data_path = 'https://drive.google.com/uc?export=download&id=1OqfPxoMy_yeRugMacQY00aGfrqYk09sK'
    x = pd.read_csv(x_data_path)
    x.drop(columns=['Unnamed: 0'], inplace=True)
    return x

@st.cache
def load_y():
    y_data_path = 'https://drive.google.com/uc?export=download&id=15Fj0lz9xCTMOarA2dJQg_8MCQRt2H1zY'
    y = pd.read_csv(y_data_path)
    y.drop(columns=['Unnamed: 0'], inplace=True)
    return y

x= load_x()
y= load_y()

@st.cache
def scaler():
    scaler= MinMaxScaler(feature_range=(1, 4))
    scaler.fit(results_mh[['size']])
    return scaler

scaler=scaler()

selected_rooms= st.number_input('Rooms', value= 2)
selected_bathrooms= st.number_input('Bathrooms', value= 2)
selected_size= st.number_input('Size', value=30)
#selected_post_code= st.number_input('Post Code', 28000, 28991, 28001)
selected_post_code= st.selectbox('Post Code', df_4_maps['post_code'])
selected_size_trans= scaler.transform(np.array([selected_size]).reshape(-1, 1))

input=pd.DataFrame([[selected_rooms, selected_bathrooms, selected_size_trans, 1, 1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0]], columns= list(x.columns.astype(str)))
input[str(selected_post_code)]=1


if st.button ('Estimate'):
    rf_reg=RandomForestRegressor(max_features= 'sqrt', n_estimators= 300, min_samples_leaf= 1, n_jobs=-2)
    rf_reg.fit(x, y)
    price_est= rf_reg.predict(input)
    postcode_avg= df_4_maps['prices'][df_4_maps['post_code']==selected_post_code]
    st.write('The value estimation for your house is')

    fig3= plt.figure()
    ax3= sns.barplot(x= ['Prediction', 'Average'],y= [price_est, int(postcode_avg)], palette='Blues_d')
    for p in ax3.patches:
                 ax3.annotate("%.0f€" % p.get_height(), (p.get_x() + p.get_width() / 2., p.get_height()),
                     ha='center', va='center', fontsize=10, color='black', xytext=(0, 5),
                     textcoords='offset points')
    st.pyplot(fig3)
    st.success('Your Report is Ready!')
else:
    print('Introduce the house to be estimated')
    st.success('Your Report is Ready!')
    st.stop()
