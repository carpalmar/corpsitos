# Principales librerias

import pandas as pd
import streamlit as st
import numpy as np
from PIL import Image
import plotly.express as px


# Importar el Excel

## Oficina
excel = pd.ExcelFile('BASE.xlsx')
## Jato
#excel = pd.ExcelFile(r'C:\Users\carpa\OneDrive\Documents\Proyectos\Corps Spreads\BASE.xlsx')

# Agregando las series de tiempo
YTW = excel.parse('YTW',header =4,index_col=0)
ZSPD = excel.parse('ZSPD',header =4,index_col=0)

# Agregando las tablas que necesitamos
filtros = excel.parse('SCREEN',header =2,index_col=0)
filtros.reset_index(inplace=True)

# Listas para filtros:
## Filtros categoricos
paises = filtros['CNTRY_OF_RISK'].sort_values().unique()
bbcomposite = filtros['BB_COMPOSITE'].sort_values().unique()
calls = filtros['CALLABLE'].unique()
nombres = filtros['Codigo'].unique()
## Filtros continuos
duraciones = filtros['DUR_ADJ_MID'].sort_values().unique()
fechas_emision = filtros['ISSUE_DT'].sort_values().unique()
madureces = filtros['MATURITY'].sort_values().unique()




# Funciones:

# Filtrar por pais y rating
@st.cache
def filtrado(panda, cntry, ratng, dur_min,dur_max):
    filtro = panda[panda['CNTRY_OF_RISK'].isin(cntry)]
    filtro = filtro[filtro['BB_COMPOSITE'].isin(ratng)]
    #filtro = filtro.query(dur)
    filtro = filtro[(filtro['DUR_ADJ_MID'] >= dur_min) & (filtro['DUR_ADJ_MID'] <= dur_max)]
    ISIN = filtro['Isin'].tolist()
    CODE = filtro['Codigo'].tolist()
    diccionario = dict(zip(filtro.Isin,filtro.Codigo))
    return filtro, ISIN, diccionario, CODE

# Extraccion de datos de YTW y ZSPD
@st.cache
def extraccion(panda,lista):
    minipanda = pd.DataFrame(index=panda.index)
    for i in lista:
        minipanda[i] = panda[i]
    return minipanda

# Matriz de diferencias
@st.cache
def spreads(panda):
    columnas = panda.columns
    spread = pd.DataFrame(index=panda.index)
    for i in range(len(columnas)):
        for j in range(len(columnas)):
            if i != j: 
                name = str(columnas[i]+"-"+columnas[j])
                spread[name] = panda[columnas[i]] - panda[columnas[j]]
    return spread



# Calcular Z-score movil
@st.cache
def zscore(panda,periodo):
    columnas = panda.columns
    zscore = pd.DataFrame(index=panda.index)
    for i in range(len(columnas)):
        zscore[columnas[i]] = (panda[columnas[i]]-panda[columnas[i]].rolling(periodo).mean()) / panda[columnas[i]].rolling(periodo).std()
    return zscore


# Hacer la tabla de zscore
@st.cache
def style_negative(value):
    color = 'red' if value < -2 else None
    color = 'green' if value > 2 else None
    return 'color: %s' % color


def tabla_zscore(panda,periodo):
    columnas = panda.columns
    spread = pd.DataFrame(index=panda.index)
    spread_m = pd.DataFrame(index=panda.index)
    spread_d = pd.DataFrame(index=panda.index)
    zscore = pd.DataFrame(index=panda.index)
    tabla_z = pd.DataFrame(index=pd.Index(columnas,name='Venta\Compra'),columns=columnas)
    tabla_s = pd.DataFrame(index=pd.Index(columnas,name='Venta\Compra'),columns=columnas)
    tabla_d = pd.DataFrame(index=pd.Index(columnas,name='Venta\Compra'),columns=columnas)
    tabla_m = pd.DataFrame(index=pd.Index(columnas,name='Venta\Compra'),columns=columnas)
    for i in range(len(columnas)):
        for j in range(len(columnas)):
            if i != j:
                spread[columnas[i] + '-' + columnas[j]] = panda[columnas[i]] - panda[columnas[j]]
                spread_m[columnas[i] + '-' + columnas[j]] = spread[columnas[i] + '-' + columnas[j]].rolling(periodo).mean()
                spread_d[columnas[i] + '-' + columnas[j]] = spread[columnas[i] + '-' + columnas[j]].rolling(periodo).std()
                zscore[columnas[i] + '-' + columnas[j]] = (spread[columnas[i] + '-' + columnas[j]]-spread_m[columnas[i] + '-' + columnas[j]]) / spread_d[columnas[i] + '-' + columnas[j]]
                last_s = spread[columnas[i] + '-' + columnas[j]].iloc[-1]
                last_m = spread_m[columnas[i] + '-' + columnas[j]].iloc[-1]
                last_d = spread_d[columnas[i] + '-' + columnas[j]].iloc[-1]
                last_z = zscore[columnas[i] + '-' + columnas[j]].iloc[-1]
                tabla_z[columnas[i]][columnas[j]] = last_z.round(2)
                tabla_s[columnas[i]][columnas[j]] = last_s.round(2)
                tabla_m[columnas[i]][columnas[j]] = last_m.round(2)
                tabla_d[columnas[i]][columnas[j]] = last_d.round(2)

    tabla_z = tabla_z.style.applymap(style_negative)
    tabla_z.precision=2
    return tabla_z,tabla_s,tabla_m,tabla_d

@st.cache
def rank_spread(zeta,spread,media,bono):
    zeta = zeta.data
    rank = pd.DataFrame(index=pd.Index(zeta.index,name='Venta'))
    rank['Z-Score'] = zeta[bono]
    rank['Spread Actual'] = spread[bono]
    rank['Compresión Potencial'] = spread[bono]- media[bono]
    rank = rank.dropna()
    rank = rank.sort_values(by='Z-Score',ascending=False)
    first, last = rank.index[0],rank.index[-1]
    return rank,first,last

@st.cache
def panda_mejor_peor(panda,bono,first,periodo):
    sprd = bono+"-"+first
    top_spread = pd.DataFrame(index=panda.index)
    top_spread[sprd] = panda[sprd]
    top_spread['Movav'] = top_spread[sprd].rolling(periodo).mean()
    top_spread['Std'] = top_spread[sprd].rolling(periodo).std()
    top_spread['+2std'] = top_spread['Movav'] + 2*top_spread['Std']
    top_spread['-2std'] = top_spread['Movav'] - 2*top_spread['Std']
    top_spread = top_spread.dropna()
    return top_spread, sprd


# Funcion para graficar spreads
@st.cache
def graph_spread(panda_spread,bono1,bono2,periodo):
    sprd = bono1+"-"+bono2
    brecha = pd.DataFrame(panda_spread[sprd])
    brecha['Movav'] = brecha[sprd].rolling(periodo).mean()
    brecha['Std'] = brecha[sprd].rolling(periodo).std()
    brecha['+2std'] = brecha['Movav'] + 2*brecha['Std']
    brecha['-2std'] = brecha['Movav'] - 2*brecha['Std']
    brecha = brecha.dropna()
    brecha['Fecha'] = brecha.index
    fig = px.line(brecha, x='Fecha', y=[sprd,'Movav','+2std','-2std'], title="Spread "+bono1+"-"+bono2)
    return fig
    



######################################################

# Usar Streamlit para crear la interfaz
imagen = Image.open('logo_web_peru.png')

st.image(imagen)

st.write("""
# Analisis de Corporativos

Muestra el analisis de los spreads de los bonos de una empresa.         
         """)


# CREAR EL SIDEBAR
st.sidebar.header('Filtros')
##  FILTROS DE SIDEBAR PAIS Y RATING
selected_pais = st.sidebar.multiselect('País',paises,'PE')
selected_rating = st.sidebar.multiselect('Rating',bbcomposite,['BB','BB+','BB-'])
## CREATE FILTER OF a range of DURATION
dur_min = int(0)
dur_max = int(duraciones.max().round(0)+2)
lista_dur = list(range(dur_min,dur_max))
selected_dur = st.sidebar.slider('Duración',dur_min,dur_max,(2,8))

filtro, isin_list, codigo, code = filtrado(filtros, selected_pais, selected_rating,
                                           selected_dur[0],selected_dur[1])

selected_bond = st.sidebar.selectbox('Bono a analizar',code)

selected_dias = st.sidebar.slider('Dias para el análisis',20,200)

refiltro = pd.DataFrame(filtro[['SECURITY_NAME','CNTRY_OF_RISK','ISSUE_DT','MATURITY','CPN','DV01','YLD_CNV_LAST','PX_MID']])
st.header('Corporativos Filtrados')
st.write('Es el resultado de los filtros aplicados')
st.dataframe(refiltro)

# Extraccion de datos de YTW y ZSPD
#ZSPD_extra = extraccion(ZSPD,code)
YTW_extra = extraccion(YTW,code)
# Calcular spreads
Spreads = spreads(YTW_extra)

# Hacer las tablas
zeta,spread,media,desvi = tabla_zscore(YTW_extra,selected_dias)

st.header('Z-Score')
st.write('Es el Z-Score de los spreads de los bonos')
st.dataframe(zeta)

# Calcular ranking para un bono de la lista
ranki, first, last = rank_spread(zeta,spread,media,selected_bond)

st.header('Ranking')
st.write('Es el ranking del bono z-spread de '+selected_bond)
st.dataframe(ranki)


segundo_bono = st.sidebar.selectbox('Bono a comparar',ranki.index)

fig =graph_spread(Spreads,selected_bond,segundo_bono,selected_dias)
st.plotly_chart(fig, use_container_width=True)    