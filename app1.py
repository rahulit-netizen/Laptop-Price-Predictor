import streamlit as st
import pickle
import numpy as np

# import the model
pipe = pickle.load(open(r"pipe.pkl",'rb'))
df = pickle.load(open(r"df.pkl",'rb'))

st.title("Laptop Price Predictor")

# brand
brand_op_map={
    'DELL':4,'LENOVO':10,'HP':7,'ASUS':2,'ACER':0,'MSI':11,'TOSHIBA':16,'APPLE':1,'SAMSUNG':15,'RAZER':14,'MEDIACOM':12,'MICROSOFT':13,'XIAOMI':18,'VERO':17,'CHUWI':3,'GOOGLE':6,'FUJITSU':5,'LG':9,'HUAWEI':8 
}
company = st.selectbox('Brand',options=brand_op_map)
if company in brand_op_map:
    brand = brand_op_map[company]

# type of laptop
type_map={
    'Notebook':3,'Gaming':1,'Ultrabook':4,'2 in 1 convertible':0,'Workstation':5,'Netbook':2
}
type = st.selectbox('Type',options=type_map)
if type in type_map:
    type = type_map[type]

# Ram
ram = st.selectbox('RAM(in GB)',[2,4,6,8,12,16,24,32,64])

# weight
weight = st.number_input('Weight of the Laptop')

# Touchscreen
touchscreen = st.selectbox('Touchscreen',['No','Yes'])

# IPS
ips = st.selectbox('IPS',['No','Yes'])

# screen size
screen_size = st.number_input('Screen Size(in inches)')

# resolution
resolution = st.selectbox('Screen Resolution',['1920x1080','1366x768','1600x900','3840x2160','3200x1800','2880x1800','2560x1600','2560x1440','2304x1440'])

#cpu
cpu = st.selectbox('CPU',df['Cpu brand'].unique())
#hdd
hdd = st.selectbox('HDD(in GB)',[0,128,256,512,1024,2048])
#ssd
ssd = st.selectbox('SSD(in GB)',[0,8,128,256,512,1024])

#gpu
GPU_map={
    'Intel':1,'Nvidia':2,'AMD':0
}
gpu = st.selectbox('GPU',options=GPU_map)
if gpu in GPU_map:
    gpu=GPU_map[gpu]

#os
OS_map={
    'Windows':2,'MAC':0,'Others/No OS/Linus':1
}
os = st.selectbox('OS',options=OS_map)
if os in OS_map:
    os=OS_map[os]

if st.button('Predict Price'):
    # query
    ppi = None
    if touchscreen == 'Yes':
        touchscreen = 1
    else:
        touchscreen = 0

    if ips == 'Yes':
        ips = 1
    else:
        ips = 0

    X_res = int(resolution.split('x')[0])
    Y_res = int(resolution.split('x')[1])
    ppi = ((X_res**2) + (Y_res**2))**0.5
    ppi=ppi/screen_size
    query = np.array([brand,type,ram,weight,touchscreen,ips,ppi,cpu,hdd,ssd,gpu,os])

    query = query.reshape(1,12)
    # st.title("The predicted price of this configuration is " + str(int(np.exp(pipe.predict(query)[0]))))
    st.write("The predicted price of this configuration is Rs "+str(pipe.predict(query)))
