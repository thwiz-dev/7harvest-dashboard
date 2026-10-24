import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from datetime import timedelta

st.set_page_config(page_title='7Harvest Commercial Dashboard', page_icon='🌾', layout='wide')

st.markdown('''
<style>
:root {
    --cream:#f7f2e8;
    --paper:#fffdf9;
    --paper-2:#ffffff;
    --ink:#2b3340;
    --muted:#667489;
    --border:#e6dccb;
    --gold:#c7a65a;
    --gold-soft:#f7edd6;
    --blue:#79a7d8;
    --blue-soft:#edf3fb;
    --rose:#d96b78;
    --rose-soft:#f8e8ec;
    --plum:#a98dc7;
    --plum-soft:#f1ebf8;
    --mauve:#eadff3;
    --amber-soft:#faf1cf;
    --red-soft:#f9e1e4;
    --green-soft:#eaf5eb;
}
html, body, [class*="css"] {color:var(--ink)}
.stApp{background:linear-gradient(180deg,#fbf8f1 0%, #f6f1e6 100%);color:var(--ink)}
.block-container{padding-top:1.2rem;padding-bottom:2rem;max-width:1460px}
h1,h2,h3{letter-spacing:-.03em;color:var(--ink)}
p,li,div,label,.stMarkdown,.stCaption{font-size:.97rem;color:var(--ink)}
section[data-testid="stSidebar"]{background:linear-gradient(180deg,#fcfaf5 0%, #f3ecdf 100%);border-right:1px solid var(--border)}
section[data-testid="stSidebar"] *{color:var(--ink)!important}
.hero-wrap{background:linear-gradient(135deg,#faf7ef 0%, #fffdfa 56%, #f6efdf 100%);border:1px solid var(--border);padding:1.1rem 1.2rem;border-radius:20px;margin-bottom:1rem;box-shadow:0 10px 28px rgba(115,98,59,.05)}
.brand-chip{display:inline-block;font-size:.76rem;font-weight:700;letter-spacing:.06em;text-transform:uppercase;color:#5b4b71;background:#f2ebfa;border:1px solid #e2d7f2;border-radius:999px;padding:.34rem .62rem;margin-bottom:.6rem}
.hero-title{font-size:2.12rem;font-weight:780;line-height:1.08;margin:0;color:var(--ink)}
.hero-sub{margin-top:.35rem;color:var(--muted);font-size:.97rem}
.section-note{background:var(--blue-soft);border:1px solid #dbe6f1;padding:.9rem 1rem;border-radius:14px;color:#41576c;margin-bottom:1rem}
[data-testid="stMetric"]{background:var(--paper-2);border:1px solid var(--border);padding:14px 16px;border-radius:16px;min-height:112px;box-shadow:0 8px 20px rgba(115,98,59,.04)}
[data-testid="stMetricLabel"]{font-size:.82rem;color:var(--muted)}
[data-testid="stMetricValue"]{font-size:clamp(1.25rem,2vw,2rem);letter-spacing:-.03em;color:var(--ink)}
[data-testid="stMetricDelta"]{font-size:.8rem}
[data-testid="stTabs"] button{font-weight:650;color:var(--ink);padding-top:.65rem;padding-bottom:.7rem}
[data-baseweb="tab-highlight"]{background-color:var(--rose)!important;height:2px!important}
/* Keep all Streamlit widgets consistent with the pale rose/lilac brand theme. */
[data-baseweb="select"] > div,
[data-baseweb="input"] > div,
[data-baseweb="base-input"] > div,
[data-testid="stNumberInput"] > div > div,
[data-testid="stTextInput"] > div > div,
[data-testid="stSelectbox"] [data-baseweb="select"] > div,
[data-testid="stMultiSelect"] [data-baseweb="select"] > div {
  background:#eee3f2 !important;
  background-color:#eee3f2 !important;
  color:#293242 !important;
  border:1px solid #dac8e6 !important;
  border-radius:10px !important;
  box-shadow:none !important;
}
[data-baseweb="select"] span,
[data-baseweb="select"] div,
[data-baseweb="input"] input,
[data-baseweb="input"] button,
[data-testid="stNumberInput"] input,
[data-testid="stNumberInput"] button {color:#293242 !important;}
[data-baseweb="select"] svg, [data-testid="stNumberInput"] svg {fill:#69557c !important;color:#69557c !important;}
/* Dropdown menu is a popover, separate from the input widget. */
div[data-baseweb="popover"],div[data-baseweb="popover"] ul,
ul[role="listbox"],li[role="option"] {
  background:#f8f0fa !important;
  color:#293242 !important;
}
li[role="option"]:hover {background:#eadcf3 !important;}
/* Softer dataframes and editable grid headers. Light base theme handles the inner Glide canvas. */
[data-testid="stDataFrame"], [data-testid="stDataEditor"],
[data-testid="stDataFrame"] > div, [data-testid="stDataEditor"] > div {
  background:#fffdfa !important;
  border-color:#e6dae9 !important;
}
[data-testid="stDataFrame"] [role="grid"], [data-testid="stDataEditor"] [role="grid"] {
  background:#fffdfa !important;
  color:#293242 !important;
}

/* Input widgets */
div[data-baseweb="select"] > div, div[data-baseweb="base-input"] > div, .stNumberInput div[data-baseweb="input"] > div {
    background:linear-gradient(180deg,#f4eef8 0%, #eee5f5 100%) !important;
    border:1px solid #dfd0ec !important;
    border-radius:12px !important;
    color:var(--ink) !important;
}
input, textarea{color:var(--ink)!important}
.stSlider [data-baseweb="slider"]{padding-top:.4rem}
.stSlider [role="slider"]{background:#d96b78!important}
.stSlider [data-testid="stTickBar"]{background:#dbcfe8!important}
/* Buttons */
.stButton>button,.stDownloadButton>button{border-radius:12px;border:1px solid #d0bb88;background:linear-gradient(180deg,#f1dfaf 0%, #e2c980 100%);color:#43351a;font-weight:700;padding:.55rem 1rem}
.stButton>button:hover,.stDownloadButton>button:hover{border:1px solid #c4aa6e;color:#2d2414}
/* Data areas */
[data-testid="stFileUploader"] section{border-radius:14px;border:1px dashed #d4c39a;background:#fffdfa}
[data-testid="stDataFrame"], [data-testid="stDataEditor"]{border:1px solid var(--border);border-radius:14px;overflow:hidden;background:#fffdfa}
/* Custom cards */
.consulting-card{background:var(--paper-2);border:1px solid var(--border);padding:1rem 1.05rem;border-radius:16px;height:100%;box-shadow:0 8px 22px rgba(115,98,59,.04)}
.consulting-card h4{margin:0 0 .45rem 0;font-size:1.03rem;color:var(--ink)}
.consulting-card p{margin:0;color:var(--muted);font-size:.92rem;line-height:1.55}
.kicker{display:inline-block;font-size:.76rem;font-weight:700;letter-spacing:.06em;text-transform:uppercase;color:#7d6a3a;margin-bottom:.45rem}
.soft-divider{height:1px;background:var(--border);margin:1rem 0 1.1rem 0}
.takeaway{background:var(--plum-soft);border:1px solid #e0d5ef;border-radius:16px;padding:1rem 1.05rem;color:#4d4761}
.roadmap-card{background:#fff;border:1px solid var(--border);border-left:5px solid var(--plum);padding:.95rem 1rem;border-radius:14px;box-shadow:0 7px 20px rgba(115,98,59,.04);height:100%}
.roadmap-card h5{margin:0 0 .35rem 0;font-size:1rem;color:var(--ink)}
.roadmap-card p{margin:0;color:var(--muted);font-size:.9rem;line-height:1.5}
.small-label{font-size:.82rem;color:var(--muted)}
.status-box{padding:.85rem 1rem;border-radius:14px;border:1px solid transparent;margin:.35rem 0 .6rem 0;font-size:.93rem;line-height:1.55}
.status-red{background:var(--red-soft);border-color:#efc4ca;color:#6e3c44}
.status-amber{background:var(--amber-soft);border-color:#eedf9b;color:#6b5a1c}
.status-green{background:var(--green-soft);border-color:#cfe1d0;color:#36503c}
.status-info{background:var(--blue-soft);border-color:#dbe6f1;color:#41576c}
hr{border-color:var(--border)}
</style>
''', unsafe_allow_html=True)

st.markdown('''
<div class="hero-wrap">
  <div class="brand-chip">7Harvest · Commercial Planning Prototype · V7</div>
  <div class="hero-title">7Harvest Commercial Dashboard</div>
  <div class="hero-sub">Integrated decision-support dashboard for demand planning, commercial scenarios, retail effectiveness and inventory review</div>
</div>
''', unsafe_allow_html=True)

st.markdown('<div class="section-note"><strong>Prototype note:</strong> operational sales, outlet and inventory data are synthetic. Reported brand cost percentages come from the supplied 7Harvest business context. This dashboard is designed for competition demonstration and strategic discussion, not live deployment.</div>', unsafe_allow_html=True)

@st.cache_data
def sample_sales():
    rng=np.random.default_rng(42)
    days=pd.date_range('2026-07-01',periods=105)
    products={'Malabar porotta':155,'Idli/dosa batter':105,'Whole wheat chapati':90,'Appam batter':65}
    rows=[]
    for product,base in products.items():
        for route in ['Route 1','Route 2','Route 3','Route 4']:
            factor={'Route 1':.7,'Route 2':.9,'Route 3':1.1,'Route 4':1.25}[route]
            for i,day in enumerate(days):
                weekend=1.23 if day.dayofweek in [4,5] else 1.0
                units=max(0,int(round(base*factor*weekend*(1+.0015*i)+rng.normal(0,base*.11))))
                rows.append((day,product,route,units))
    return pd.DataFrame(rows,columns=['date','product','route','units'])

@st.cache_data
def sample_sku_review():
    return pd.DataFrame({
        'SKU':['RK Black-eye Beans 1kg','RK Black-eye Beans 500g','RK Biryani Masala','RK Sambar Mix','RK Chilli Powder','RK Turmeric Powder','RK Premium Dry Fruit Pack'],
        'Annual sales AED':[142000,68000,195000,174000,122000,99000,158000],
        'Contribution %':[18,11,31,29,24,21,34],
        'Variants in cluster':[11,11,2,2,4,4,1],
        'Recommendation':['Review / consolidate','Review / consolidate','Keep / scale','Keep / scale','Keep','Keep','Scale as value-added']
    })

def prepare_upload(file):
    if file is None:
        return sample_sales(),False
    data=pd.read_csv(file) if file.name.lower().endswith('.csv') else pd.read_excel(file)
    needed={'date','product','route','units'}
    if not needed.issubset(set(data.columns)):
        raise ValueError('Sales file must contain date, product, route and units columns.')
    data=data[list(needed)].copy()
    data['date']=pd.to_datetime(data['date'],errors='coerce')
    data['units']=pd.to_numeric(data['units'],errors='coerce')
    data=data.dropna(subset=['date','product','route','units'])
    data=data[data.units>=0]
    data['product']=data['product'].astype(str)
    data['route']=data['route'].astype(str)
    if data.empty:
        raise ValueError('No valid sales rows found.')
    return data,True

def daily_features(dates,origin):
    dates=pd.DatetimeIndex(pd.to_datetime(dates))
    X=pd.DataFrame({'trend':(dates-origin).days.astype(float)})
    for day in range(7):
        X[f'weekday_{day}']=(dates.dayofweek==day).astype(int)
    return X

def run_forecast(df,product,route,days=7):
    selected=df[(df['product']==product)&(df['route']==route)].groupby('date',as_index=False).units.sum().sort_values('date')
    if len(selected)<28:
        raise ValueError('At least 28 historical days are required for this product and route.')
    origin=selected.date.min()
    cutoff=max(14,int(len(selected)*.8))
    if cutoff>=len(selected):
        cutoff=len(selected)-7
    X=daily_features(selected.date,origin)
    y=selected.units.to_numpy()
    model=Ridge(alpha=10).fit(X.iloc[:cutoff],y[:cutoff])
    predicted=np.maximum(0,model.predict(X.iloc[cutoff:]))
    holdout_mae=mean_absolute_error(y[cutoff:],predicted)
    training=selected.iloc[:cutoff].copy()
    weekday_means=training.groupby(training.date.dt.dayofweek).units.mean()
    default=training.units.mean()
    baseline=np.array([weekday_means.get(d.dayofweek,default) for d in selected.date.iloc[cutoff:]])
    baseline_mae=mean_absolute_error(y[cutoff:],baseline)
    model=Ridge(alpha=10).fit(X,y)
    future=pd.date_range(selected.date.max()+timedelta(days=1),periods=days)
    forecast=np.maximum(0,np.rint(model.predict(daily_features(future,origin)))).astype(int)
    return pd.DataFrame({'date':future,'predicted_units':forecast}),round(holdout_mae,2),round(baseline_mae,2),len(selected)

with st.sidebar:
    st.markdown('### Dashboard controls')
    uploaded=st.file_uploader('Upload historical sales (CSV or Excel)',type=['csv','xlsx'])
    st.caption('Required fields: date, product, route, units. For demo purposes only; avoid uploading confidential records to an unapproved environment.')
    try:
        sales,real=prepare_upload(uploaded)
        st.success(f'{len(sales):,} sales rows loaded' + (' from upload' if real else ' (demo dataset)'))
    except Exception as ex:
        st.error(str(ex))
        sales=sample_sales()
        real=False
    st.markdown('<div class="soft-divider"></div>', unsafe_allow_html=True)
    st.caption('Brand cost percentages are from the 7Harvest Business Context (pages 6–7). Scenario tables elsewhere use editable assumptions.')
    if st.button('Refresh dashboard'):
        st.rerun()

if 'inventory_defaults' not in st.session_state:
    st.session_state.inventory_defaults = pd.DataFrame({
        'Product':['Malabar porotta','Idli/dosa batter','Whole wheat chapati','Appam batter'],
        'Saleable stock units':[260,570,120,340],
        'Expected units per day':[155,95,85,65],
        'Days remaining shelf life':[4,4,5,4]
    })
if 'latest_forecast_context' not in st.session_state:
    st.session_state.latest_forecast_context = None


def plot_style(fig):
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='#fffdfa',
        font=dict(color='#2b3340', size=13),
        margin=dict(l=10,r=10,t=45,b=12)
    )
    fig.update_yaxes(gridcolor='rgba(120,120,120,.13)', zerolinecolor='rgba(120,120,120,.13)', tickfont=dict(color='#49566b'))
    fig.update_xaxes(gridcolor='rgba(120,120,120,.08)', tickfont=dict(color='#49566b'))
    return fig

# reusable data styling

def light_style(df, color='#f6eff9'):
    return (df.style
            .set_properties(**{'background-color':'#fffdfa','color':'#2b3340','border-color':'#eadfce'})
            .set_table_styles([
                {'selector':'thead th','props':[('background-color',color),('color','#2b3340'),('border','1px solid #eadfce')]},
                {'selector':'tbody td','props':[('border','1px solid #f0e7d8')]}
            ]))


tabs=st.tabs(['Executive summary','Demand planning','Commercial scenarios','Retail effectiveness','Inventory review'])

with tabs[0]:
    st.subheader('Business performance summary')
    a,b,c,d=st.columns(4)
    a.metric('RK contribution before overheads','25.50%')
    b.metric('Soorya contribution before overheads','45.20%')
    c.metric('Soorya expiry / damage','2.00%')
    d.metric('RK trade expense','10.26%')
    st.caption('RK trade expense = rent 1.50 + pallet 1.25 + gondola 0.88 + price-offs 2.25 + promoter 1.50 + rebate 2.88.')

    costs=pd.DataFrame({
        'Cost':['Cost of goods','Vehicles','Trade expense','Expiry / damage','Residual before overheads'],
        'RK Foods':[62.5,1.75,10.26,0,25.49],
        'Soorya Foods':[40,5,7.8,2,45.2]
    })
    fig=px.bar(
        costs.melt(id_vars='Cost',var_name='Brand',value_name='Percent'),
        x='Brand',y='Percent',color='Cost',barmode='stack',
        title='Illustrative 100 AED sales allocation by brand',
        color_discrete_map={
            'Cost of goods':'#79a7d8',
            'Vehicles':'#c9b6dc',
            'Trade expense':'#d96b78',
            'Expiry / damage':'#ee9aa7',
            'Residual before overheads':'#9f86c8'
        }
    )
    fig.update_layout(yaxis_title='% of sales',legend_title_text='')
    fig = plot_style(fig)
    st.plotly_chart(fig,use_container_width=True)

    st.markdown('### Strategic direction')
    c1,c2,c3=st.columns(3)
    with c1:
        st.markdown('''<div class="consulting-card"><div class="kicker">Objective 1</div><h4>Create value beyond price</h4><p>Prioritise SKU rationalisation and higher-value offers under <strong>RK Kitchen</strong>, develop recurring <strong>foodservice partnerships</strong>, and improve Soorya planning through <strong>SmartFresh</strong>.</p></div>''', unsafe_allow_html=True)
    with c2:
        st.markdown('''<div class="consulting-card"><div class="kicker">Objective 2</div><h4>Strengthen retail partnerships</h4><p>Evaluate listing fees, rebate intensity and retail-support spending using outlet-level contribution logic rather than broad trade-spend assumptions.</p></div>''', unsafe_allow_html=True)
    with c3:
        st.markdown('''<div class="consulting-card"><div class="kicker">Prototype role</div><h4>Management decision support</h4><p>The dashboard combines demand planning, commercial scenario testing, retail evaluation and stock review into one integrated management view for pilot discussions.</p></div>''', unsafe_allow_html=True)

    st.markdown('### Suggested pilot roadmap')
    r1,r2,r3=st.columns(3)
    with r1:
        st.markdown('''<div class="roadmap-card"><h5>Phase 1 · Baseline</h5><p>Validate product-level sales history, stock write-offs and outlet trade-spend data. Confirm which figures will be used as the pilot baseline.</p></div>''', unsafe_allow_html=True)
    with r2:
        st.markdown('''<div class="roadmap-card"><h5>Phase 2 · Pilot</h5><p>Run SmartFresh on selected Soorya routes, and evaluate one focused foodservice / retail commercial test instead of full rollout.</p></div>''', unsafe_allow_html=True)
    with r3:
        st.markdown('''<div class="roadmap-card"><h5>Phase 3 · Review</h5><p>Compare expiry, stockouts, contribution and retail-investment recovery against the baseline before recommending scale-up.</p></div>''', unsafe_allow_html=True)

    st.markdown('<div class="takeaway"><strong>Management takeaway:</strong> 7Harvest should not treat forecasting, channel strategy and retail spending as separate decisions. The strongest opportunity is to connect them through controlled pilots and outlet-level contribution analysis.</div>', unsafe_allow_html=True)

    st.warning('The company context contains conflicting RK/Soorya sales-mix figures (67/33 in the chart versus 73/27 in the text). Confirm the correct split before using a weighted group forecast.')

with tabs[1]:
    st.subheader('Soorya SmartFresh — demand planning model')
    st.caption('Estimate short-term product demand by route to support production and replenishment planning. The current model is fitted on the loaded dataset.')
    col1,col2,col3,col4=st.columns(4)
    product=col1.selectbox('Product',sorted(sales['product'].unique()))
    route=col2.selectbox('Route',sorted(sales['route'].unique()))
    stock=col3.number_input('Available saleable units',min_value=0,value=120,step=10)
    buffer=col4.slider('Safety buffer %',0,50,10)
    if st.button('Run forecast',type='primary'):
        try:
            prediction,mae,baseline,nrows=run_forecast(sales,product,route)
            tomorrow=int(prediction.iloc[0].predicted_units)
            suggested=max(0,int(np.ceil(tomorrow*(1+buffer/100)-stock)))
            st.session_state.latest_forecast_context={'product':product,'route':route,'tomorrow_forecast':tomorrow,'buffer':buffer,'suggested_additional_load':suggested,'mae':mae,'baseline':baseline}
            inv=st.session_state.inventory_defaults.copy()
            mask=inv['Product']==product
            if mask.any():
                inv.loc[mask,'Expected units per day']=tomorrow
                inv.loc[mask,'Saleable stock units']=stock
            st.session_state.inventory_defaults=inv

            a,b,c,d=st.columns(4)
            a.metric('Tomorrow forecast',f'{tomorrow:,} units')
            b.metric('7-day demand',f'{prediction.predicted_units.sum():,} units')
            c.metric('Suggested additional load',f'{suggested:,} units')
            d.metric('Holdout MAE',f'{mae:.1f} units')

            plot=go.Figure()
            plot.add_trace(go.Scatter(x=prediction['date'].dt.strftime('%d %b'),y=prediction['predicted_units'],mode='lines+markers',line=dict(color='#9f86c8',width=3),marker=dict(size=7,color='#d96b78')))
            plot.update_layout(height=340,margin=dict(l=8,r=15,t=15,b=20),yaxis_title='Units per day',showlegend=False,hovermode='x unified')
            plot.update_xaxes(type='category',categoryorder='array',categoryarray=prediction['date'].dt.strftime('%d %b').tolist(),showgrid=False,title_text='Forecast date')
            plot.update_yaxes(rangemode='tozero',gridcolor='rgba(120,120,120,.13)')
            plot=plot_style(plot)
            st.plotly_chart(plot,use_container_width=True)
            st.dataframe(light_style(prediction.assign(date=prediction['date'].dt.strftime('%Y-%m-%d'))[['date','predicted_units']], '#f1ebf8'),hide_index=True,use_container_width=True)
            st.markdown(f'<div class="status-box status-info"><strong>Operational recommendation:</strong> Review adding <strong>{suggested:,} units</strong> for {product} on {route}, based on tomorrow’s estimated demand and current stock. Confirm shelf life, capacity and existing orders before approval.</div>', unsafe_allow_html=True)
            st.write(f'**Validation summary:** Trained on **{nrows}** daily observations. Holdout MAE = **{mae:.1f}** units versus simple weekday baseline = **{baseline:.1f}** units.')
            st.caption('MAE measures average absolute forecast error on historical holdout observations. It does not translate directly into business savings.')
            if mae >= baseline:
                st.markdown('<div class="status-box status-amber"><strong>Model caution:</strong> Under the current data and settings, the fitted model does not outperform the simple baseline. Treat this as a prototype and refine before operational use.</div>', unsafe_allow_html=True)
            st.download_button('Download forecast CSV',prediction.to_csv(index=False),file_name='7harvest_demand_forecast.csv',mime='text/csv')
        except Exception as ex:
            st.error(str(ex))

    st.divider()
    st.subheader('SmartFresh — illustrative savings & pilot business case')
    st.caption('This scenario estimates the effect of reducing recorded damage / expiry expense. It is a financial illustration, not evidence of achieved savings.')
    s1,s2,s3=st.columns(3)
    with s1:
        annual_revenue=st.number_input('Assumed annual Soorya sales (AED)',min_value=10000,value=10000000,step=100000)
    with s2:
        target_expiry=st.slider('Target expiry / damage (% of sales)',min_value=0.0,max_value=2.0,value=1.2,step=0.1)
    with s3:
        pilot_cost=st.number_input('Annualised implementation & running cost (AED)',min_value=0,value=35000,step=5000)
    current_expiry_value=annual_revenue*.02
    scenario_expiry_value=annual_revenue*target_expiry/100
    avoided=current_expiry_value-scenario_expiry_value
    net=avoided-pilot_cost
    v1,v2,v3,v4=st.columns(4)
    v1.metric('Current annual loss',f'AED {current_expiry_value:,.0f}')
    v2.metric('Target annual loss',f'AED {scenario_expiry_value:,.0f}')
    v3.metric('Potential avoided loss',f'AED {avoided:,.0f}')
    v4.metric('Net annual scenario benefit',f'AED {net:,.0f}')
    st.progress(min(1,max(0,(2-target_expiry)/2)),text=f'Illustrative reduction in recorded damage / expiry expense: {(2-target_expiry)/2*100:.0f}%')
    if net >= 0:
        st.markdown(f'<div class="status-box status-green"><strong>Scenario outcome:</strong> Under these assumptions, the illustrative annual benefit is AED {net:,.0f} after annualised implementation costs. Validate against real write-offs, stockouts and pilot results before using externally.</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="status-box status-red"><strong>Scenario outcome:</strong> Under these assumptions, estimated costs exceed avoided damage / expiry expense by AED {-net:,.0f}. The pilot would need either better waste reduction or lower implementation cost to justify rollout.</div>', unsafe_allow_html=True)
    pilot=pd.DataFrame({'Metric':['Assumed annual sales AED','Current damage / expiry %','Target damage / expiry %','Estimated avoided loss AED','Annualised project cost AED','Net scenario benefit AED'], 'Value':[annual_revenue,2.0,target_expiry,avoided,pilot_cost,net]})
    st.download_button('Download savings scenario CSV',pilot.to_csv(index=False),file_name='7harvest_smartfresh_savings.csv',mime='text/csv')
    st.caption('The forecasting model uses real machine-learning fitting, but the built-in demonstration dataset is synthetic.')

with tabs[2]:
    st.subheader('Commercial scenario simulator')
    st.caption('Use these controls to test how contribution changes under different cost and channel assumptions.')
    c1,c2=st.columns(2)
    with c1:
        st.markdown('#### RK Foods')
        rk_cogs=st.slider('RK cost-of-goods change (%)',-10.0,15.0,0.0,.5)
        rk_trade=st.slider('Trade cost reduction (percentage points)',0.0,5.0,1.0,.25)
        rk_margin=25.5-62.5*rk_cogs/100+rk_trade
        st.metric('Projected contribution before overheads',f'{rk_margin:.2f}%',f'{rk_margin-25.5:+.2f} pp')
    with c2:
        st.markdown('#### Sooryya Foods' if False else '#### Soorya Foods')
        s_cogs=st.slider('Soorya cost-of-goods change (%)',-10.0,15.0,0.0,.5)
        s_expiry=st.slider('Expiry / damage expense (% of sales)',0.0,4.0,1.2,.1)
        s_margin=45.2-40*s_cogs/100+(2-s_expiry)
        st.metric('Projected contribution before overheads',f'{s_margin:.2f}%',f'{s_margin-45.2:+.2f} pp')

    st.divider()
    st.subheader('7Harvest Foodservice — channel comparison')
    st.caption('Illustrative per-AED-100 comparison. Price differences, customer credit risk, route efficiency and cannibalisation would need validation with actual account data.')
    a,b,c,d=st.columns(4)
    discount=a.number_input('B2B discount %',0.,50.,3.,.5)
    fulfil=b.number_input('B2B fulfilment %',0.,50.,4.,.5)
    credit=c.number_input('B2B credit / admin %',0.,50.,1.5,.5)
    trade=d.number_input('Retail trade spend %',0.,50.,10.26,.25)
    retail=100-62.5-1.75-trade
    foodservice=100-62.5-discount-fulfil-credit
    x,y,z=st.columns(3)
    x.metric('Illustrative retail contribution',f'{retail:.2f}%')
    y.metric('Illustrative B2B contribution',f'{foodservice:.2f}%')
    z.metric('Difference',f'{foodservice-retail:+.2f} pp')

    st.divider()
    st.subheader('RK Kitchen — portfolio review')
    st.caption('Illustrative SKU review logic for Objective 1. This section shows how 7Harvest could identify products to retain, scale or rationalise before launching value-added lines.')
    sku=sample_sku_review()
    st.dataframe(light_style(sku, '#f8e8ec'), hide_index=True, use_container_width=True)
    k1,k2,k3=st.columns(3)
    k1.metric('SKUs flagged for review', str((sku['Recommendation']=='Review / consolidate').sum()))
    k2.metric('Keep / scale / assess', str((sku['Recommendation']!='Review / consolidate').sum()))
    k3.metric('Max variant cluster', f"{sku['Variants in cluster'].max()} variants")
    bubble=px.scatter(sku, x='Annual sales AED', y='Contribution %', size='Variants in cluster', color='Recommendation',
                      hover_name='SKU',
                      color_discrete_map={'Review / consolidate':'#d96b78','Keep / scale':'#79a7d8','Keep':'#9f86c8','Scale as value-added':'#c7a65a'})
    bubble.update_layout(height=360, legend_title_text='')
    bubble = plot_style(bubble)
    st.plotly_chart(bubble, use_container_width=True)
    st.markdown('<div class="status-box status-info"><strong>Portfolio implication:</strong> High-variant, low-contribution clusters can be reviewed for simplification, while high-contribution mixes and giftable / premium products can support the RK Kitchen value proposition.</div>', unsafe_allow_html=True)

with tabs[3]:
    st.subheader('Retail trade investment review')
    st.write('Enter estimated **incremental contribution before the specific investment** and the related trade cost for each outlet. These results are arithmetic scenarios, not proof of causal uplift.')
    default=pd.DataFrame({'Outlet':['Outlet A','Outlet B','Outlet C','Outlet D'],'Incremental contribution AED':[18500.,5200.,15800.,7400.],'Trade investment AED':[12000.,8000.,10500.,6500.]})
    edits=st.data_editor(default,num_rows='dynamic',use_container_width=True,key='retail_edit')
    work=edits.copy()
    for col in ['Incremental contribution AED','Trade investment AED']:
        work[col]=pd.to_numeric(work[col],errors='coerce').fillna(0)
    work['Net incremental contribution AED']=work['Incremental contribution AED']-work['Trade investment AED']
    work['Investment recovery ratio']=np.where(work['Trade investment AED']>0,work['Incremental contribution AED']/work['Trade investment AED'],np.nan)
    st.dataframe(light_style(work, '#f1ebf8'), hide_index=True, use_container_width=True)
    a,b,c=st.columns(3)
    a.metric('Trade investment',f"AED {work['Trade investment AED'].sum():,.0f}")
    b.metric('Net contribution',f"AED {work['Net incremental contribution AED'].sum():,.0f}")
    c.metric('Outlets below breakeven',str((work['Net incremental contribution AED']<0).sum()))

    bar=px.bar(work, x='Outlet', y='Net incremental contribution AED', color='Net incremental contribution AED',
               color_continuous_scale=['#d96b78','#f3b2ba','#c9b6dc','#79a7d8'], title='Outlet-level net contribution view')
    bar.update_layout(height=320, coloraxis_showscale=False)
    bar = plot_style(bar)
    st.plotly_chart(bar, use_container_width=True)

    bad=work[work['Net incremental contribution AED']<0]
    if len(bad):
        st.markdown('<div class="status-box status-amber"><strong>Commercial action:</strong> Review or renegotiate underperforming outlets: ' + ', '.join(bad.Outlet.astype(str)) + '.</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="status-box status-green"><strong>Commercial action:</strong> Every listed outlet clears its trade cost under the current assumptions.</div>', unsafe_allow_html=True)
    st.download_button('Download retail scenario',work.to_csv(index=False),'7harvest_trade_roi.csv','text/csv')

with tabs[4]:
    st.subheader('Stock availability and expiry review')
    st.caption('Edit quantities below. Simple coverage rules highlight likely understock and short-shelf-life overstock. These are operating rules, not model predictions.')
    st.markdown('<span class="small-label">If you run a forecast in the Demand planning tab, the relevant product\'s expected daily units and current stock will flow into this table automatically for a more connected demo.</span>', unsafe_allow_html=True)

    edited=st.data_editor(st.session_state.inventory_defaults,use_container_width=True,num_rows='dynamic',key='inventory_edit')
    inv=edited.copy()
    numeric=['Saleable stock units','Expected units per day','Days remaining shelf life']
    for col in numeric:
        inv[col]=pd.to_numeric(inv[col],errors='coerce').fillna(0).clip(lower=0)
    inv['Days of coverage']=np.where(inv['Expected units per day']>0,inv['Saleable stock units']/inv['Expected units per day'],np.nan)
    inv['Alert']=np.select([
        inv['Days of coverage']<2,
        inv['Days of coverage']>inv['Days remaining shelf life']
    ],['Replenish soon','Expiry risk'],default='Healthy')
    st.dataframe(light_style(inv, '#edf3fb'), hide_index=True, use_container_width=True)

    if st.session_state.latest_forecast_context is not None:
        ctx=st.session_state.latest_forecast_context
        st.markdown(f'<div class="status-box status-info"><strong>Linked from Demand planning:</strong> latest forecast used {ctx["product"]} on {ctx["route"]}, with tomorrow forecast {ctx["tomorrow_forecast"]} units and suggested additional load {ctx["suggested_additional_load"]} units.</div>', unsafe_allow_html=True)

    flags=inv.copy()
    if len(flags)==0:
        st.markdown('<div class="status-box status-green">No immediate inventory flags under the current assumptions.</div>', unsafe_allow_html=True)
    else:
        for _,r in flags.iterrows():
            if r['Alert']=='Replenish soon':
                st.markdown(f'<div class="status-box status-red"><strong>{r["Product"]}</strong> — Replenish soon: approx. {r["Days of coverage"]:.1f} days of supply. Review actual stock age and demand before acting.</div>', unsafe_allow_html=True)
            elif r['Alert']=='Expiry risk':
                st.markdown(f'<div class="status-box status-amber"><strong>{r["Product"]}</strong> — Expiry risk: approx. {r["Days of coverage"]:.1f} days of supply. Review actual stock age and demand before acting.</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="status-box status-green"><strong>{r["Product"]}</strong> — Healthy coverage: approx. {r["Days of coverage"]:.1f} days of supply.</div>', unsafe_allow_html=True)
    st.download_button('Download inventory plan',inv.to_csv(index=False),'7harvest_inventory_plan.csv','text/csv')

st.divider()
st.caption('Prototype scope: one forecasting model plus transparent commercial and inventory decision rules. Not connected to 7Harvest ERP, retailer POS or confidential live data. Brand cost data source: Seven Harvest Business Context (FMCG Consulting Challenge 2026), pp. 6–7.')
