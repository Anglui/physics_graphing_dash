import base64
import io
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
import dash_extendable_graph as deg
from dash import Dash, dcc, html, Input, Output, State, dash_table, callback_context, no_update
from dash.exceptions import PreventUpdate

# Initialize the Dash app
app = Dash(__name__)
server = app.server

# Fit functions
def linear_func(x, a, b): return a*x + b
def quad_func(x, a, b, c): return a*x**2 + b*x + c
def sqrt_func(x, a, b): return a*np.sqrt(x) + b
def sine_func(x, a, b, c, d): return a*np.sin(b*x + c) + d
def cos_func(x, a, b, c, d): return a*np.cos(b*x + c) + d
def tan_func(x, a, b, c, d): return a*np.tan(b*x + c) + d

FIT_FUNCTIONS = {
    'Linear': (linear_func, ['a','b']),
    'Quadratic': (quad_func, ['a','b','c']),
    'Square Root': (sqrt_func, ['a','b']),
    'Sine': (sine_func, ['a','b','c','d']),
    'Cosine': (cos_func, ['a','b','c','d']),
    'Tangent': (tan_func, ['a','b','c','d']),
}

# Initial scatter figure
INITIAL_FIG = {
    'data': [{'x':[], 'y':[], 'mode':'markers'}],
    'layout': {
        'template':'plotly_white',
        'showlegend':False,
        'xaxis':{'title':'X','showgrid':True,'gridcolor':'lightgrey','linecolor':'white','zerolinecolor':'black'},
        'yaxis':{'title':'Y','showgrid':True,'gridcolor':'lightgrey','linecolor':'white','zerolinecolor':'black'},
        'margin':{'l':40,'r':10,'t':40,'b':40}
    }
}

# Helper: blank row
def blank_row(): 
    return {'x': None, 'y': None}
INITIAL_ROWS = [blank_row()]
INITIAL_COLUMNS = [{'name': 'x', 'id': 'x', 'type': 'numeric'}, {'name': 'y', 'id': 'y', 'type': 'numeric'}]

app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>Mr.Angeles' Lite Physics Modeling</title>
        {%favicon%}
        {%css%}
        <style>
            html, body {
                margin: 0;
                padding: 0;
                height: 100%;
                overflow: hidden;
            }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''

# Layout
app.layout = html.Div(style={'display':'flex', 'flexDirection':'column', 'height':'100vh','boxSizing':'border-box'}, children=[
    html.Div('Mr. Angeles\' Lite Physics Modeling', style={
        'flex': '0 0 5%',
        'textAlign':'left',
        'fontSize':'2rem',
        'fontWeight':'bold',
        'margin':'0',
        'padding':'0.5rem',
        'borderBottom':'2px solid #ccc',
        'background-color':'rgba(120,169,153,0.35)'
    }),

    html.Div(
        style={'flex':'1','display':'flex','fontFamily':'Helvetica, Arial'}, 
        children=[
            dcc.Store(id='prev-count', data=0),
            dcc.Store(id='change-in-rows', data=0),
            
            # Left panel: table and fit controls
            html.Div(style={'width':'33%','padding':'0rem 1rem','borderRight':'1px solid #ccc','display':'flex','flexDirection':'column','background-color':'rgba(85,117,203,0.10)'}, children=[
            
                # Table section
                html.Div(style={'flex':'1', 'overflowY':'auto', 'paddingBottom':'1rem'}, children=[
                    html.H3('Data Entry'),
                    dcc.Upload(id='upload-csv', children=html.Div([html.Strong('📄 Drag & Drop or '), html.A('Select Data', style={'color':'#007bff', 'textDecoration':'underline'})]),
                            style={'width':'95%',
                                   'height':'3rem',
                                   'lineHeight':'3rem',
                                   'borderWidth':'1px',
                                   'borderStyle':'dashed',
                                   'borderRadius':'6px',
                                   'textAlign':'center',
                                   'marginBottom':'1rem',
                                   'cursor':'pointer',
                                   'transition':'background-color 0.2s ease-in-out'}, 
                            multiple=False),
                    html.Div(style={'marginBottom':'1rem'}, children=[
                        html.Button('Clear Data', id='clear-btn', n_clicks=0),
                        html.Button('Add Row', id='add-row', n_clicks=0, style={'marginLeft':'0.5rem'})
                        ]),
                    dash_table.DataTable(id='data-table', 
                                        columns=INITIAL_COLUMNS,
                                        data=INITIAL_ROWS.copy(), 
                                        editable=True, 
                                        row_deletable=True,
                                        style_table={'minWidth':'0','maxWidth':'95%'})
                ]),
                # Fit controls section
                html.Div(style={'flex':'1','paddingTop':'1rem','borderTop':'1px solid #ccc','overflowY':'auto'}, children=[
                    html.H4('Curve Fitting'),
                    dcc.Dropdown(id='fit-method', options=[{'label':'No Fit','value':'None'}] + [{'label':k,'value':k} for k in FIT_FUNCTIONS], value='None'),
                    html.Div(id='fit-output', style={'marginTop':'1rem','whiteSpace':'pre-wrap','fontFamily':'monospace'})
                ])
            ]),
            # Right panel: extendable graph
            html.Div(style={'width':'67%','padding':'0rem 1rem'}, children=[
                    html.H3('Graph'),
                    deg.ExtendableGraph(
                        id='scatter-plot', 
                        figure=INITIAL_FIG, 
                        extendData=None,
                        config={'scrollZoom':True,'doubleClick':'reset','modeBarButtonsToRemove':['zoom2d','toggleSpikelines','hoverCompareCartesian'],}, 
                        style={'height':'80vh'})
            ])
    ])
])

# Callbacks
# Callback: Handle CSV uploads
@app.callback(
    Output('data-table','data'),
    Input('upload-csv','contents'), 
    Input('clear-btn','n_clicks'), 
    Input('add-row','n_clicks'), 
    State('data-table','data')
)
def modify_table(contents, clear_n, add_n, rows):
    ctx = callback_context
    if not ctx.triggered: 
        raise PreventUpdate
    trig = ctx.triggered[0]['prop_id'].split('.')[0]
    
    if trig == 'upload-csv' and contents:
        _,b64=contents.split(',')
        decoded = base64.b64decode(b64)
        df=pd.read_csv(io.BytesIO(decoded))
        if {'x','y'}.issubset(df.columns): 
            df = df[['x','y']]
        else: 
            df = df.iloc[:,:2]
            df.columns=['x','y']
            record_dict = df.to_dict('records')
        return record_dict
    
    if trig == 'clear-btn': 
        return INITIAL_ROWS.copy()
    
    if trig == 'add-row': 
        rows.append(blank_row())
        return rows
    
    return rows


# Callback: Extend plot to stop jitter
@app.callback(
    Output('scatter-plot','extendData'), 
    Output('change-in-rows','data'),
    Output('prev-count','data'),
    Input('data-table','data'), 
    State('prev-count','data')
)
def extend_graph(rows, prev_count):    
    df = pd.DataFrame(rows).dropna()
    current_count = len(df)
    
    # Clear graph
    if current_count == 0:
        return ([{'x': [], 'y': []}], [0], None), current_count - prev_count, 0
    
    # Add data point
    if current_count > prev_count:
        new = df.iloc[prev_count:]
        return ([{'x':new['x'].tolist(),'y':new['y'].tolist()}],[0],None), current_count - prev_count, current_count
    
    # Erase data point
    if current_count < prev_count:
        return ([{'x':df['x'].tolist(),'y':df['y'].tolist()}],[0],None), current_count - prev_count, current_count
    
    return no_update, no_update, no_update


# Callback: Overlay curve on graph
@app.callback(
    Output('scatter-plot','figure'), 
    Output('fit-output','children'), 
    Output('change-in-rows','data', allow_duplicate=True),
    Input('clear-btn', 'n_clicks'),
    Input('fit-method','value'), 
    Input('scatter-plot','extendData'),
    State('data-table','data'),
    State('scatter-plot','figure'),
    State('change-in-rows','data'),
    prevent_initial_call=True
)
def update_fit(clear_n, method, ext_data, data, fig, change_in_rows):
    ctx = callback_context
    if not ctx.triggered: 
        raise PreventUpdate
    trig = ctx.triggered[0]['prop_id'].split('.')[0]
    
    if trig == 'clear-btn':
        return INITIAL_FIG, '', 0
    
    df = pd.DataFrame(data).dropna()
    out=''
    
    if trig == 'scatter-plot' and change_in_rows < 0:
        scatter_trace = {'x': df['x'].tolist(), 'y': df['y'].tolist(), 'mode': 'markers'}
        layout = fig.get('layout', INITIAL_FIG['layout'])
        fig = {
            'data': [scatter_trace],
            'layout': layout
        }
    
    if method in FIT_FUNCTIONS and not df.empty:
        func, params = FIT_FUNCTIONS[method]
        x,y=df['x'].values, df['y'].values
        p0=np.ones(len(params))
        try:
            popt,_ = curve_fit(func, x, y, p0=p0, maxfev=2000)
        except RuntimeError:
            return fig, 'Could not fit curve to data.', 0
        except TypeError:
            return fig, 'Not enough data. Please input at least 5 data points.', 0
        
        # prepare fit line
        xf = np.linspace(x.min(), x.max(),200)
        yf = func(xf, *popt)
        
        # update fig data: keep scatter at idx0, replace/append fit at idx1
        newdata = [fig['data'][0]] + [dict(x=xf.tolist(), y=yf.tolist(), mode='lines', name=f'{method} Fit')]
        fig['data'] = newdata
        
        # compute R2
        ss_res = np.sum((y - func(x, *popt))**2)
        ss_tot = np.sum((y - y.mean())**2)
        r2 = 1-ss_res/ss_tot if ss_tot else np.nan
        
        # format output
        eq = f"{method} fit parameters:\n"+"; ".join([f"{p}={v:.3f}" for p,v in zip(params,popt)])+f"\nR²={r2:.3f}"
        out = eq
    else:
        # remove fit trace if exists
        fig['data'] = [fig['data'][0]]
    return fig, out, 0


if __name__=='__main__':
    app.run(debug=True,port=8050)