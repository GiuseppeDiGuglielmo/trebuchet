import pandas as pd
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

from mpl_toolkits.axes_grid1.inset_locator import inset_axes

import ipywidgets as widgets
from IPython.display import display, clear_output

def plot_area_latency_vs_reusefactor(data, inputs, outputs, strategy, iotype):
    selected_rows = data[(data['Inputs'] == inputs) & (data['Outputs'] == outputs) & (data['Strategy'] == strategy)  & (data['IOType'] == iotype)]

    fig, ax1 = plt.subplots(figsize=(12, 6))
    
    color_area = 'tab:blue'
    ax1.set_xlabel('Reuse Factor')
    ax1.set_ylabel('Area', color=color_area)
    line1 = ax1.plot(selected_rows['ReuseFactor'], selected_rows['AreaHLS'], marker='o', color=color_area, linestyle='--', label='Area (HLS)')
    line2 = ax1.plot(selected_rows['ReuseFactor'], selected_rows['AreaSYN'], marker='o', color=color_area, label='Area (SYN)')
    ax1.tick_params(axis='y', labelcolor=color_area)
    
    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    color_latency = 'tab:red'
    ax2.set_ylabel('Latency', color=color_latency)
    line3 = ax2.plot(selected_rows['ReuseFactor'], selected_rows['LatencyHLS'], marker='o', color=color_latency, linestyle='-', label='Latency')
    ax2.tick_params(axis='y', labelcolor=color_latency)
    
    # added a legend
    lines = line1 + line2 + line3
    labels = [l.get_label() for l in lines]
    plt.legend(lines, labels, loc=0)
    
    plt.title(f'Area and Latency vs. Reuse Factor, {inputs}x{outputs} ({strategy}, {iotype})')

    # Create an inset axes for the logo
    ax_logo = inset_axes(ax1, width='10%', height='10%', loc='lower right', borderpad=1)
    logo = mpimg.imread('hls4ml_logo.png')
    ax_logo.imshow(logo)
    ax_logo.axis('off')  # Hide the axes ticks and spines
    
    plt.show()


# Plotting with Latency on the X-axis and Area on the Y-axis
def plot_area_vs_latency(data, inputs, outputs, strategy, iotype):
    selected_rows = data[(data['Inputs'] == inputs) & (data['Outputs'] == outputs) & (data['Strategy'] == strategy) & (data['IOType'] == iotype)]

    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plotting Area vs. Latency
    line1 = ax.plot(selected_rows['LatencyHLS'], selected_rows['AreaHLS'], marker='o', linestyle='--', color='tab:green', label='Area (HLS)')
    line2 = ax.plot(selected_rows['LatencyHLS'], selected_rows['AreaSYN'], marker='o', linestyle='-', color='tab:blue', label='Area (SYN)')
    
    ax.set_xlabel('Latency')
    ax.set_ylabel('Area')
    ax.set_title(f'Area vs. Latency, {inputs}x{outputs} ({strategy}, {iotype})')
    
    # Adding a legend
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax.legend(lines, labels, loc='best')

    # Creating an inset axes for the logo
    ax_logo = inset_axes(ax, width='10%', height='10%', loc='lower right', borderpad=3)
    logo = mpimg.imread('hls4ml_logo.png')
    ax_logo.imshow(logo)
    ax_logo.axis('off')  # Hide the axes ticks and spines

    ax.grid(True)
    plt.show()

# Show interactive table
def display_interactive_table(data):
    # Create widgets for categorical columns
    layer_dropdown = widgets.Dropdown(options=pd.unique(data['Layer']), description='Layer:')
    iotype_dropdown = widgets.Dropdown(options=pd.unique(data['IOType']), description='IOType:')
    strategy_dropdown = widgets.Dropdown(options=pd.unique(data['Strategy']), description='Strategy:')
    precision_dropdown = widgets.Dropdown(options=pd.unique(data['Precision']), description='Precision:')
    
    # Create range sliders for numerical columns
    inputs_slider = widgets.IntRangeSlider(
        value=[data['Inputs'].min(), data['Inputs'].max()],
        min=data['Inputs'].min(),
        max=data['Inputs'].max(),
        step=1,
        description='Inputs Range:'
    )
    
    outputs_slider = widgets.IntRangeSlider(
        value=[data['Outputs'].min(), data['Outputs'].max()],
        min=data['Outputs'].min(),
        max=data['Outputs'].max(),
        step=1,
        description='Outputs Range:'
    )
    
    reuse_factor_slider = widgets.IntRangeSlider(
        value=[data['ReuseFactor'].min(), data['ReuseFactor'].max()],
        min=data['ReuseFactor'].min(),
        max=data['ReuseFactor'].max(),
        step=1,
        description='Reuse Factor Range:'
    )
    
    # Function to filter data based on selections
    def filter_data(layer, iotype, strategy, precision, inputs_range, outputs_range, reuse_factor_range):
        filtered_data = data[(data['Layer'] == layer) & 
                           (data['IOType'] == iotype) & 
                           (data['Strategy'] == strategy) &
                           (data['Precision'] == precision) &
                           (data['Inputs'] >= inputs_range[0]) & (data['Inputs'] <= inputs_range[1]) &
                           (data['Outputs'] >= outputs_range[0]) & (data['Outputs'] <= outputs_range[1]) &
                           (data['ReuseFactor'] >= reuse_factor_range[0]) & (data['ReuseFactor'] <= reuse_factor_range[1])]
        clear_output(wait=True)
        display(filtered_data)
    
    # Interactive output linking the widgets and the display function
    out = widgets.interactive_output(filter_data, {
        'layer': layer_dropdown,
        'iotype': iotype_dropdown,
        'strategy': strategy_dropdown,
        'precision': precision_dropdown,
        'inputs_range': inputs_slider,
        'outputs_range': outputs_slider,
        'reuse_factor_range': reuse_factor_slider
    })
    
    # Display the widgets and the output
    display(layer_dropdown, iotype_dropdown, strategy_dropdown, precision_dropdown, inputs_slider, outputs_slider, reuse_factor_slider, out)