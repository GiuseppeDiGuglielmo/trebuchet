import pandas as pd
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

from matplotlib.ticker import MaxNLocator

from adjustText import adjust_text

from mpl_toolkits.axes_grid1.inset_locator import inset_axes

import ipywidgets as widgets
from IPython.display import display, clear_output

def plot_area_latency_vs_reusefactor(data, inputs, outputs, strategy, iotype, show_area_hls=True, show_area_syn=True, show_latency=True, show_rf1=True):
    selected_rows = data[(data['Inputs'] == inputs) & (data['Outputs'] == outputs) & (data['Strategy'] == strategy)  & (data['IOType'] == iotype)]

    # Drop rows whose RF is 1
    if not show_rf1:
        print('ATTENTION: Results for ReuseFactor==1 are dropped!')
        selected_rows = selected_rows[selected_rows['ReuseFactor'] != 1]
    
    fig, ax1 = plt.subplots(figsize=(12, 6))
    
    color_area = 'tab:blue'
    ax1.set_xlabel('Reuse Factor')
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
    if show_area_hls or show_area_syn:
        ax1.set_ylabel('Area', color=color_area)

    if show_area_hls:
        line1 = ax1.plot(selected_rows['ReuseFactor'], selected_rows['AreaHLS'], marker='x', color=color_area, linestyle='--', label='Area (HLS)')
    if show_area_syn:
        line2 = ax1.plot(selected_rows['ReuseFactor'], selected_rows['AreaSYN'], marker='o', color=color_area, label='Area (SYN)')

    if show_area_hls or show_area_syn:
        ax1.tick_params(axis='y', labelcolor=color_area)
    else:
        ax1.set_yticks([])
    
    if show_latency:
        ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
        color_latency = 'tab:red'
        ax2.set_ylabel('Latency', color=color_latency)
        ax2.yaxis.set_major_locator(MaxNLocator(integer=True))
        line3 = ax2.plot(selected_rows['ReuseFactor'], selected_rows['LatencyHLS'], marker='o', color=color_latency, linestyle='-', label='Latency')
        ax2.tick_params(axis='y', labelcolor=color_latency)
    
    # added a legend
    lines = []
    if show_area_hls:
        lines = lines + line1
    if show_area_syn:
        lines = lines + line2
    if show_latency:
        lines = lines + line3
    labels = [l.get_label() for l in lines]
    plt.legend(lines, labels, loc=0)
    
    plt.title(f'Area and Latency vs. Reuse Factor, {inputs}x{outputs} ({strategy}, {iotype})')

    # Create an inset axes for the logo
    ax_logo = inset_axes(ax1, width='10%', height='10%', loc='lower right', borderpad=1)
    logo = mpimg.imread('hls4ml_logo.png')
    ax_logo.imshow(logo)
    ax_logo.axis('off')  # Hide the axes ticks and spines
    
    plt.show()


def plot_interactive_area_latency_vs_reusefactor(data):
    # Create widgets for the inputs
    inputs_widget = widgets.Dropdown(options=sorted(data['Inputs'].unique()), description='Inputs')
    outputs_widget = widgets.Dropdown(options=sorted(data['Outputs'].unique()), description='Outputs')
    strategy_widget = widgets.Dropdown(options=sorted(data['Strategy'].unique()), description='Strategy')
    iotype_widget = widgets.Dropdown(options=sorted(data['IOType'].unique()), description='IOType')
    show_area_hls_widget = widgets.Checkbox(value=True, description='Show HLS Area')
    show_area_syn_widget = widgets.Checkbox(value=True, description='Show SYN Area')
    show_latency_widget = widgets.Checkbox(value=True, description='Show Latency')
    show_rf1_widget = widgets.Checkbox(value=True, description='Show RF=1')

    # Organize widgets in a grid layout
    grid = widgets.GridBox(
        children=[
            inputs_widget, outputs_widget,
            strategy_widget, iotype_widget,
            show_area_hls_widget, show_area_syn_widget,
            show_latency_widget,
            show_rf1_widget
        ],
        layout=widgets.Layout(
            grid_template_rows='auto auto auto',  # Specifies the rows size
            grid_template_columns='50% 50%',      # Specifies the columns size
            grid_gap='10px 10px'                  # Space between items
        )
    )

    # Combine the widgets and the function into an interactive widget
    interactive_plot = widgets.interactive_output(
        plot_area_latency_vs_reusefactor,
        {
            'data': widgets.fixed(data),
            'inputs': inputs_widget,
            'outputs': outputs_widget,
            'strategy': strategy_widget,
            'iotype': iotype_widget,
            'show_area_hls': show_area_hls_widget,
            'show_area_syn': show_area_syn_widget,
            'show_latency': show_latency_widget,
            'show_rf1': show_rf1_widget
        }
    )

    # Display the interactive plot
    display(grid, interactive_plot)


def plot_multiple_area_latency_vs_reusefactor(data, quadrant_pairs, strategy, iotype, vertical=True, show_area_hls=True, show_area_syn=True, show_latency=True, show_rf1=True):
    num_plots = len(quadrant_pairs)

    # Determine the layout of subplots based on orientation
    if vertical:  # 'vertical'
        # Desired width and height per subplot
        subplot_width = 8
        subplot_height = 4

        # Calculate total figure dimensions
        fig_width = subplot_width  # Since ncols = 1, the width is the same as for one subplot
        fig_height = subplot_height * num_plots  # Total height is the height per subplot times the number of plots

        nrows, ncols = num_plots, 1
        fig, axes = plt.subplots(nrows, ncols, figsize=(fig_width, fig_height))
    else:
        nrows, ncols = 1, num_plots
        fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 5 * nrows))

    # Flatten the axes array if there are multiple subplots
    if num_plots > 1:
        axes = axes.flatten()
    else:
        axes = [axes]
        
    color_area = 'tab:blue'
    color_latency = 'tab:red'

    for i, (inputs, outputs) in enumerate(quadrant_pairs):
        ax = axes[i]
        # Filter data for each specific input-output pair
        selected_rows = data[(data['Inputs'] == inputs) & (data['Outputs'] == outputs) & (data['Strategy'] == strategy) & (data['IOType'] == iotype)]

        # Conditionally exclude data with RF=1
        if not show_rf1:
            selected_rows = selected_rows[selected_rows['ReuseFactor'] != 1]

        # Setup the area plot
        ax.set_xlabel('Reuse Factor')
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        if show_area_hls or show_area_syn:
            ax.set_ylabel('Area')
        else:
            ax.set_yticks([])
            
        if show_area_hls:
            ax.plot(selected_rows['ReuseFactor'], selected_rows['AreaHLS'], marker='x', color=color_area, linestyle='--', label=f'Area HLS ({inputs}x{outputs})')
        if show_area_syn:
            ax.plot(selected_rows['ReuseFactor'], selected_rows['AreaSYN'], marker='o', color=color_area, label=f'Area SYN ({inputs}x{outputs})')
        
        # Setup the latency plot on the same axis
        if show_latency:
            ax2 = ax.twinx()
            ax2.plot(selected_rows['ReuseFactor'], selected_rows['LatencyHLS'], marker='o', color=color_latency, linestyle='-', label='Latency HLS')
            ax2.set_ylabel('Latency')
            ax2.yaxis.set_major_locator(MaxNLocator(integer=True))
        
        # Set plot titles and legends
        ax.set_title(f'{inputs}x{outputs} ({strategy}, {iotype})')
        if show_area_hls or show_area_syn:
            lines, labels = ax.get_legend_handles_labels()
        else:
            lines = []
            labels = []
        if show_latency:
            lines2, labels2 = ax2.get_legend_handles_labels()
        else:
            lines2 = []
            labels2 = []
        ax.legend(lines + lines2, labels + labels2, loc='upper left')

    plt.tight_layout()
    plt.show()


# +
def is_pareto_optimal(costs):
    is_efficient = np.ones(costs.shape[0], dtype=bool)
    for i, c in enumerate(costs):
        if is_efficient[i]:
            is_efficient[is_efficient] = np.any(costs[is_efficient] < c, axis=1)  # Keep any point with a lower cost
            is_efficient[i] = True  # And keep self
    return is_efficient

# Plotting with Latency on the X-axis and Area on the Y-axis
def plot_area_vs_latency(data, inputs, outputs, strategy, iotype, show_area_hls=True, show_area_syn=True, show_rf1=True, show_rf=True, show_pareto=True):
    selected_rows = data[(data['Inputs'] == inputs) & (data['Outputs'] == outputs) & (data['Strategy'] == strategy) & (data['IOType'] == iotype)]
    
    # Drop rows whose RF is 1
    if not show_rf1:
        selected_rows = selected_rows[selected_rows['ReuseFactor'] != 1]
    
    # Get pareto points
    pareto_mask_hls = np.array([False] * len(selected_rows))
    pareto_mask_syn = np.array([False] * len(selected_rows))
    if show_pareto:
        costs_hls = np.vstack((selected_rows['LatencyHLS'], selected_rows['AreaHLS'])).T
        pareto_mask_hls = is_pareto_optimal(costs_hls)
        costs_syn = np.vstack((selected_rows['LatencyHLS'], selected_rows['AreaSYN'])).T
        pareto_mask_syn = is_pareto_optimal(costs_syn)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plotting Area vs. Latency
    if show_area_hls:
        line1 = ax.plot(selected_rows['LatencyHLS'], selected_rows['AreaHLS'], marker='x', linestyle='--', color='tab:green', label='Area (HLS)')
        # Higlight Pareto points
        for latency, area, pareto in zip(selected_rows['LatencyHLS'], selected_rows['AreaHLS'], pareto_mask_hls):
            ax.plot(latency, area, marker='x', color='tab:orange' if pareto else 'tab:green', markersize=6)
    
    if show_area_syn:
        line2 = ax.plot(selected_rows['LatencyHLS'], selected_rows['AreaSYN'], marker='o', linestyle='-', color='tab:green', label='Area (SYN)')
        # Higlight Pareto points
        for latency, area, pareto in zip(selected_rows['LatencyHLS'], selected_rows['AreaSYN'], pareto_mask_syn):
            ax.plot(latency, area, marker='o', color='tab:red' if pareto else 'tab:green', markersize=6)
    
    
    ax.set_xlabel('Latency')
    ax.set_ylabel('Area')
    ax.set_title(f'Area vs. Latency, {inputs}x{outputs} ({strategy}, {iotype})')

    # List to hold the text objects for adjustment
    texts = []
    
    # Annotate each point with the ReuseFactor
    if show_rf:
        for idx, row in selected_rows.iterrows():
            if show_area_hls:
                texts.append(ax.text(row['LatencyHLS'], row['AreaHLS'], f'{row["ReuseFactor"]}', fontsize=10, ha='right'))
            if show_area_syn:
                texts.append(ax.text(row['LatencyHLS'], row['AreaSYN'], f'{row["ReuseFactor"]}', fontsize=10, ha='left'))

    # Adding a legend
    lines = []
    if show_area_hls:
        lines = lines + line1
    if show_area_syn:
        lines = lines + line2
    labels = [l.get_label() for l in lines]
    ax.legend(lines, labels, loc='best')

    ax.grid(True)

    # Adjust text to minimize overlaps
    adjust_text(texts, arrowprops=dict(arrowstyle='->', color='red'))
    # adjust_text(texts, 
    #             ax=ax, 
    #             expand_points=(1.2, 1.5), 
    #             expand_text=(1.1, 1.2), 
    #             force_points=(0.5, 0.9), 
    #             force_text=(0.5, 0.9), 
    #             arrowprops=dict(arrowstyle='->', color='red'))

    # Creating an inset axes for the logo
    ax_logo = inset_axes(ax, width='10%', height='10%', loc='lower right', borderpad=3)
    logo = mpimg.imread('hls4ml_logo.png')
    ax_logo.imshow(logo)
    ax_logo.axis('off')  # Hide the axes ticks and spines
    
    plt.show()


# -

def plot_interactive_area_vs_latency(data):
    # Create widgets for user inputs
    inputs_widget = widgets.Dropdown(options=sorted(data['Inputs'].unique()), description='Inputs')
    outputs_widget = widgets.Dropdown(options=sorted(data['Outputs'].unique()), description='Outputs')
    strategy_widget = widgets.Dropdown(options=sorted(data['Strategy'].unique()), description='Strategy')
    iotype_widget = widgets.Dropdown(options=sorted(data['IOType'].unique()), description='IOType')
    show_area_hls_widget = widgets.Checkbox(value=True, description='Show HLS Area')
    show_area_syn_widget = widgets.Checkbox(value=True, description='Show SYN Area')
    show_rf1_widget = widgets.Checkbox(value=True, description='Show RF=1')
    show_pareto_widget = widgets.Checkbox(value=False, description='Show Pareto Points')

    # Organize widgets in a grid
    grid = widgets.GridBox(children=[
        inputs_widget, outputs_widget,
        strategy_widget, iotype_widget,
        show_area_hls_widget, show_area_syn_widget,
        show_rf1_widget,
        show_pareto_widget
    ], layout=widgets.Layout(
        grid_template_rows='auto auto auto',  # Specifies the rows size
        grid_template_columns='50% 50%',      # Specifies the columns size
        grid_gap='10px 10px'                  # Space between items
    ))

    # Link the widgets to the plotting function
    interactive_plot = widgets.interactive_output(
        plot_area_vs_latency,
        {
            'data': widgets.fixed(data),
            'inputs': inputs_widget,
            'outputs': outputs_widget,
            'strategy': strategy_widget,
            'iotype': iotype_widget,
            'show_area_hls': show_area_hls_widget,
            'show_area_syn': show_area_syn_widget,
            'show_rf1': show_rf1_widget,
            'show_pareto': show_pareto_widget
        }
    )

    # Display the interactive controls and the plot
    display(grid, interactive_plot)


def display_interactive_table(data):
    # Create widgets for categorical columns with multiple selections
    layer_widget = widgets.SelectMultiple(
        options=pd.unique(data['Layer']),
        value=[pd.unique(data['Layer'])[0]],  # Default to the first available option
        description='Layer:',
        disabled=False
    )
    iotype_widget = widgets.SelectMultiple(
        options=pd.unique(data['IOType']),
        value=[pd.unique(data['IOType'])[0]],  # Default to the first available option
        description='IOType:',
        disabled=False
    )
    strategy_widget = widgets.SelectMultiple(
        options=pd.unique(data['Strategy']),
        value=[pd.unique(data['Strategy'])[0]],  # Default to the first available option
        description='Strategy:',
        disabled=False
    )
    precision_widget = widgets.SelectMultiple(
        options=pd.unique(data['Precision']),
        value=[pd.unique(data['Precision'])[0]],  # Default to the first available option
        description='Precision:',
        disabled=False
    )
    
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
    def filter_data(layers, iotypes, strategies, precisions, inputs_range, outputs_range, reuse_factor_range):
        filtered_data = data[
            (data['Layer'].isin(layers)) & 
            (data['IOType'].isin(iotypes)) & 
            (data['Strategy'].isin(strategies)) &
            (data['Precision'].isin(precisions)) &
            (data['Inputs'] >= inputs_range[0]) & (data['Inputs'] <= inputs_range[1]) &
            (data['Outputs'] >= outputs_range[0]) & (data['Outputs'] <= outputs_range[1]) &
            (data['ReuseFactor'] >= reuse_factor_range[0]) & (data['ReuseFactor'] <= reuse_factor_range[1])
        ]
        clear_output(wait=True)
        display(filtered_data)
    
    # Arrange widgets in a grid
    grid = widgets.GridBox(children=[
        layer_widget, iotype_widget, 
        strategy_widget, precision_widget, 
        inputs_slider, outputs_slider, 
        reuse_factor_slider
    ], layout=widgets.Layout(grid_template_columns="repeat(2, 1fr)"))
    
    # Interactive output linking the widgets and the display function
    out = widgets.interactive_output(filter_data, {
        'layers': layer_widget,
        'iotypes': iotype_widget,
        'strategies': strategy_widget,
        'precisions': precision_widget,
        'inputs_range': inputs_slider,
        'outputs_range': outputs_slider,
        'reuse_factor_range': reuse_factor_slider
    })
    
    # Display the grid and the output
    display(grid, out)


def plot_multiple_area_vs_latency(data, quadrant_pairs, strategy, iotype, vertical=True, show_area_hls=True, show_area_syn=True, show_rf1=True):
    num_plots = len(quadrant_pairs)

    # Determine the layout of subplots based on orientation
    if vertical:
        # Desired width and height per subplot
        subplot_width = 8
        subplot_height = 4

        # Calculate total figure dimensions
        fig_width = subplot_width  # Since ncols = 1, the width is the same as for one subplot
        fig_height = subplot_height * num_plots  # Total height is the height per subplot times the number of plots

        nrows, ncols = num_plots, 1
        fig, axes = plt.subplots(nrows, ncols, figsize=(fig_width, fig_height))
    else:
        nrows, ncols = 1, num_plots
        fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 5 * nrows))

    # Create a figure with subplots based on the orientation
    if num_plots > 1:
        axes = axes.flatten()
    else:
        axes = [axes]

    for i, (inputs, outputs) in enumerate(quadrant_pairs):
        ax = axes[i]
        selected_rows = data[(data['Inputs'] == inputs) & (data['Outputs'] == outputs) & (data['Strategy'] == strategy) & (data['IOType'] == iotype)]
        
        if not show_rf1:
            selected_rows = selected_rows[selected_rows['ReuseFactor'] != 1]
        
        # Plotting Area vs. Latency
        if show_area_hls:
            line1 = ax.plot(selected_rows['LatencyHLS'], selected_rows['AreaHLS'], marker='x', linestyle='--', color='tab:green', label='Area (HLS)')
        if show_area_syn:
            line2 = ax.plot(selected_rows['LatencyHLS'], selected_rows['AreaSYN'], marker='o', linestyle='-', color='tab:green', label='Area (SYN)')
        
        ax.set_xlabel('Latency')
        ax.set_ylabel('Area')
        ax.set_title(f'Area vs. Latency, {inputs}x{outputs} ({strategy}, {iotype})')

        # Annotate each point with the ReuseFactor
        # texts = []
        # for idx, row in selected_rows.iterrows():
        #     if show_area_hls:
        #         texts.append(ax.text(row['LatencyHLS'], row['AreaHLS'], f'{row["ReuseFactor"]}', fontsize=10, ha='right'))
        #     if show_area_syn:
        #         texts.append(ax.text(row['LatencyHLS'], row['AreaSYN'], f'{row["ReuseFactor"]}', fontsize=10, ha='left'))

        # Adding a legend
        lines = []
        if show_area_hls:
            lines = lines + line1
        if show_area_syn:
            lines = lines + line2
        labels = [l.get_label() for l in lines]
        ax.legend(lines, labels, loc='best')

        ax.grid(True)
        
        # Adjust text to minimize overlaps
        # adjust_text(texts, ax=ax, arrowprops=dict(arrowstyle='->', color='red'))
        
    plt.tight_layout()
    plt.show()


def plot_runtime_vs_reusefactor(data, inputs, outputs, strategy, iotype, show_runtime_hls=True, show_runtime_syn=True, show_rf1=True):
    selected_rows = data[(data['Inputs'] == inputs) & (data['Outputs'] == outputs) & (data['Strategy'] == strategy) & (data['IOType'] == iotype)]
    
    # Drop rows whose RF is 1, if specified
    if not show_rf1:
        selected_rows = selected_rows[selected_rows['ReuseFactor'] != 1]
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plotting RuntimeHLS vs. ReuseFactor
    if show_runtime_hls:
        line1 = ax.plot(selected_rows['ReuseFactor'], selected_rows['RuntimeHLS'], marker='x', linestyle='--', color='tab:orange', label='Runtime HLS')
    
    # Plotting RuntimeSYN vs. ReuseFactor
    if show_runtime_syn:
        line2 = ax.plot(selected_rows['ReuseFactor'], selected_rows['RuntimeSYN'], marker='o', linestyle='-', color='tab:orange', label='Runtime SYN')
    
    ax.set_xlabel('Reuse Factor')
    ax.set_ylabel('Runtime')
    ax.set_title(f'Runtime HLS vs SYN Comparison, {inputs}x{outputs} ({strategy}, {iotype})')
    
    # Adding a legend
    lines = []
    if show_runtime_hls:
        lines.append(line1[0])
    if show_runtime_syn:
        lines.append(line2[0])
    ax.legend(lines, [l.get_label() for l in lines], loc='best')
    
    ax.grid(True)
    
    # Creating an inset axes for the logo
    ax_logo = inset_axes(ax, width='10%', height='10%', loc='lower right', borderpad=3)
    logo = mpimg.imread('hls4ml_logo.png')
    ax_logo.imshow(logo)
    ax_logo.axis('off')  # Hide the axes ticks and spines
    
    plt.show()


def plot_interactive_runtime_vs_reusefactor(data):
    # Create widgets for the inputs
    inputs_widget = widgets.Dropdown(options=sorted(data['Inputs'].unique()), description='Inputs')
    outputs_widget = widgets.Dropdown(options=sorted(data['Outputs'].unique()), description='Outputs')
    strategy_widget = widgets.Dropdown(options=sorted(data['Strategy'].unique()), description='Strategy')
    iotype_widget = widgets.Dropdown(options=sorted(data['IOType'].unique()), description='IOType')
    show_runtime_hls_widget = widgets.Checkbox(value=True, description='Show Runtime HLS')
    show_runtime_syn_widget = widgets.Checkbox(value=True, description='Show Runtime SYN')
    show_rf1_widget = widgets.Checkbox(value=True, description='Show RF=1')

    def update_plot(inputs, outputs, strategy, iotype, show_runtime_hls, show_runtime_syn, show_rf1):
        plot_runtime_vs_reusefactor(data, inputs, outputs, strategy, iotype, show_runtime_hls, show_runtime_syn, show_rf1)

    # Organizing dropdowns in a 2x2 grid layout
    grid = widgets.GridBox(children=[
        inputs_widget, outputs_widget,
        strategy_widget, iotype_widget
    ], layout=widgets.Layout(
        grid_template_rows='auto auto auto',  # Specifies the rows size
        grid_template_columns='50% 50%',      # Specifies the columns size
        grid_gap='10px 10px'                  # Space between items
    ))

    # Combine the grid and checkboxes into a VBox
    settings = widgets.VBox([grid, show_runtime_hls_widget, show_runtime_syn_widget, show_rf1_widget])

    # Create an interactive widget
    interactive_plot = widgets.interactive_output(
        update_plot,
        {
            'inputs': inputs_widget,
            'outputs': outputs_widget,
            'strategy': strategy_widget,
            'iotype': iotype_widget,
            'show_runtime_hls': show_runtime_hls_widget,
            'show_runtime_syn': show_runtime_syn_widget,
            'show_rf1': show_rf1_widget
        }
    )

    # Display the interactive plot widget
    display(settings, interactive_plot)


# +
def plot_multiple_runtime_vs_reusefactor(data, quadrant_pairs, strategy, iotype, vertical=True, show_runtime_hls=True, show_runtime_syn=True, show_rf1=True):
    num_plots = len(quadrant_pairs)

    # Determine the layout of subplots based on orientation
    if vertical:
        nrows, ncols = num_plots, 1
        fig, axes = plt.subplots(nrows, ncols, figsize=(8, 4 * num_plots))  # 8 inches wide, 4 inches per plot
    else:
        nrows, ncols = 1, num_plots
        fig, axes = plt.subplots(nrows, ncols, figsize=(8 * num_plots, 6))  # 6 inches tall, 8 inches per plot

    # Make sure axes is an iterable (array) even if there is only one subplot
    if num_plots == 1:
        axes = [axes]

    for i, (inputs, outputs) in enumerate(quadrant_pairs):
        ax = axes[i]
        selected_rows = data[(data['Inputs'] == inputs) & (data['Outputs'] == outputs) & (data['Strategy'] == strategy) & (data['IOType'] == iotype)]
        
        # Drop rows whose RF is 1, if specified
        if not show_rf1:
            selected_rows = selected_rows[selected_rows['ReuseFactor'] != 1]

        # Plotting RuntimeHLS vs. ReuseFactor
        if show_runtime_hls:
            ax.plot(selected_rows['ReuseFactor'], selected_rows['RuntimeHLS'], marker='x', linestyle='--', color='tab:orange', label='Runtime HLS')

        # Plotting RuntimeSYN vs. ReuseFactor
        if show_runtime_syn:
            ax.plot(selected_rows['ReuseFactor'], selected_rows['RuntimeSYN'], marker='o', linestyle='-', color='tab:orange', label='Runtime SYN')

        ax.set_xlabel('Reuse Factor')
        ax.set_ylabel('Runtime')
        ax.set_title(f'Runtime HLS vs SYN, {inputs}x{outputs} ({strategy}, {iotype})')

        ax.legend(loc='best')
        ax.grid(True)

#         # Optional: Add a logo to each subplot
#         ax_logo = inset_axes(ax, width='10%', height='10%', loc='lower right', borderpad=3)
#         logo = mpimg.imread('hls4ml_logo.png')
#         ax_logo.imshow(logo)
#         ax_logo.axis('off')  # Hide the axes ticks and spines

    plt.tight_layout()
    plt.show()
