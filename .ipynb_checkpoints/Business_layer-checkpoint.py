import math
import numpy as np
import plotly.express as px
import scipy
import dash_bootstrap_components as dbc
from dash_bootstrap_templates import load_figure_template
from data_layer import Data 
import plotly.graph_objects as go
import pandas as pd 
import os 
import plotly.express as px 
import matplotlib.pyplot as plt 
class GraphBuilder:
    """Methods for building Graphs."""

    def __init__(self,data=Data()):
        self.data=data
    def plot_source_geometry (self):
        # Extract the first point as the origin
        df=self.data.read_cable_positions() 
        origin_lat = df.iloc[0]['Latitude']
        origin_lon = df.iloc[0]['Longitude']
        def latlon_to_meters(lat, lon, origin_lat, origin_lon):
            lat_m = (lat - origin_lat) * 111194.9
            lon_m = (lon - origin_lon) * (111194.9 * np.cos(np.radians(origin_lat)))
            return lat_m, lon_m

        # Convert latitude and longitude to meters from the origin

        df['Y'], df['X'] = latlon_to_meters(df['Latitude'], df['Longitude'], origin_lat, origin_lon)

        # Source coordinates
        source_latitude = -35.386486
        source_longitude = 148.980000
        source_y, source_x = latlon_to_meters(source_latitude, source_longitude, origin_lat, origin_lon)

        # Last station coordinates
        last_station_lat = -35.391000
        last_station_lon = 148.980000
        last_station_y, last_station_x = latlon_to_meters(last_station_lat, last_station_lon, origin_lat, origin_lon)

        # Distances from the last station to the sources in meters
        distances = np.array([500])

        # Function to convert distance and angle to new coordinates in meters
        def get_new_coords(x, y, distances, angles):
            new_coords = {}
            for angle in angles:
                angle_rad = np.radians(angle)
                delta_x = distances * np.sin(angle_rad)
                delta_y = distances * np.cos(angle_rad)
                new_x = x + delta_x
                new_y = y + delta_y
                new_coords[angle] = (new_x, new_y)
            return new_coords

        # Angles in degrees
        angles = [22, 45, 64, 90]

        new_coords = get_new_coords(last_station_x, last_station_y, distances, angles)

        # Extracting coordinates
        source_x_22, source_y_22 = new_coords[22]
        source_x_45, source_y_45 = new_coords[45]
        source_x_64, source_y_64 = new_coords[64]
        source_x_90, source_y_90 = new_coords[90]
        # Plotting with Plotly
        fig = go.Figure()

        # Plot channels
        fig.add_trace(go.Scatter(
            x=df['X'], y=df['Y'],
            mode='markers',
            name='channels',
            marker=dict(size=3, color='green')
        ))

        # Plot sources
        fig.add_trace(go.Scatter(
            x=[source_x], y=[source_y],
            mode='markers',
            name='Source 0°',
            marker=dict(size=10, color='red', symbol='star')
        ))
        fig.add_trace(go.Scatter(
            x=source_x_22, y=source_y_22,
            mode='markers',
            name='Source 22°',
            marker=dict(size=10, color='blue', symbol='star')
        ))
        fig.add_trace(go.Scatter(
            x=source_x_45, y=source_y_45,
            mode='markers',
            name='Source 45°',
            marker=dict(size=10, color='yellow', symbol='star')
        ))
        fig.add_trace(go.Scatter(
            x=source_x_64, y=source_y_64,
            mode='markers',
            name='Source 64°',
            marker=dict(size=10, color='cyan', symbol='star')
        ))
        fig.add_trace(go.Scatter(
            x=source_x_90, y=source_y_90,
            mode='markers',
            name='Source 90°',
            marker=dict(size=10, color='magenta', symbol='star')
        ))

        fig.update_layout(xaxis=dict(range=[-1000,1000]))
        fig.layout.title="Geometry"
        fig.layout.xaxis.title="X(M)"
        fig.layout.yaxis.title="Y(M)"
        fig.update_layout(
    autosize=True)
        fig.update_layout(
    margin=dict(l=20, r=20, t=50, b=20)
)

        return fig
    
    
    def plot_vs_vp_profiles(self, model_idx=1):
        velocity_models=self.data.get_velocity_models()
        model = velocity_models[model_idx]
        def prepare_and_plot(vs_array, vp_array, depth_array, color_vs, color_vp):
            vs_array = np.array(vs_array) * 1000
            vp_array = np.array(vp_array) * 1000
            depth_array = np.array(depth_array) * 1000

            # Cumulative depth from provided depths
            depths = np.cumsum(depth_array) #calculate cum sum
            depths = np.insert(depths, 0, 0) #add 0 in the 0 index

            vs_extended = np.repeat(vs_array, 2) #repeat_vs_array #Used in the Plot
            vp_extended = np.repeat(vp_array, 2) #repeat_vp_array #Used in The Plot

            depths_extended = np.empty((depths.size - 1) * 2)
            depths_extended[0::2] = depths[:-1]  # Start of layer depth # used in plot
            depths_extended[1::2] = depths[1:]   # End of layer depth #used in plot 

            # Create a DataFrame for plotting
            df = pd.DataFrame({
                'depths_extended': depths_extended,
                'vs_extended': vs_extended,
                'vp_extended': vp_extended
            })

            # Plotting with Plotly Graph Objects
            fig = go.Figure()

            fig.add_trace(go.Scatter(
                x=df['vs_extended'], y=df['depths_extended'],
                mode='lines', name='Vs Profile', line=dict(color=color_vs)
            ))

            fig.add_trace(go.Scatter(
                x=df['vp_extended'], y=df['depths_extended'],
                mode='lines', name='Vp Profile', line=dict(color=color_vp)
            ))

            fig.update_layout(
                title=f'Velocity Profiles vs. Depth for Model {model_idx}',
                xaxis_title='Velocity (m/s)',
                yaxis_title='Depth (m)',
                yaxis_autorange='reversed',  # Invert y-axis
                xaxis_side='bottom',
                legend=dict(x=0.01, y=0.99),
                margin=dict(l=0, r=0, t=40, b=0),
                showlegend=True,
            )

            fig.update_xaxes(showgrid=True, gridwidth=0.5, gridcolor='LightGrey')
            fig.update_yaxes(showgrid=True, gridwidth=0.5, gridcolor='LightGrey')

            return fig
        fig = prepare_and_plot(model['vs'], model['vp'], model['depth'], 'red', 'blue')
   
        return fig


    def plot_shot_gather_XD(self, model_idx=0, source_idx=0, file_idx=0, slider_time=5, slider_amp=20, dt=0.005, stmin=0.0, tshift=0.0, lnrm=True, b1=0.0, b2=0.0, epic=1.0, eps=0.0, real_color='black', modeled_color='red', ylim=None, velocity_line=None, amp_factor=2.0, amp_offset=0.0):
        model = self.data.get_velocity_models()[model_idx]
        velocities = model['vp'] + model['vs']

        input_file = self.data.get_input_files()[file_idx]
        offset = self.data.get_offsets()[file_idx]

        model_name = f"model_{model_idx}_vp_{'_'.join(map(str, model['vp']))}_vs_{'_'.join(map(str, model['vs']))}_depth_{'_'.join(map(str, model['depth']))}"
        source_name = f"source_{source_idx}_moment_{'_'.join(['_'.join(s.split()) for s in self.data.get_source_configurations()[source_idx]])}"
        input_file_name = os.path.splitext(os.path.basename(input_file))[0]
        output_dir = os.path.join(model_name, source_name, input_file_name)

        fr_path = os.path.join(output_dir, "FR.txt")
        print(f"Constructed file path: {fr_path}")
        if not os.path.exists(fr_path):
            print(f"Warning: {fr_path} does not exist.")
            return
        FR = self.data.read_FR(fr_path)

        XD = self.data.read_distances(input_file)
        XD=XD * 1000
        # Debugging: Print the data shapes and some values to ensure correctness
        print(f"XD size: {XD.size}, real_data shape: {FR.shape}")
        print(f"First 5 values of XD: {XD[:5]}")
        print(f"First 5 rows of FR: {FR[:5]}")

        sorted_indices = np.argsort(XD)
        print(f"Sorted indices: {sorted_indices}")
        print(f"Sorted indices size: {sorted_indices.size}")

        XD_sorted = XD[sorted_indices]
        real_data_sorted = FR[sorted_indices]

        num_traces = real_data_sorted.shape[0]
        num_samples = real_data_sorted.shape[1]
        time_axis = np.arange(num_samples) * dt + stmin + tshift

        fig = go.Figure()

        def plot_data(data, color, label):
            for i in range(num_traces):
                trace = data[i, :]
                xx = XD_sorted[i]

                if lnrm:
                    afactr = slider_amp
                else:
                    afactr = 1000.0 * slider_amp * (b1 + b2 * (xx / epic)) ** eps

                toff = stmin + tshift
                loff = int(toff / dt)
                npts = num_samples + loff

                t0 = stmin

                if lnrm:
                    trace = trace / np.max(np.abs(trace))

                if velocity_line is not None:
                    velocity_time = xx / velocity_line
                    amp_index = int(velocity_time / dt)
                    if amp_offset >= 0:
                        trace[amp_index:] *= amp_factor
                    else:
                        trace[:amp_index] *= amp_factor

                scaled_trace = trace[loff:npts] * afactr

                fig.add_trace(go.Scatter(
                    x=[xx] * (npts - loff),
                    y=time_axis[:npts-loff] + t0,
                    mode='lines',
                    line=dict(color=color, width=0.4),
                    name=label if i == 0 else "",
                    showlegend=i == 0
                ))

        plot_data(real_data_sorted, real_color, 'Real Data')

        x_min = np.min(XD_sorted)
        x_max = np.max(XD_sorted)
        x = np.linspace(x_min, x_max, 300)
        sorted_velocities = sorted(set(velocities))
        y_values = [x / (v * 1000) for v in sorted_velocities]
        colors = [f'rgba({int(c[0]*255)}, {int(c[1]*255)}, {int(c[2]*255)}, 1.0)' for c in plt.cm.jet(np.linspace(0, 1, len(velocities)))]
        for y, v, c in zip(y_values, sorted_velocities, colors):
            fig.add_trace(go.Scatter(
                x=x,
                y=y,
                mode='lines',
                line=dict(color=c),
                name=f'{v * 1000} m/s'
            ))

        fig.update_layout(
            title='Shot Gather',
            xaxis_title='Offset (m)',
            yaxis_title='Time (s)',
            yaxis_autorange='reversed',
            xaxis_side='top',
            legend=dict(x=0.01, y=0.2),
            margin=dict(l=0, r=0, t=40, b=0),
            showlegend=True,
        )

        fig.update_yaxes(range=[0, 5])

        fig.update_xaxes(showgrid=True, gridwidth=0.5, gridcolor='LightGrey')
        fig.update_yaxes(showgrid=True, gridwidth=0.5, gridcolor='LightGrey')

        return fig






    