import matplotlib.pyplot as plt
import pandas as pd

def make_detailed_scatter_figure(df: pd.DataFrame, plot_type: str = "pie"):
    """Create detailed visualization figures for compliance data"""
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from matplotlib.patches import Circle

    STATE_NAME_MAPPING = {
        'ANDAMAN & NICOBAR ISLANDS': 'Andaman and Nicobar Islands',
        'ANDAMAN AND NICOBAR ISLANDS': 'Andaman and Nicobar Islands',
        'ARUNACHAL PRADESH': 'Arunachal Pradesh',
        'ASSAM': 'Assam',
        'BIHAR': 'Bihar',
        'CHANDIGARH': 'Chandigarh',
        'CHHATTISGARH': 'Chhattisgarh',
        'DADRA & NAGAR HAVELI': 'Dadra and Nagar Haveli',
        'DADRA AND NAGAR HAVELI': 'Dadra and Nagar Haveli',
        'DAMAN & DIU': 'Daman and Diu',
        'DAMAN AND DIU': 'Daman and Diu',
        'DELHI': 'Delhi',
        'NCT OF DELHI': 'Delhi',
        'NATIONAL CAPITAL TERRITORY OF DELHI': 'Delhi',
        'GOA': 'Goa',
        'GUJARAT': 'Gujarat',
        'HARYANA': 'Haryana',
        'HIMACHAL PRADESH': 'Himachal Pradesh',
        'JAMMU AND KASHMIR': 'Jammu and Kashmir',
        'JHARKHAND': 'Jharkhand',
        'KARNATAKA': 'Karnataka',
        'KERALA': 'Kerala',
        'LADAKH': 'Ladakh',
        'LAKSHADWEEP': 'Lakshadweep',
        'MADHYA PRADESH': 'Madhya Pradesh',
        'MAHARASHTRA': 'Maharashtra',
        'MANIPUR': 'Manipur',
        'MEGHALAYA': 'Meghalaya',
        'MIZORAM': 'Mizoram',
        'NAGALAND': 'Nagaland',
        'ODISHA': 'Odisha',
        'ORISSA': 'Odisha',
        'PUDUCHERRY': 'Puducherry',
        'PONDICHERRY': 'Puducherry',
        'PUNJAB': 'Punjab',
        'RAJASTHAN': 'Rajasthan',
        'SIKKIM': 'Sikkim',
        'TAMIL NADU': 'Tamil Nadu',
        'TELANGANA': 'Telangana',
        'TRIPURA': 'Tripura',
        'UTTAR PRADESH': 'Uttar Pradesh',
        'UTTARAKHAND': 'Uttarakhand',
        'UTTARANCHAL': 'Uttarakhand',
        'WEST BENGAL': 'West Bengal'
    }

    def standardize_state_names(df):
        """Standardize state names to consistent format"""
        if 'state' not in df.columns:
            return df
        df_copy = df.copy()
        df_copy['state_upper'] = df_copy['state'].str.upper()
        df_copy['state_standardized'] = df_copy['state_upper'].map(STATE_NAME_MAPPING)
        df_copy['state_standardized'] = df_copy['state_standardized'].fillna(df_copy['state'].str.title())
        df_copy['state'] = df_copy['state_standardized']
        return df_copy.drop(['state_upper', 'state_standardized'], axis=1)

    # Color scheme for compliance categories
    violation_colors = {
        'Fully Compliant': '#059669',
        '1 Violation': '#facc15',
        '2 Violations': '#f97316',
        '3 Violations': '#dc2626'
    }
    
    # Marker styles for different compliance categories
    markers = {
        'Fully Compliant': 'o', 
        '1 Violation': '^',
        '2 Violations': 's', 
        '3 Violations': 'X'
    }
    
    # Ensure proper state name standardization
    df = standardize_state_names(df)

    if plot_type == "pie":
        # Create pie chart for compliance distribution
        category_counts = df['compliance_category'].value_counts()
        total = sum(category_counts.values)
        
        # Prepare data for pie chart
        category_info = []
        for i, (category, count) in enumerate(category_counts.items()):
            percentage = (count / total) * 100
            color = violation_colors.get(category.split(' (')[0], '#6b7280')
            category_info.append((category, count, percentage, color, i))
        
        # Sort by percentage for better visual organization
        category_info = sorted(category_info, key=lambda x: x[2])
        color_sorted = [c[3] for c in category_info]
        count_sorted = [c[1] for c in category_info]
        
        fig, ax = plt.subplots(figsize=(9, 7), dpi=120)
        
        # Create donut chart
        wedges, _ = ax.pie(
            count_sorted,
            colors=color_sorted,
            startangle=90,
            wedgeprops=dict(width=0.5, edgecolor='white', linewidth=3)
        )
        
        # Add center circle for donut effect
        centre_circle = Circle((0, 0), 0.50, fc='#f8fafc', linewidth=3, edgecolor='#e2e8f0')
        ax.add_artist(centre_circle)
        
        # Add center text with key metrics
        total_kilns = len(df)
        compliance_count = df['overall_compliant'].sum()
        compliance_rate = (compliance_count / total_kilns) * 100 if total_kilns > 0 else 0
        
        ax.text(0, 0.13, f"{total_kilns:,}", ha='center', va='center',
                fontsize=22, fontweight='bold', color='#1e293b')
        ax.text(0, 0, "Total Kilns", ha='center', va='center',
                fontsize=13, color='#64748b')
        ax.text(0, -0.17, f"{compliance_rate:.1f}% Compliant",
                ha='center', va='center', fontsize=15, fontweight='600',
                color=violation_colors['Fully Compliant'])
        
        ax.set_title("Compliance Status Distribution",
                     fontsize=18, fontweight='bold', pad=24, color='#1e293b')
        
        # Create legend
        legend_labels = [f"{c[0]} ({c[1]}) - {c[2]:.1f}%" for c in category_info]
        ax.legend(wedges, legend_labels, loc='center left', bbox_to_anchor=(1, 0.5), fontsize=12)
        
        plt.tight_layout()
        return fig

    elif plot_type == "scatter":
        # Create geographic scatter plot
        fig, ax = plt.subplots(figsize=(12, 10), dpi=120)
        
        # Standardize state names
        df = standardize_state_names(df)
        
        # Calculate aspect ratio for better visualization
        if 'lat' in df.columns and 'lon' in df.columns and len(df) > 0:
            lat_range = df['lat'].max() - df['lat'].min()
            lon_range = df['lon'].max() - df['lon'].min()
            aspect_ratio = max(0.6, min(1.4, lon_range / lat_range)) if lat_range > 0 else 1
        else:
            aspect_ratio = 1
        
        # Add India state boundaries
        plot_state_boundaries(ax)
        
        # Calculate appropriate marker size based on data density
        total_points = len(df)
        base_size = max(60, min(150, 2000/total_points))
        
        # Plot points by compliance category
        for i, category in enumerate(sorted(df['compliance_category'].unique())):
            cat_type = category.split(' (')[0] if '(' in category else category
            color = violation_colors.get(cat_type, '#6b7280')
            marker = markers.get(cat_type, 'o')
            
            subset = df[df['compliance_category'] == category]
            
            if not subset.empty and 'lat' in subset.columns and 'lon' in subset.columns:
                ax.scatter(subset['lon'], subset['lat'],
                          c=color, marker=marker, s=base_size,
                          label=f"{category} ({len(subset)})",
                          alpha=0.8, edgecolors='white', linewidth=1.5,
                          zorder=5+i)
        
        # Set proper axis limits with padding
        if not df.empty and 'lat' in df.columns and 'lon' in df.columns:
            padding_lon = (df['lon'].max() - df['lon'].min()) * 0.05
            padding_lat = (df['lat'].max() - df['lat'].min()) * 0.05
            ax.set_xlim(df['lon'].min() - padding_lon, df['lon'].max() + padding_lon)
            ax.set_ylim(df['lat'].min() - padding_lat, df['lat'].max() + padding_lat)
        
        # Apply styling
        enhance_axes_styling(ax, "Geographic Distribution of Compliance Status",
                            "Longitude (°)", "Latitude (°)")
        
        # Add legend
        legend = ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left',
                          frameon=True, fancybox=True, shadow=True,
                          title="Compliance Categories", title_fontsize=12)
        legend.get_frame().set_facecolor('#ffffff')
        legend.get_frame().set_alpha(0.95)
        
        plt.tight_layout()
        return fig

    elif plot_type == "bar":
        # Create bar chart for violation types
        violation_data = {
            'Kiln\nDistance': int(df['kiln_violation'].sum()),
            'Hospital\nDistance': int(df['hospital_violation'].sum()),
            'Water\nDistance': int(df['water_violation'].sum())
        }
        
        colors = ['#dc2626', '#f97316', '#eab308']
        x_pos = np.arange(len(violation_data))
        
        fig, ax = plt.subplots(figsize=(10, 6), dpi=120)
        
        bars = ax.bar(x_pos, violation_data.values(),
                      color=colors, alpha=0.9, width=0.7,
                      edgecolor='white', linewidth=2)
        
        # Customize x-axis
        ax.set_xticks(x_pos)
        ax.set_xticklabels(violation_data.keys(), fontsize=13, fontweight='600')
        
        # Add value labels on bars
        max_val = max(violation_data.values()) if violation_data.values() else 1
        for i, (bar, value) in enumerate(zip(bars, violation_data.values())):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, height + max_val * 0.01,
                    f'{value:,}', ha='center', va='bottom',
                    fontweight='bold', fontsize=13, color='#1e293b')
        
        # Set y-axis limit with padding
        ax.set_ylim(0, max_val * 1.15)
        
        # Apply styling
        enhance_axes_styling(ax, "Distribution of Violation Types", 
                            None, "Number of Violations")
        
        plt.tight_layout()
        return fig

    elif plot_type == "state":
        # Create state-wise horizontal stacked bar chart
        state_data = []
        for state, group in df.groupby("state"):
            state_data.append({
                'state': state,
                'fully_compliant': int(group['overall_compliant'].sum()),
                'one_violation': int(group['compliance_category'].str.startswith("1 Violation").sum()),
                'two_violations': int(group['compliance_category'].str.startswith("2 Violations").sum()),
                'three_violations': int((group['compliance_category'] == "3 Violations (all)").sum()),
                'total': len(group)
            })
        
        state_df = pd.DataFrame(state_data).sort_values('total', ascending=True)
        
        # Calculate figure size based on number of states
        n_states = len(state_df)
        y_pos = np.arange(len(state_df))
        bar_height = max(0.6, min(0.9, 15/n_states)) if n_states > 0 else 0.8
        
        fig, ax = plt.subplots(figsize=(12, max(6, n_states * 0.4)), dpi=120)
        
        # Create stacked horizontal bars
        ax.barh(y_pos, state_df['fully_compliant'], bar_height,
                label='Fully Compliant', color=violation_colors['Fully Compliant'],
                alpha=0.9, edgecolor='white', linewidth=0.5)
        
        ax.barh(y_pos, state_df['one_violation'], bar_height,
                left=state_df['fully_compliant'], label='1 Violation',
                color=violation_colors['1 Violation'], alpha=0.9,
                edgecolor='white', linewidth=0.5)
        
        ax.barh(y_pos, state_df['two_violations'], bar_height,
                left=state_df['fully_compliant'] + state_df['one_violation'],
                label='2 Violations', color=violation_colors['2 Violations'],
                alpha=0.9, edgecolor='white', linewidth=0.5)
        
        ax.barh(y_pos, state_df['three_violations'], bar_height,
                left=state_df['fully_compliant'] + state_df['one_violation'] + state_df['two_violations'],
                label='3 Violations', color=violation_colors['3 Violations'],
                alpha=0.9, edgecolor='white', linewidth=0.5)
        
        # Customize y-axis labels (truncate long state names)
        state_labels = [s if len(s) < 20 else s[:17] + '...' for s in state_df['state']]
        ax.set_yticks(y_pos)
        ax.set_yticklabels(state_labels, fontsize=12, fontweight='500')
        
        # Apply styling
        enhance_axes_styling(ax, "State-wise Compliance Breakdown", 
                            "Number of Kilns", None)
        
        # Add legend
        ax.legend(loc='lower right', frameon=True, fancybox=True, shadow=True)
        
        plt.tight_layout()
        return fig

    else:
        # Default fallback to pie chart
        return make_detailed_scatter_figure(df, plot_type="pie")
    
def make_all_compliance_figures(df):
    try:
        return (
            make_detailed_scatter_figure(df, plot_type="pie"),
            make_detailed_scatter_figure(df, plot_type="scatter"),
            make_detailed_scatter_figure(df, plot_type="bar"),
            make_detailed_scatter_figure(df, plot_type="state"),
        )
    except Exception as e:
        print(f"Error creating figures: {e}")
        return None, None, None, None