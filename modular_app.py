#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import io
import gradio as gr
from data_utils import _read_csv, _ensure_cols
from geo_utils import load_states_geojson_folder, assign_kilns_to_states
from compliance import compute_compliance
from plotting import make_all_compliance_figures
import pandas as pd

shapefile_path = "india_shapefiles/"

def filter_data_by_state(full_data: pd.DataFrame, selected_state: str):
    if selected_state is None or selected_state == "All States":
        filtered_data = full_data
        title_prefix = "All States"
    else:
        filtered_data = full_data[full_data["state"] == selected_state]
        title_prefix = selected_state
        if len(filtered_data) == 0:
            return f"No data found for {selected_state}", b""
    total = len(filtered_data)
    fully_compliant = int(filtered_data["overall_compliant"].sum())
    one_violation = int((filtered_data["compliance_category"].str.startswith("1 Violation")).sum())
    two_violations = int((filtered_data["compliance_category"].str.startswith("2 Violations")).sum())
    three_violations = int((filtered_data["compliance_category"] == "3 Violations (all)").sum())
    kiln_violations = int(filtered_data["kiln_violation"].sum())
    hospital_violations = int(filtered_data["hospital_violation"].sum())
    water_violations = int(filtered_data["water_violation"].sum())
    summary = (
        f"## {title_prefix} - Detailed Compliance Report\n\n"
        f"**Total kilns: {total}**\n\n"
        f"### Compliance Overview\n"
        f"- **Fully Compliant:** {fully_compliant} ({fully_compliant/total*100:.1f}%)\n"
        f"- **1 Violation:** {one_violation} ({one_violation/total*100:.1f}%)\n"
        f"- **2 Violations:** {two_violations} ({two_violations/total*100:.1f}%)\n"
        f"- **3 Violations:** {three_violations} ({three_violations/total*100:.1f}%)\n\n"
        f"### Individual Violation Types\n"
        f"- **Kiln Distance Violations:** {kiln_violations}\n"
        f"- **Hospital Distance Violations:** {hospital_violations}\n"
        f"- **Water Body Distance Violations:** {water_violations}\n\n"
    )
    category_summary = filtered_data['compliance_category'].value_counts().sort_index()
    summary += "### Detailed Breakdown\n\n"
    summary += "| Compliance Category | Count | Percentage |\n"
    summary += "|-------------------|-------|------------|\n"
    for category, count in category_summary.items():
        percentage = count/total*100
        summary += f"| {category} | {count} | {percentage:.1f}% |\n"
    csv_buffer = io.BytesIO()
    filtered_data.to_csv(csv_buffer, index=False)
    csv_buffer.seek(0)
    return summary, csv_buffer.read()

def save_bytes_to_tempfile(data_bytes, prefix="tempfile_", suffix=".csv"):
    import tempfile
    fd, path = tempfile.mkstemp(prefix=prefix, suffix=suffix)
    with os.fdopen(fd, 'wb') as f:
        f.write(data_bytes)
    return path

global_data = None

def _run(use_demo_flag, k, h, w, states_folder, kt, ht, wt, heat, cluster):
    global global_data
    try:
        if use_demo_flag:
            k = "data/kilns_clean.csv"
            h = "data/hospitals.csv" if os.path.exists("data/hospitals.csv") else None
            w = "data/waterways_wkt.csv" if os.path.exists("data/waterways_wkt.csv") else None

        map_html, summary_text, csv_bytes, full_data, state_list = compute_compliance(
            k, h, w, float(kt), float(ht), float(wt), bool(heat), bool(cluster),
            states_geojson_folder=states_folder
        )

        global_data = full_data
        pie, scatter, bar, state = make_all_compliance_figures(full_data)
        csv_path = save_bytes_to_tempfile(csv_bytes, prefix="full_dataset_")

        return (
            map_html,
            summary_text,
            pie,
            scatter,
            bar,
            state,
            gr.update(choices=state_list),
            gr.update(value="All States"),
            csv_path,
            "",
            None,
        )
    except Exception as e:
        error_msg = f"Error: {str(e)}"
        print(error_msg)
        return (
            f"<p style='color:red'>{error_msg}</p>",
            error_msg,
            None,
            None,
            None,
            None,
            gr.update(choices=["All States"]),
            gr.update(value="All States"),
            None,
            "",
            None,
        )

def _filter_by_state(selected_state):
    global global_data
    if global_data is None:
        return "Please run the analysis first.", None
    try:
        if selected_state == "All States":
            state_summary_text, state_csv_bytes = filter_data_by_state(global_data, selected_state=None)
            state_csv_path = save_bytes_to_tempfile(state_csv_bytes, prefix="all_states_data_") if len(state_csv_bytes) > 0 else None
            return state_summary_text, state_csv_path

        state_summary_text, state_csv_bytes = filter_data_by_state(global_data, selected_state)
        state_csv_path = save_bytes_to_tempfile(state_csv_bytes, prefix="state_data_") if len(state_csv_bytes) > 0 else None
        return state_summary_text, state_csv_path
    except Exception as e:
        error_msg = f"Error filtering data: {str(e)}"
        return error_msg, None

def _state_plot(selected_state):
    global global_data
    if global_data is None:
        return None, None, None, None

    try:
        if selected_state == "All States":
            return make_all_compliance_figures(global_data)

        filtered = global_data[global_data["state"] == selected_state]
        if len(filtered) == 0:
            return None, None, None, None

        return make_all_compliance_figures(filtered)
    except Exception as e:
        print(f"Error creating state plots: {e}")
        return None, None, None, None

## Gradio interface
with gr.Blocks(title="Enhanced Brick Kiln Compliance Monitor") as demo:
    gr.Markdown(
        "## Enhanced Compliance Monitoring for Brick Kilns\n"
        "Upload CSVs, set thresholds, and view detailed compliance analysis per state.\n"
        "- **New Features:** Detailed violation categories, state-wise filtering, enhanced CSV output\n"
        "- **Kilns CSV** must include columns: `lat, lon` (WGS84).\n"
        "- Hospitals CSV: `Latitude, Longitude` or `lat, lon`.\n"
        "- Waterways CSV: points (`lat, lon`) or WKT.LineString/MultiLineString in `geometry`.\n"
        "- State boundaries: drop GeoJSON files into folder; set folder path below.\n"
        "- **Compliance Categories:** Fully Compliant, 1 Violation, 2 Violations, 3 Violations"
    )

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### Data Input")
            use_demo = gr.Checkbox(value=True, label="Use bundled demo data (skip uploads)")
            kilns_csv = gr.File(label="Kilns CSV (required if demo OFF)", file_types=[".csv"])
            hospitals_csv = gr.File(label="Hospitals CSV (optional)", file_types=[".csv"])
            waterways_csv = gr.File(label="Waterways CSV or WKT (optional)", file_types=[".csv"])
            states_geojson_folder = gr.Textbox(
                value=shapefile_path,
                label="States GeoJSON Folder"
            )
            gr.Markdown("### Compliance Thresholds (km)")
            kiln_thresh = gr.Number(value=1.0, label="Min distance to nearest kiln (km)")
            hosp_thresh = gr.Number(value=0.8, label="Min distance to hospital (km)")
            water_thresh = gr.Number(value=0.5, label="Min distance to water body (km)")
            gr.Markdown("### Map Options")
            add_heatmap = gr.Checkbox(value=False, label="Add heatmap layer")
            cluster_points = gr.Checkbox(value=True, label="Cluster markers for speed")
            run_btn = gr.Button("Compute & Map", variant="primary")

        with gr.Column(scale=2):
            fmap = gr.HTML(label="Interactive Map with Detailed Categories")
            summary = gr.Markdown(label="Overall Summary")

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### State-wise Analysis")
            state_dropdown = gr.Dropdown(
                choices=["All States"],
                value="All States",
                label="Select State for Detailed Analysis",
                interactive=True
            )
            filter_btn = gr.Button("Show State Report", variant="secondary")
            state_plot_btn = gr.Button("Show State Plots", variant="secondary")
        with gr.Column(scale=2):
            state_summary = gr.Markdown(label="State-specific Analysis")

    with gr.Row():
        pie_plot = gr.Plot(label="Compliance Pie/Donut Chart")
    with gr.Row():
        scatter_plot = gr.Plot(label="Geographic Scatter Plot")
    with gr.Row():
        bar_plot = gr.Plot(label="Violation Types Bar Chart")
    with gr.Row():
        state_bar_plot = gr.Plot(label="State-wise Compliance Bar Chart")

    with gr.Row():
        gr.Markdown("### Download Options")
        download_csv = gr.File(label="Download Full Dataset CSV", interactive=False)
        download_state_csv = gr.File(label="Download State-specific CSV", interactive=False)

    run_btn.click(
        _run,
        inputs=[use_demo, kilns_csv, hospitals_csv, waterways_csv, states_geojson_folder,
                kiln_thresh, hosp_thresh, water_thresh, add_heatmap, cluster_points],
        outputs=[
            fmap,
            summary,
            pie_plot,
            scatter_plot,
            bar_plot,
            state_bar_plot,
            state_dropdown,  # updates choices here
            state_dropdown,  # updates value here
            download_csv,
            state_summary,
            download_state_csv,
        ],
    )

    filter_btn.click(
        _filter_by_state,
        inputs=[state_dropdown],
        outputs=[state_summary, download_state_csv]
    )

    state_plot_btn.click(
        _state_plot,
        inputs=[state_dropdown],
        outputs=[pie_plot, scatter_plot, bar_plot, state_bar_plot]
    )

if __name__ == "__main__":
    demo.launch()