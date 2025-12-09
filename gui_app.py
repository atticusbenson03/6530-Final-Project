"""
Desktop GUI for Query Explanation Heatmaps
"""
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import json
from collections import defaultdict
import pandas as pd
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.backends._backend_tk import NavigationToolbar2Tk
from matplotlib.figure import Figure
import seaborn as sns
from DBFinal import (
    get_root, walk_plan, add_columns_from_text,
    scale_tables, scale_nodes, scale_columns,
    load_csv, run_query, row_scores,
    table_from_query, guess_agg,
    where_clause, split_predicates,
    build_predicate_functions, predicate_scores,
)
class QueryExplanationGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Query Explanation Heatmaps")
        self.root.geometry("1400x900")
        self.root.minsize(1200, 800)
        
        self.data_frame = None
        self.sqlite_connection = None
        self.result_frame = None
        self.table_info = {}
        self.column_scores = {}
        self.predicate_scores_list = []
        self.predicate_funcs = []
        self.predicate_row_masks = {}
        self.column_row_masks = {}
        self.row_scores_list = []
        self.current_table = None
        self.csv_path = None
        
        self.setup_ui()
        
    def setup_ui(self):
        main_paned = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        main_paned.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        left_frame = ttk.Frame(main_paned, width=450)
        main_paned.add(left_frame, weight=1)
        
        right_frame = ttk.Frame(main_paned)
        main_paned.add(right_frame, weight=3)
        
        self.setup_input_panel(left_frame)
        self.setup_visualization_panel(right_frame)
        
    def setup_input_panel(self, parent):
        parent.columnconfigure(0, weight=1)
        
        file_frame = ttk.LabelFrame(parent, text="Data Source", padding=10)
        file_frame.grid(row=0, column=0, sticky="ew", padx=5, pady=5)
        file_frame.columnconfigure(1, weight=1)
        
        ttk.Label(file_frame, text="CSV File:").grid(row=0, column=0, sticky="w")
        self.file_path_var = tk.StringVar()
        ttk.Entry(file_frame, textvariable=self.file_path_var, state="readonly").grid(row=0, column=1, sticky="ew", padx=5)
        ttk.Button(file_frame, text="Browse...", command=self.browse_csv).grid(row=0, column=2)
        
        plan_frame = ttk.LabelFrame(parent, text="Query Execution Plan (EXPLAIN FORMAT JSON)", padding=10)
        plan_frame.grid(row=1, column=0, sticky="nsew", padx=5, pady=5)
        plan_frame.columnconfigure(0, weight=1)
        plan_frame.rowconfigure(0, weight=1)
        parent.rowconfigure(1, weight=1)
        
        self.plan_text = scrolledtext.ScrolledText(plan_frame, height=10, wrap=tk.WORD, font=("Consolas", 9))
        self.plan_text.grid(row=0, column=0, sticky="nsew")
        
        query_frame = ttk.LabelFrame(parent, text="SQL Query", padding=10)
        query_frame.grid(row=2, column=0, sticky="nsew", padx=5, pady=5)
        query_frame.columnconfigure(0, weight=1)
        query_frame.rowconfigure(0, weight=1)
        parent.rowconfigure(2, weight=1)
        
        self.query_text = scrolledtext.ScrolledText(query_frame, height=8, wrap=tk.WORD, font=("Consolas", 9))
        self.query_text.grid(row=0, column=0, sticky="nsew")
        
        button_frame = ttk.Frame(parent)
        button_frame.grid(row=3, column=0, sticky="ew", padx=5, pady=10)
        button_frame.columnconfigure(0, weight=1)
        
        self.analyze_btn = ttk.Button(button_frame, text="Analyze Query", command=self.analyze_query)
        self.analyze_btn.pack(fill=tk.X, pady=5)
        
        ttk.Button(button_frame, text="Clear All", command=self.clear_all).pack(fill=tk.X, pady=2)
        
        nav_frame = ttk.LabelFrame(parent, text="Navigation", padding=10)
        nav_frame.grid(row=4, column=0, sticky="ew", padx=5, pady=5)
        
        self.breadcrumb_var = tk.StringVar(value="Home")
        ttk.Label(nav_frame, textvariable=self.breadcrumb_var, font=("Arial", 10, "bold")).pack(anchor="w")
        
        nav_buttons = ttk.Frame(nav_frame)
        nav_buttons.pack(fill=tk.X, pady=5)
        
        ttk.Button(nav_buttons, text="Database View", command=self.show_table_view).pack(side=tk.LEFT, padx=2)
        ttk.Button(nav_buttons, text="Back", command=self.go_back).pack(side=tk.LEFT, padx=2)
        
        status_frame = ttk.LabelFrame(parent, text="Status", padding=10)
        status_frame.grid(row=5, column=0, sticky="ew", padx=5, pady=5)
        
        self.status_var = tk.StringVar(value="Ready. Load a CSV file and enter query plan/SQL to begin.")
        ttk.Label(status_frame, textvariable=self.status_var, wraplength=400).pack(anchor="w")
        
    def setup_visualization_panel(self, parent):
        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(0, weight=1)
        
        self.viz_notebook = ttk.Notebook(parent)
        self.viz_notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.heatmap_frame = ttk.Frame(self.viz_notebook)
        self.viz_notebook.add(self.heatmap_frame, text="Heatmap Visualization")
        
        self.details_frame = ttk.Frame(self.viz_notebook)
        self.viz_notebook.add(self.details_frame, text="Row Details")
        
        self.setup_heatmap_canvas(self.heatmap_frame)
        self.setup_details_view(self.details_frame)
        
        self.view_stack = []
        self.current_view = "tables"
        
    def setup_heatmap_canvas(self, parent):
        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(0, weight=1)
        
        self.figure = Figure(figsize=(10, 7), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.figure, parent)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        toolbar_frame = ttk.Frame(parent)
        toolbar_frame.pack(fill=tk.X)
        self.toolbar = NavigationToolbar2Tk(self.canvas, toolbar_frame)
        self.toolbar.update()
        
        self.canvas.mpl_connect('button_press_event', self.on_heatmap_click)
        
    def setup_details_view(self, parent):
        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(0, weight=1)
        
        columns = ("Row Index", "Score", "Data")
        self.details_tree = ttk.Treeview(parent, columns=columns, show="headings")
        
        for col in columns:
            self.details_tree.heading(col, text=col)
            self.details_tree.column(col, width=200)
        
        scrollbar = ttk.Scrollbar(parent, orient=tk.VERTICAL, command=self.details_tree.yview)
        self.details_tree.configure(yscrollcommand=scrollbar.set)
        
        self.details_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
    def browse_csv(self):
        file_path = filedialog.askopenfilename(
            title="Select CSV File",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        if file_path:
            self.csv_path = file_path
            self.file_path_var.set(file_path)
            self.status_var.set(f"Loaded: {file_path.split('/')[-1]}")
            
    def analyze_query(self):
        if not self.csv_path:
            messagebox.showerror("Error", "Please select a CSV file first.")
            return
            
        plan_text = self.plan_text.get("1.0", tk.END).strip()
        sql_text = self.query_text.get("1.0", tk.END).strip()
        
        if not plan_text:
            messagebox.showerror("Error", "Please enter the query execution plan.")
            return
            
        if not sql_text:
            messagebox.showerror("Error", "Please enter the SQL query.")
            return
            
        try:
            self.status_var.set("Parsing plan...")
            self.root.update()
            
            plan = json.loads(plan_text)
            root_node = get_root(plan)
            
            tbl_name = table_from_query(sql_text) or "T"
            self.current_table = tbl_name
            
            self.status_var.set("Loading CSV data...")
            self.root.update()
            
            self.data_frame, self.sqlite_connection = load_csv(self.csv_path, tbl_name)
            
            self.status_var.set("Running query...")
            self.root.update()
            
            self.result_frame = run_query(self.sqlite_connection, sql_text)
            self.status_var.set("Analyzing contributions...")
            self.root.update()

            # Infer aggregate and input column
            agg_name, value_column, input_column = guess_agg(sql_text, self.result_frame, self.data_frame)
            if value_column not in self.result_frame.columns:
                value_column = self.result_frame.columns[0]

            # Valid columns: result columns plus aggregate input column (e.g., Index)
            valid_columns = set(self.result_frame.columns)
            if input_column is not None:
                valid_columns.add(input_column)

            self.table_info = defaultdict(lambda: {"total_cost": 0.0, "row_count": 0.0})
            node_info = {}
            column_counts = defaultdict(float)

            walk_plan(root_node, self.table_info, node_info, column_counts, valid_columns)
            scale_tables(self.table_info)
            scale_nodes(node_info)
            self.column_scores = scale_columns(column_counts)

            self.row_scores_list = row_scores(self.result_frame, value_column, agg_name)
                
            where_text = where_clause(sql_text)
            predicate_texts = split_predicates(where_text)
            self.predicate_funcs = build_predicate_functions(predicate_texts, self.result_frame)
            self.predicate_scores_list = predicate_scores(self.result_frame, self.row_scores_list, self.predicate_funcs)
            
            self.predicate_row_masks = {}
            for label, check_func in self.predicate_funcs:
                mask = []
                for idx, row_data in self.result_frame.iterrows():
                    mask.append(check_func(row_data))
                self.predicate_row_masks[label] = mask
            
            self.column_row_masks = {}
            for (tbl, col), score in self.column_scores.items():
                if col in self.result_frame.columns:
                    mask = self.result_frame[col].notna().tolist()
                    self.column_row_masks[col] = mask
            
            self.status_var.set(f"Analysis complete! Found {len(self.result_frame)} rows.")
            self.show_table_view()
            
        except json.JSONDecodeError as e:
            messagebox.showerror("Error", f"Could not parse JSON plan: {str(e)}")
        except Exception as e:
            messagebox.showerror("Error", f"Analysis error: {str(e)}")
            import traceback
            traceback.print_exc()
            
    def show_table_view(self):
        self.current_view = "tables"
        self.view_stack = []
        self.breadcrumb_var.set("Database Overview > Tables")
        
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        
        if not self.table_info:
            ax.text(0.5, 0.5, "No data. Run analysis first.", ha='center', va='center', fontsize=14)
            self.canvas.draw()
            return
            
        tables = list(self.table_info.keys())
        scores = [self.table_info[t].get("score", 0) for t in tables]
        
        data = np.array(scores).reshape(1, -1)
        sns.heatmap(data, ax=ax, annot=True, fmt=".4f", cmap="YlOrRd",
                   xticklabels=tables, yticklabels=["Contribution"],
                   vmin=0, vmax=1, cbar_kws={'label': 'Contribution Score'})
        
        ax.set_title("Table-Level Contribution Scores\n(Click on a table to drill down)", fontsize=12, fontweight='bold')
        
        self.clickable_regions = []
        for i, table in enumerate(tables):
            self.clickable_regions.append({
                "type": "table", "name": table,
                "x_range": (i, i + 1), "y_range": (0, 1)
            })
        
        self.figure.tight_layout()
        self.canvas.draw()
        
    def show_table_details(self, table_name):
        self.current_view = "table_details"
        self.view_stack.append(("tables", None))
        self.breadcrumb_var.set(f"Database Overview > {table_name}")
        
        self.figure.clear()
        ax1 = self.figure.add_subplot(121)
        ax2 = self.figure.add_subplot(122)
        
        table_columns = [col for (tbl, col), score in self.column_scores.items() 
                        if tbl == table_name or tbl is None]
        col_scores = [self.column_scores.get((table_name, col), 0) or 
                     self.column_scores.get((None, col), 0) for col in table_columns]
        
        if table_columns:
            data = np.array(col_scores).reshape(-1, 1)
            sns.heatmap(data, ax=ax1, annot=True, fmt=".4f", cmap="Blues",
                       yticklabels=table_columns, xticklabels=["Score"],
                       vmin=0, vmax=max(col_scores) if col_scores else 1,
                       cbar_kws={'label': 'Column Contribution'})
            ax1.set_title(f"Column Contributions\n{table_name}", fontsize=11, fontweight='bold')
        else:
            ax1.text(0.5, 0.5, "No column data", ha='center', va='center')
        
        self.clickable_regions = []
        
        if self.predicate_scores_list:
            pred_labels = [p[0][:30] + "..." if len(p[0]) > 30 else p[0] for p in self.predicate_scores_list]
            pred_values = [p[1] for p in self.predicate_scores_list]
            
            data = np.array(pred_values).reshape(-1, 1)
            sns.heatmap(data, ax=ax2, annot=True, fmt=".4f", cmap="Greens",
                       yticklabels=pred_labels, xticklabels=["Score"],
                       vmin=0, vmax=max(pred_values) if pred_values else 1,
                       cbar_kws={'label': 'Predicate Contribution'})
            ax2.set_title("Predicate Contributions\n(Click to see rows)", fontsize=11, fontweight='bold')
            
            for i, (label, score) in enumerate(self.predicate_scores_list):
                self.clickable_regions.append({
                    "type": "predicate", "name": label,
                    "index": i, "ax": ax2, "y_range": (i, i + 1)
                })
        else:
            ax2.text(0.5, 0.5, "No predicates found", ha='center', va='center')
        
        for i, col in enumerate(table_columns):
            self.clickable_regions.append({
                "type": "column", "name": col, "table": table_name,
                "index": i, "ax": ax1, "y_range": (i, i + 1)
            })
        
        self.figure.tight_layout()
        self.canvas.draw()
        
    def show_row_contributions(self, filter_type=None, filter_value=None):
        self.current_view = "rows"
        self.view_stack.append(("table_details", self.current_table))
        
        if filter_type == "predicate":
            display_val = filter_value[:30] + "..." if len(filter_value) > 30 else filter_value
            self.breadcrumb_var.set(f"Database > {self.current_table} > Predicate: {display_val}")
        elif filter_type == "column":
            self.breadcrumb_var.set(f"Database > {self.current_table} > Column: {filter_value}")
        else:
            self.breadcrumb_var.set(f"Database > {self.current_table} > All Rows")
        
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        
        if self.result_frame is None or self.result_frame.empty:
            ax.text(0.5, 0.5, "No row data available", ha='center', va='center')
            self.canvas.draw()
            return
            
        display_frame = self.result_frame.copy()
        display_frame["__score__"] = self.row_scores_list
        
        if filter_type == "predicate" and filter_value in self.predicate_row_masks:
            mask = self.predicate_row_masks[filter_value]
            display_frame = display_frame[mask]
        elif filter_type == "column" and filter_value in self.column_row_masks:
            mask = self.column_row_masks[filter_value]
            display_frame = display_frame[mask]
        
        if display_frame.empty:
            ax.text(0.5, 0.5, f"No rows match the selected {filter_type}", ha='center', va='center')
            self.canvas.draw()
            return
        
        display_frame = display_frame.sort_values("__score__", ascending=False).head(50)
        data_for_heatmap = display_frame[["__score__"]].values
        row_labels = [f"Row {i}" for i in display_frame.index[:50]]
        
        max_val = data_for_heatmap.max() if data_for_heatmap.max() > 0 else 1
        sns.heatmap(data_for_heatmap, ax=ax, annot=True, fmt=".4f", cmap="YlOrRd",
                   yticklabels=row_labels, xticklabels=["Contribution"],
                   vmin=0, vmax=max_val, cbar_kws={'label': 'Row Contribution Score'})
        
        ax.set_title("Row-Level Contributions (Top 50)", fontsize=12, fontweight='bold')
        self.figure.tight_layout()
        self.canvas.draw()
        
        self.update_details_tree(display_frame)
        self.viz_notebook.select(1)
        
    def update_details_tree(self, df):
        for item in self.details_tree.get_children():
            self.details_tree.delete(item)
        for idx, row in df.iterrows():
            score = row.get("__score__", 0)
            data_dict = {k: v for k, v in row.items() if k != "__score__"}
            data_str = ", ".join([f"{k}={v}" for k, v in list(data_dict.items())[:5]])
            self.details_tree.insert("", tk.END, values=(idx, f"{score:.6f}", data_str))
        
    def on_heatmap_click(self, event):
        if event.inaxes is None:
            return
        x, y = event.xdata, event.ydata
        if x is None or y is None:
            return
            
        for region in getattr(self, 'clickable_regions', []):
            if region["type"] == "table":
                x_min, x_max = region["x_range"]
                y_min, y_max = region["y_range"]
                if x_min <= x < x_max and y_min <= y < y_max:
                    self.show_table_details(region["name"])
                    return
            elif region["type"] == "predicate":
                if event.inaxes == region.get('ax'):
                    y_min, y_max = region["y_range"]
                    if y_min <= y < y_max:
                        self.show_row_contributions("predicate", region["name"])
                        return
            elif region["type"] == "column":
                if event.inaxes == region.get('ax'):
                    y_min, y_max = region["y_range"]
                    if y_min <= y < y_max:
                        self.show_row_contributions("column", region["name"])
                        return
                        
    def go_back(self):
        if not self.view_stack:
            self.show_table_view()
            return
        view_type, view_data = self.view_stack.pop()
        if view_type == "tables":
            self.show_table_view()
        elif view_type == "table_details":
            self.show_table_details(view_data)
            
    def clear_all(self):
        self.plan_text.delete("1.0", tk.END)
        self.query_text.delete("1.0", tk.END)
        self.file_path_var.set("")
        self.csv_path = None
        self.data_frame = None
        self.result_frame = None
        self.table_info = {}
        self.column_scores = {}
        self.predicate_scores_list = []
        self.row_scores_list = []
        if self.sqlite_connection:
            self.sqlite_connection.close()
            self.sqlite_connection = None
        self.figure.clear()
        self.canvas.draw()
        for item in self.details_tree.get_children():
            self.details_tree.delete(item)
        self.status_var.set("Cleared. Ready for new analysis.")
        self.breadcrumb_var.set("Home")
def main():
    root = tk.Tk()
    style = ttk.Style()
    style.theme_use('clam')
    app = QueryExplanationGUI(root)
    root.mainloop()
if __name__ == "__main__":
    main()