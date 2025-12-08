import json
from collections import defaultdict
import sqlite3
import re

import pandas as pd
import numpy as np


def read_plan():
    print("Paste EXPLAIN (FORMAT JSON) output, end with an empty line:")
    lines = []
    while True:
        line = input()
        if line.strip() == "":
            break
        lines.append(line)
    text = "\n".join(lines).strip()
    if text == "":
        print("No plan text entered. Exiting.")
        return None
    try:
        plan = json.loads(text)
        return plan
    except Exception as error:
        print("Could not parse JSON: " + str(error))
        return None


def get_root(plan):
    if not isinstance(plan, list) or len(plan) == 0:
        raise ValueError("Plan JSON must be a non-empty list.")
    first_entry = plan[0]
    if "Plan" not in first_entry:
        raise ValueError("Top-level object has no 'Plan' key.")
    return first_entry["Plan"]


def walk_plan(node, table_info, node_info, column_counts, valid_columns):
    node_type = node.get("Node Type", "Unknown")
    relation_name = node.get("Relation Name")
    alias_name = node.get("Alias")
    total_cost = float(node.get("Total Cost", 0.0))
    estimated_rows = float(node.get("Plan Rows", 0.0))

    node_key = id(node)

    node_info[node_key] = {
        "node_type": node_type,
        "relation": relation_name,
        "alias": alias_name,
        "total_cost": total_cost,
        "row_count": estimated_rows,
    }

    if relation_name is not None:
        table_name = relation_name
        table_entry = table_info[table_name]
        table_entry["total_cost"] += total_cost
        table_entry["row_count"] += estimated_rows

    keys_with_expressions = [
        "Filter",
        "Index Cond",
        "Hash Cond",
        "Merge Cond",
        "Join Filter",
        "Recheck Cond",
        "Output",
        "Group Key",
    ]

    for expression_key in keys_with_expressions:
        value = node.get(expression_key)
        if value is None:
            continue
        if isinstance(value, list):
            for expression_text in value:
                add_columns_from_text(expression_text, column_counts, valid_columns)
        elif isinstance(value, str):
            add_columns_from_text(value, column_counts, valid_columns)

    child_nodes = node.get("Plans", [])
    if isinstance(child_nodes, list):
        for child_node in child_nodes:
            walk_plan(child_node, table_info, node_info, column_counts, valid_columns)


def add_columns_from_text(expression_text, column_counts, valid_columns):
    text = str(expression_text)
    cleaned_characters = []
    for character in text:
        if character.isalnum() or character == "." or character == "_":
            cleaned_characters.append(character)
        else:
            cleaned_characters.append(" ")
    cleaned_text = "".join(cleaned_characters)
    tokens = cleaned_text.split()
    for token in tokens:
        table_name = None
        column_name = None

        if "." in token:
            parts = token.split(".")
            if len(parts) >= 2:
                table_name = parts[-2]
                column_name = parts[-1]
        else:
            column_name = token

        if column_name is None:
            continue

        if column_name not in valid_columns:
            continue

        column_key = (table_name, column_name)
        column_counts[column_key] += 1.0


def scale_tables(table_info):
    total_cost_all_tables = sum(entry["total_cost"] for entry in table_info.values())
    if total_cost_all_tables > 0.0:
        for table_name, entry in table_info.items():
            entry["score"] = entry["total_cost"] / total_cost_all_tables
    else:
        total_rows_all_tables = sum(entry["row_count"] for entry in table_info.values())
        if total_rows_all_tables <= 0.0:
            for table_name, entry in table_info.items():
                entry["score"] = 0.0
        else:
            for table_name, entry in table_info.items():
                entry["score"] = entry["row_count"] / total_rows_all_tables


def scale_nodes(node_info):
    total_cost_all_nodes = 0.0
    for node_entry in node_info.values():
        total_cost_all_nodes += node_entry["total_cost"]
    if total_cost_all_nodes <= 0.0:
        for node_key, node_entry in node_info.items():
            node_entry["score"] = 0.0
    else:
        for node_key, node_entry in node_info.items():
            node_entry["score"] = node_entry["total_cost"] / total_cost_all_nodes


def scale_columns(column_counts):
    total_mentions = sum(column_counts.values())
    column_scores = {}
    if total_mentions <= 0.0:
        for column_key, count in column_counts.items():
            column_scores[column_key] = 0.0
    else:
        for column_key, count in column_counts.items():
            column_scores[column_key] = count / total_mentions
    return column_scores


def show_tables(table_info):
    print("\nTable contribution scores (from plan cost and rows):")
    if not table_info:
        print("No tables found in plan.")
        return
    sorted_tables = sorted(
        table_info.items(),
        key=lambda pair: pair[1].get("score", 0.0),
        reverse=True,
    )
    for table_name, entry in sorted_tables:
        score_value = entry.get("score", 0.0)
        print(
            "Table " + str(table_name)
            + " score " + str(round(score_value, 4))
            + " total_cost " + str(round(entry["total_cost"], 1))
            + " row_count " + str(round(entry["row_count"], 1))
        )


def show_nodes(node_info, max_nodes_to_show):
    print("\nPlan node contribution scores (by cost):")
    if not node_info:
        print("No plan nodes found.")
        return
    sorted_nodes = sorted(
        node_info.items(),
        key=lambda pair: pair[1].get("score", 0.0),
        reverse=True,
    )
    shown_count = 0
    for node_key, entry in sorted_nodes:
        if shown_count >= max_nodes_to_show:
            break
        print(
            "Node " + str(entry["node_type"])
            + " table " + str(entry["relation"])
            + " alias " + str(entry["alias"])
            + " score " + str(round(entry.get("score", 0.0), 4))
            + " cost " + str(round(entry["total_cost"], 1))
            + " rows " + str(round(entry["row_count"], 1))
        )
        shown_count += 1
    if len(sorted_nodes) > max_nodes_to_show:
        print("...", len(sorted_nodes) - max_nodes_to_show, "more nodes not shown")


def show_columns(column_scores, max_columns_to_show):
    print("\nColumn contribution scores (by mention frequency):")
    if not column_scores:
        print("No columns detected in plan expressions.")
        return
    sorted_columns = sorted(column_scores.items(), key=lambda pair: pair[1], reverse=True)
    shown_count = 0
    for (table_name, column_name), score_value in sorted_columns:
        if table_name is None:
            full_name = str(column_name)
        else:
            full_name = str(table_name) + "." + str(column_name)
        print("Column " + full_name + " score " + str(round(score_value, 4)))
        shown_count += 1
        if shown_count >= max_columns_to_show:
            break
    if len(sorted_columns) > max_columns_to_show:
        print("...", len(sorted_columns) - max_columns_to_show, "more columns not shown")


def load_csv(csv_path, table_name):
    print("\nLoading CSV into pandas and SQLite...")
    data_frame = pd.read_csv(csv_path)
    sqlite_connection = sqlite3.connect(":memory:")
    data_frame.to_sql(table_name, sqlite_connection, index=False, if_exists="replace")
    return data_frame, sqlite_connection


def run_query(sqlite_connection, sql_text):
    result_frame = pd.read_sql_query(sql_text, sqlite_connection)
    return result_frame


def row_scores(data_frame, value_column, agg_name):
    if data_frame.empty:
        return []

    numeric_series = pd.to_numeric(data_frame[value_column], errors="coerce").fillna(0.0)
    values = numeric_series.to_numpy(dtype=float)
    row_count = len(values)

    aggregate_name_upper = agg_name.upper()

    if aggregate_name_upper == "SUM":
        absolute_values = np.abs(values)
        total_absolute = absolute_values.sum()
        if total_absolute == 0.0:
            total_absolute = 1.0
        return (absolute_values / total_absolute).tolist()

    if aggregate_name_upper == "COUNT":
        if row_count == 0:
            return [0.0] * row_count
        return [1.0 / float(row_count)] * row_count

    if aggregate_name_upper == "AVG":
        total_sum = values.sum()
        if row_count == 0:
            return [0.0] * row_count
        average_value = total_sum / float(row_count)
        differences = []
        for value in values:
            sum_without = total_sum - value
            count_without = row_count - 1
            if count_without <= 0:
                average_without = average_value
            else:
                average_without = sum_without / float(count_without)
            differences.append(abs(average_value - average_without))
        differences_array = np.array(differences, dtype=float)
        total_difference = differences_array.sum()
        if total_difference == 0.0:
            total_difference = 1.0
        return (differences_array / total_difference).tolist()

    if row_count == 0:
        return [0.0] * row_count
    return [1.0 / float(row_count)] * row_count


def show_rows(data_frame, scores, max_rows_to_show):
    print("\nRow-level contributions (top rows):")
    if len(scores) == 0 or data_frame.empty:
        print("No row-level contributions to show.")
        return

    frame_with_scores = data_frame.copy()
    frame_with_scores["__score__"] = scores
    sorted_frame = frame_with_scores.sort_values("__score__", ascending=False)

    shown_count = 0
    for index, row in sorted_frame.iterrows():
        if shown_count >= max_rows_to_show:
            break
        score_value = row["__score__"]
        row_data = row.drop("__score__").to_dict()
        print("Score " + str(round(score_value, 6)) + " Row " + str(row_data))
        shown_count += 1

    total_rows = len(sorted_frame)
    if total_rows > max_rows_to_show:
        print("...", total_rows - max_rows_to_show, "more rows not shown")


def table_from_query(sql_text):
    text = sql_text.strip()
    text_upper = text.upper()
    from_pos = text_upper.find("FROM")
    if from_pos == -1:
        return None
    after_from = text[from_pos + len("FROM"):].strip()
    if after_from == "":
        return None
    parts = after_from.split()
    table_name = parts[0].replace('"', "").replace("'", "")
    return table_name


def guess_agg(sql_text, data_frame_result, data_frame_full):
    sum_with_alias = re.search(
        r"SUM\s*\(\s*\"?([A-Za-z0-9_ ]+)\"?\s*\)\s+AS\s+\"?([A-Za-z0-9_ ]+)\"?",
        sql_text,
        flags=re.IGNORECASE,
    )
    avg_with_alias = re.search(
        r"AVG\s*\(\s*\"?([A-Za-z0-9_ ]+)\"?\s*\)\s+AS\s+\"?([A-Za-z0-9_ ]+)\"?",
        sql_text,
        flags=re.IGNORECASE,
    )

    if sum_with_alias:
        input_col = sum_with_alias.group(1).strip()
        output_col = sum_with_alias.group(2).strip()
        if output_col in data_frame_result.columns:
            return "SUM", output_col
        if input_col in data_frame_result.columns:
            return "SUM", input_col
        if input_col in data_frame_full.columns:
            return "SUM", input_col

    if avg_with_alias:
        input_col = avg_with_alias.group(1).strip()
        output_col = avg_with_alias.group(2).strip()
        if output_col in data_frame_result.columns:
            return "AVG", output_col
        if input_col in data_frame_result.columns:
            return "AVG", input_col
        if input_col in data_frame_full.columns:
            return "AVG", input_col

    sum_no_alias = re.search(
        r"SUM\s*\(\s*\"?([A-Za-z0-9_ ]+)\"?\s*\)",
        sql_text,
        flags=re.IGNORECASE,
    )
    avg_no_alias = re.search(
        r"AVG\s*\(\s*\"?([A-Za-z0-9_ ]+)\"?\s*\)",
        sql_text,
        flags=re.IGNORECASE,
    )

    if sum_no_alias:
        col_name = sum_no_alias.group(1).strip()
        if col_name in data_frame_result.columns:
            return "SUM", col_name
        if col_name in data_frame_full.columns:
            return "SUM", col_name

    if avg_no_alias:
        col_name = avg_no_alias.group(1).strip()
        if col_name in data_frame_result.columns:
            return "AVG", col_name
        if col_name in data_frame_full.columns:
            return "AVG", col_name

    count_match = re.search(r"COUNT\s*\(", sql_text, flags=re.IGNORECASE)
    if count_match:
        for result_col in data_frame_result.columns:
            try:
                pd.to_numeric(data_frame_result[result_col].head(10), errors="raise")
                return "COUNT", result_col
            except Exception:
                continue

    for result_col in data_frame_result.columns:
        try:
            pd.to_numeric(data_frame_result[result_col].head(10), errors="raise")
            return "SUM", result_col
        except Exception:
            continue

    return "COUNT", data_frame_result.columns[0]


def where_clause(sql_text):
    text = sql_text.strip()
    text_upper = text.upper()
    where_pos = text_upper.find("WHERE")
    if where_pos == -1:
        return ""
    after_where = text[where_pos + len("WHERE"):]

    upper_after = after_where.upper()
    group_by_pos = upper_after.find("GROUP BY")
    if group_by_pos != -1:
        where_part = after_where[:group_by_pos]
    else:
        where_part = after_where

    where_part = where_part.split(";")[0]
    return where_part.strip()


def split_predicates(where_text):
    if where_text == "":
        return []
    parts = re.split(r"\bAND\b", where_text, flags=re.IGNORECASE)
    cleaned = []
    for part in parts:
        stripped = part.strip()
        if stripped != "":
            cleaned.append(stripped)
    return cleaned


def build_predicate_functions(predicate_texts, result_frame):
    predicate_funcs = []
    for text in predicate_texts:
        predicate = text.strip()

        like_match = re.match(r'"?([A-Za-z0-9_ ]+)"?\s+LIKE\s+\'(.+)\'', predicate, flags=re.IGNORECASE)
        if like_match:
            col_name = like_match.group(1).strip()
            pattern = like_match.group(2)
            if col_name not in result_frame.columns:
                continue
            if pattern.endswith("%"):
                prefix = pattern[:-1]

                def check_like_prefix(row, column_name=col_name, pref=prefix):
                    value = str(row[column_name])
                    return value.startswith(pref)

                predicate_funcs.append((col_name + " LIKE '" + pattern + "'", check_like_prefix))
                continue

        eq_match = re.match(r'"?([A-Za-z0-9_ ]+)"?\s*=\s*\'?(.+?)\'?$', predicate, flags=re.IGNORECASE)
        if eq_match:
            col_name = eq_match.group(1).strip()
            value = eq_match.group(2)
            if col_name not in result_frame.columns:
                continue

            def check_equal(row, column_name=col_name, val=value):
                return str(row[column_name]) == val

            predicate_funcs.append((col_name + " = " + str(value), check_equal))
            continue

        gt_match = re.match(r'"?([A-Za-z0-9_ ]+)"?\s*>\s*([0-9\.]+)$', predicate, flags=re.IGNORECASE)
        if gt_match:
            col_name = gt_match.group(1).strip()
            value_text = gt_match.group(2)
            if col_name not in result_frame.columns:
                continue
            try:
                threshold = float(value_text)
            except Exception:
                continue

            def check_greater(row, column_name=col_name, thresh=threshold):
                try:
                    val = float(row[column_name])
                    return val > thresh
                except Exception:
                    return False

            predicate_funcs.append((col_name + " > " + str(threshold), check_greater))
            continue

    return predicate_funcs


def predicate_scores(result_frame, row_scores_list, predicate_funcs):
    scores_for_predicates = []
    if result_frame.empty or len(row_scores_list) == 0:
        return scores_for_predicates

    for (pred_label, check_func) in predicate_funcs:
        total_score = 0.0
        for index, row in result_frame.iterrows():
            if check_func(row):
                total_score += row_scores_list[index]
        scores_for_predicates.append((pred_label, total_score))

    total_all = sum(score for (_, score) in scores_for_predicates)
    normalized = []
    if total_all > 0.0:
        for (label, score) in scores_for_predicates:
            normalized.append((label, score / total_all))
    else:
        for (label, score) in scores_for_predicates:
            normalized.append((label, 0.0))

    return normalized


def show_predicates(pred_score_list):
    print("\nPredicate-level contribution scores:")
    if not pred_score_list:
        print("No simple predicates found or matched on result columns.")
        return
    for (label, score) in pred_score_list:
        print("Predicate " + str(label) + " score " + str(round(score, 4)))


def main():
    print("Auto plan, predicate, and row contribution tool (Postgres plan + CSV)")

    plan = read_plan()
    if plan is None:
        return

    print("\nNow enter the same SQL query whose plan you explained (end with empty line):")
    query_lines = []
    while True:
        line = input()
        if line.strip() == "":
            break
        query_lines.append(line)
    sql_text = "\n".join(query_lines).strip()
    if sql_text == "":
        print("No query entered. Exiting.")
        return

    try:
        root_node = get_root(plan)
    except Exception as error:
        print("Error reading plan: " + str(error))
        return

    csv_path = input("\nPath to CSV file (for example customers-100.csv): ").strip()
    table_name = table_from_query(sql_text) or "T"

    full_frame, sqlite_connection = load_csv(csv_path, table_name)

    try:
        result_frame = run_query(sqlite_connection, sql_text)
    except Exception as error:
        print("Error running query on SQLite: " + str(error))
        sqlite_connection.close()
        return

    print("Rows returned by query on CSV: " + str(len(result_frame)))

    valid_columns = set(result_frame.columns)

    table_info = defaultdict(lambda: {"total_cost": 0.0, "row_count": 0.0})
    node_info = {}
    column_counts = defaultdict(float)

    walk_plan(root_node, table_info, node_info, column_counts, valid_columns)
    scale_tables(table_info)
    scale_nodes(node_info)
    column_scores = scale_columns(column_counts)

    show_tables(table_info)
    show_nodes(node_info, max_nodes_to_show=15)
    show_columns(column_scores, max_columns_to_show=20)

    agg_name, value_column = guess_agg(sql_text, result_frame, full_frame)
    if value_column not in result_frame.columns:
        print("\nChosen value column " + str(value_column) + " is not in result. Falling back to first column.")
        value_column = result_frame.columns[0]

    print("\nUsing aggregate " + agg_name + " on column " + str(value_column))

    scores = row_scores(result_frame, value_column, agg_name)
    show_rows(result_frame, scores, max_rows_to_show=10)

    where_text = where_clause(sql_text)
    predicate_texts = split_predicates(where_text)
    predicate_funcs = build_predicate_functions(predicate_texts, result_frame)
    pred_score_list = predicate_scores(result_frame, scores, predicate_funcs)
    show_predicates(pred_score_list)

    sqlite_connection.close()
    print("\nFinished.")


if __name__ == "__main__":
    main()
