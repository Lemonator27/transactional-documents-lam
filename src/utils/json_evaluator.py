"""
This code is adapted from the Donut project by Clova AI Research.
Original source: https://github.com/clovaai/donut/blob/1.0.7/donut/util.py

The JSONParseEvaluator class provides functionality to evaluate JSON parsing accuracy
using normalized Tree Edit Distance (nTED) and F1 scores.
"""

from collections import defaultdict
from typing import Any, Dict, List, Tuple, Union

import zss
from editdistance import eval as edit_distance
from zss import Node


class JSONParseEvaluator:
    """
    Calculate n-TED(Normalized Tree Edit Distance) based accuracy and F1 accuracy score
    """

    @staticmethod
    def flatten(data: dict):
        """
        Convert Dictionary into Non-nested Dictionary
        Example:
            input(dict)
                {
                    "menu": [
                        {"name" : ["cake"], "count" : ["2"]},
                        {"name" : ["juice"], "count" : ["1"]},
                    ]
                }
            output(list)
                [
                    ("menu.name", "cake"),
                    ("menu.count", "2"),
                    ("menu.name", "juice"),
                    ("menu.count", "1"),
                ]
        """
        flatten_data = list()

        def _flatten(value, key=""):
            if type(value) is dict:
                for child_key, child_value in value.items():
                    _flatten(child_value, f"{key}.{child_key}" if key else child_key)
            elif type(value) is list:
                for value_item in value:
                    _flatten(value_item, key)
            else:
                flatten_data.append((key, value))

        _flatten(data)
        return flatten_data

    @staticmethod
    def update_cost(label1: str, label2: str):
        """
        Update cost for tree edit distance.
        If both are leaf node, calculate string edit distance between two labels (special token '<leaf>' will be ignored).
        If one of them is leaf node, cost is length of string in leaf node + 1.
        If neither are leaf node, cost is 0 if label1 is same with label2 othewise 1
        """
        label1_leaf = "<leaf>" in label1
        label2_leaf = "<leaf>" in label2
        if label1_leaf == True and label2_leaf == True:
            return edit_distance(label1.replace("<leaf>", ""), label2.replace("<leaf>", ""))
        elif label1_leaf == False and label2_leaf == True:
            return 1 + len(label2.replace("<leaf>", ""))
        elif label1_leaf == True and label2_leaf == False:
            return 1 + len(label1.replace("<leaf>", ""))
        else:
            return int(label1 != label2)

    @staticmethod
    def insert_and_remove_cost(node: Node):
        """
        Insert and remove cost for tree edit distance.
        If leaf node, cost is length of label name.
        Otherwise, 1
        """
        label = node.label
        if "<leaf>" in label:
            return len(label.replace("<leaf>", ""))
        else:
            return 1

    def normalize_dict(self, data: Union[Dict, List, Any]):
        """
        Sort by value, while iterate over element if data is list
        """
        if not data:
            return {}

        if isinstance(data, dict):
            new_data = dict()
            for key, value in sorted(data.items()):
                value = self.normalize_dict(value)
                if value:
                    if not isinstance(value, list):
                        value = [value]
                    new_data[key] = value

        elif isinstance(data, list):
            if all(isinstance(item, dict) for item in data):
                new_data = []
                for item in data:
                    item = self.normalize_dict(item)
                    if item:
                        new_data.append(item)
                new_data = sorted(new_data, key=lambda x: str(x.keys())+str(x.values()))
            else:
                new_data = sorted([str(item) for item in data if type(item) in {str, int, float} and str(item)])
        else:
            new_data = [str(data)]

        return new_data

    def cal_f1(self, preds: List[dict], answers: List[dict]) -> float:
        """
        Calculate global F1 accuracy score (field-level, micro-averaged) by counting all true positives, false negatives and false positives
        """
        total_tp, total_fn, total_fp = self.cal_tp_fn_fp(preds, answers)
        if total_tp == 0 and total_fn == 0 and total_fp == 0:
            return 0.0
        return total_tp / (total_tp + (total_fn + total_fp) / 2)
    
    def cal_tp_fn_fp(self, preds: List[dict], answers: List[dict]) -> Tuple[int, int, int]:
        total_tp, total_fn, total_fp = 0, 0, 0
        for pred, answer in zip(preds, answers):
            pred, answer = self.flatten(self.normalize_dict(pred)), self.flatten(self.normalize_dict(answer))
            for field in pred:
                if field in answer:
                    total_tp += 1
                    answer.remove(field)
                else:
                    total_fp += 1
            # False negatives are the remaining fields in answer
            total_fn += len(answer)
        return total_tp, total_fn, total_fp
    
    def cal_per_field_metrics(self, preds: List[dict], answers: List[dict]) -> Dict[str, Dict[str, Union[int, float]]]:
        field_metrics = defaultdict(lambda: {"tp": 0, "fn": 0, "fp": 0})
        
        for pred, answer in zip(preds, answers):
            pred = self.flatten(self.normalize_dict(pred))
            answer = self.flatten(self.normalize_dict(answer))
            for field in pred:
                if field in answer:
                    field_metrics[field[0]]["tp"] += 1
                    answer.remove(field)
                else:
                    field_metrics[field[0]]["fp"] += 1
            for field in answer:
                field_metrics[field[0]]["fn"] += 1
        
        # Calculate F1 score for each field
        for _, metrics in field_metrics.items():
            tp, fn, fp = metrics["tp"], metrics["fn"], metrics["fp"]
            if tp + fn + fp == 0:
                f1 = 0.0
            else:
                f1 = tp / (tp + (fn + fp) / 2.0)
            metrics["f1"] = f1
        
        out_metrics = {}
        for field, metrics in field_metrics.items():
            for k, v in metrics.items():
                out_metrics[f"{field}/{k}"] = v

        return out_metrics

    def construct_tree_from_dict(self, data: Union[Dict, List], node_name: str = None):
        """
        Convert Dictionary into Tree

        Example:
            input(dict)

                {
                    "menu": [
                        {"name" : ["cake"], "count" : ["2"]},
                        {"name" : ["juice"], "count" : ["1"]},
                    ]
                }

            output(tree)
                                     <root>
                                       |
                                     menu
                                    /    \
                             <subtree>  <subtree>
                            /      |     |      \
                         name    count  name    count
                        /         |     |         \
                  <leaf>cake  <leaf>2  <leaf>juice  <leaf>1
         """
        if node_name is None:
            node_name = "<root>"

        node = Node(node_name)

        if isinstance(data, dict):
            for key, value in data.items():
                kid_node = self.construct_tree_from_dict(value, key)
                node.addkid(kid_node)
        elif isinstance(data, list):
            if all(isinstance(item, dict) for item in data):
                for item in data:
                    kid_node = self.construct_tree_from_dict(
                        item,
                        "<subtree>",
                    )
                    node.addkid(kid_node)
            else:
                for item in data:
                    node.addkid(Node(f"<leaf>{item}"))
        else:
            raise Exception(data, node_name)
        return node

    def cal_acc(self, pred: dict, answer: dict) -> float:
        """
        Calculate normalized tree edit distance(nTED) based accuracy.
        1) Construct tree from dict,
        2) Get tree distance with insert/remove/update cost,
        3) Divide distance with GT tree size (i.e., nTED),
        4) Calculate nTED based accuracy. (= max(1 - nTED, 0 ).
        """
        pred = self.construct_tree_from_dict(self.normalize_dict(pred))
        answer = self.construct_tree_from_dict(self.normalize_dict(answer))
        return max(
            0,
            1
            - (
                zss.distance(
                    pred,
                    answer,
                    get_children=zss.Node.get_children,
                    insert_cost=self.insert_and_remove_cost,
                    remove_cost=self.insert_and_remove_cost,
                    update_cost=self.update_cost,
                    return_operations=False,
                )
                / zss.distance(
                    self.construct_tree_from_dict(self.normalize_dict({})),
                    answer,
                    get_children=zss.Node.get_children,
                    insert_cost=self.insert_and_remove_cost,
                    remove_cost=self.insert_and_remove_cost,
                    update_cost=self.update_cost,
                    return_operations=False,
                )
            ),
        )

if __name__ == "__main__":
    evaluator = JSONParseEvaluator()
    pred = {'menu': [{'name': ['cake'], 'count': '3'}, {'name': ['juice'], 'count': ['1']}]}
    answer = {'menu': [{'name': ['cake'], 'count': ['2']}, {'name': ['juice'], 'count': ['1']}]}
    print(evaluator.cal_acc(pred, answer))
    print(evaluator.cal_f1(pred, answer))
    print(evaluator.cal_per_field_metrics(pred, answer))