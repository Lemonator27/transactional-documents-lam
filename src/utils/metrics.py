import dataclasses
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Type, get_args

import numpy as np
from editdistance import eval as edit_distance
from pydantic import BaseModel

from model.document import Document
from model.invoice import Invoice
from utils.json_evaluator import JSONParseEvaluator
from utils.parallel import pmap
from utils.utils import format_value


@dataclass
class Metrics:
    correct: int = 0
    coverage: int = 0
    support: int = 0
    false_pred: int = 0
    fp_support: int = 0

    def __add__(self, other: 'Metrics'):
        return Metrics(correct=self.correct + other.correct,
                       coverage=self.coverage + other.coverage,
                       support=self.support + other.support,
                       false_pred=self.false_pred + other.false_pred,
                       fp_support=self.fp_support + other.fp_support)

def compare_field(preds: List[Any], target: Optional[Any]) -> Metrics:
    m = Metrics()

    kps = [format_value(p) for p in preds] if preds is not None else []

    # Can happen in case of nested fields
    if target is None:
        m.fp_support += 1
        if any(kp is not None for kp in kps):
            m.false_pred += 1
        return m

    kt = format_value(target)
    if kt is not None:
        m.support += 1
        if kps:
            m.coverage += 1
            if kt in kps:
                m.correct += 1

    if kt is None:
        m.fp_support += 1
        if any(kp is not None for kp in kps):
            m.false_pred += 1

    return m

def compare(preds: List[BaseModel], target: Optional[BaseModel], target_type: Type = None) -> Dict[str, Metrics]:
    assert target is not None or target_type is not None, "Either target or target_type must be provided"
    target_type = type(target) if target is not None else target_type

    m = defaultdict(Metrics)
    complex_types = target_type.get_list_types()
    for field_name, field_info in target_type.model_fields.items():
        # Handle lists
        if field_name in complex_types:
            
            target_type = field_info.annotation.__args__[0].__args__[0]
            identifier_field_name = field_info.json_schema_extra.get('identifier_field_name', None)
            
            if identifier_field_name is None:
                max_list_size = max([len(getattr(p, field_name) or []) for p in preds + [target] if p is not None])
                items_to_compare = [(getattr(target, field_name)[i] if target is not None and len(getattr(target, field_name) or []) > i else None,
                                     [getattr(p, field_name)[i] for p in preds if p is not None and len(getattr(p, field_name) or []) > i])
                                    for i in range(max_list_size)]
            else:
                target_items = getattr(target, field_name) or [] if target is not None else []
                pred_items = [item for p in preds if p is not None for item in (getattr(p, field_name) or [])]
                all_ids = set([getattr(item, identifier_field_name) for item in target_items + pred_items])
                items_to_compare = [(next((item for item in target_items if getattr(item, identifier_field_name) == item_id), None),
                                     [item for p in preds if p is not None for item in (getattr(p, field_name) or []) if getattr(item, identifier_field_name) == item_id])
                                    for item_id in all_ids]

            for t, ps in items_to_compare:
                if issubclass(target_type, BaseModel):
                    nested_metrics = compare(ps, t, target_type)
                    for k, v in nested_metrics.items():
                        m[f'{field_name}.{k}'] += v
                else:
                    m[field_name] += compare_field(ps, t)
        else:
            ps = [getattr(p, field_name) for p in preds if p is not None]
            t = getattr(target, field_name) if target is not None else None
            m[field_name] += compare_field(ps, t)

    return dict(m)

def compute_infer_single(args):
    docid, target, preds = args
    target_dict = target.model_dump(exclude_none=True)
    target_implicit = target.to_implicit()
    target_implicit_dict = target_implicit.model_dump(exclude_none=True)
    
    pred_dicts = []
    pred_implicit_dicts = []
    preds_valid = []
    for pred in preds:
        pred_dict = {}
        pred_implicit_dict = {}
        if pred is None:
            preds_valid.append(False)
        else:
            pred_dict = pred.model_dump(exclude_none=True)
            pred_implicit = pred.to_implicit()
            pred_implicit_dict = pred_implicit.model_dump(exclude_none=True)
            preds_valid.append(pred_implicit.is_valid())

        pred_dicts.append(pred_dict)
        pred_implicit_dicts.append(pred_implicit_dict)
    
    return (docid, target_dict, target_implicit_dict, pred_dicts, pred_implicit_dicts, preds_valid)

def compute_metrics_single(args):
    docid, target_dict, target_implicit_dict, pred_dicts, pred_implicit_dicts = args
    json_evaluator = JSONParseEvaluator()
    
    f1s = []
    f1s_implicit = []
    nteds = []
    nteds_implicit = []
    tp_total, fn_total, fp_total = 0, 0, 0
    tp_implicit_total, fn_implicit_total, fp_implicit_total = 0, 0, 0

    for pred_dict, pred_implicit_dict in zip(pred_dicts, pred_implicit_dicts):
        f1 = json_evaluator.cal_f1([pred_dict], [target_dict])
        f1_implicit = json_evaluator.cal_f1([pred_implicit_dict], [target_implicit_dict])

        f1s.append(f1)
        f1s_implicit.append(f1_implicit)
        nteds.append(json_evaluator.cal_acc(pred_dict, target_dict))
        nteds_implicit.append(json_evaluator.cal_acc(pred_implicit_dict, target_implicit_dict))

        tp, fn, fp = json_evaluator.cal_tp_fn_fp([pred_dict], [target_dict])
        tp_total += tp
        fn_total += fn
        fp_total += fp

        tp_implicit, fn_implicit, fp_implicit = json_evaluator.cal_tp_fn_fp([pred_implicit_dict], [target_implicit_dict])
        tp_implicit_total += tp_implicit
        fn_implicit_total += fn_implicit
        fp_implicit_total += fp_implicit

    return (docid, f1s, f1s_implicit, nteds, nteds_implicit, 
            tp_total, fn_total, fp_total, 
            tp_implicit_total, fn_implicit_total, fp_implicit_total)

def compute_metrics(docid2doc: Dict[str, Document[BaseModel]], docid2preds: Dict[str, List[BaseModel]], parallelize: int = None) -> Dict[str, float]:
    # Prepare args for parallel processing
    infer_args = [(docid, doc.target, docid2preds[docid]) for docid, doc in docid2doc.items()]
    
    # Use pmap for infer computation
    infer_results = pmap(
        compute_infer_single,
        infer_args,
        num_processes=parallelize,
        desc="Computing infer results",
        use_tqdm=True
    )
    
    # Prepare metrics args
    metrics_args = [(docid, target_dict, target_implicit_dict, pred_dicts, pred_implicit_dicts) 
                   for docid, target_dict, target_implicit_dict, pred_dicts, pred_implicit_dicts, _ in infer_results]
    
    # Use pmap for metrics computation
    metrics_results = pmap(
        compute_metrics_single,
        metrics_args,
        num_processes=parallelize,
        desc="Computing metrics",
        use_tqdm=True
    )

    # Process results
    docid2f1s = {}
    docid2f1s_implicit = {}
    docid2nteds = {}
    docid2nteds_implicit = {}
    total_tp, total_fn, total_fp = 0, 0, 0
    total_tp_implicit, total_fn_implicit, total_fp_implicit = 0, 0, 0

    for result in metrics_results:
        (docid, f1s, f1s_implicit, nteds, nteds_implicit, 
         tp, fn, fp, tp_implicit, fn_implicit, fp_implicit) = result

        docid2f1s[docid] = f1s or [0.0]
        docid2f1s_implicit[docid] = f1s_implicit or [0.0]
        docid2nteds[docid] = nteds or [0.0]
        docid2nteds_implicit[docid] = nteds_implicit or [0.0]

        total_tp += tp
        total_fn += fn
        total_fp += fp
        total_tp_implicit += tp_implicit
        total_fn_implicit += fn_implicit
        total_fp_implicit += fp_implicit

    # Convert infer_results to dict for easier lookup
    infer_results_dict = {docid: (target_dict, target_implicit_dict, pred_dicts, pred_implicit_dicts, preds_valid) 
                         for docid, target_dict, target_implicit_dict, pred_dicts, pred_implicit_dicts, preds_valid in infer_results}

    # Rest of the function remains the same but uses infer_results_dict
    docid2best_f1s_i = {docid: np.argmax(f1s) for docid, f1s in docid2f1s.items()}
    docid2best_f1s_i_implicit = {docid: np.argmax(f1s) for docid, f1s in docid2f1s_implicit.items()}

    s = {}
    json_evaluator = JSONParseEvaluator()
    docid_order = list(docid2doc.keys())
    
    targets = [infer_results_dict[docid][0] for docid in docid_order]
    preds = [infer_results_dict[docid][2][docid2best_f1s_i[docid]] if docid2preds.get(docid) else {} for docid in docid_order]
    
    best_f1 = json_evaluator.cal_f1(preds, targets)
    per_field_metrics = json_evaluator.cal_per_field_metrics(preds, targets)
    s.update({f'test/{k}': v for k, v in per_field_metrics.items()})

    s['test/f1'] = best_f1
    s['test/nted'] = 1 - np.mean([np.max(docid2nteds[docid]) for docid in docid2nteds])

    targets_implicit = [infer_results_dict[docid][1] for docid in docid_order]
    preds_implicit = [infer_results_dict[docid][3][docid2best_f1s_i_implicit[docid]] if docid2preds.get(docid) else {} for docid in docid_order]
    
    best_f1_implicit = json_evaluator.cal_f1(preds_implicit, targets_implicit)
    per_field_metrics_implicit = json_evaluator.cal_per_field_metrics(preds_implicit, targets_implicit)
    s.update({f'test/{k}_implicit': v for k, v in per_field_metrics_implicit.items()})

    s['test/f1_implicit'] = best_f1_implicit
    s['test/nted_implicit'] = 1 - np.mean([np.max(docid2nteds_implicit[docid]) for docid in docid2nteds_implicit])

    best_case_valid_samples = 0
    for docid in docid_order:
        preds_valid = infer_results_dict[docid][4]
        best_case_valid_samples += int(any(preds_valid))
    s['test/doc_valid'] = best_case_valid_samples / len(docid2doc)

    docid2best_f1_implicit = {docid: any(f1 == 1.0 for f1 in f1s) for docid, f1s in docid2f1s_implicit.items()}
    s['test/doc_accuracy_implicit'] = sum(docid2best_f1_implicit.values()) / len(docid2best_f1_implicit)

    return s

@dataclass
class FieldData:
    doc: str
    sample: int
    field_path: str
    short_field_path: str

    correct: int
    edit_distance: int
    correct_implicit: int

    target: Optional[str] = None
    target_implicit: Optional[str] = None
    pred: Optional[str] = None
    pred_implicit: Optional[str] = None

    prob: Optional[float] = None
    prob_min: Optional[float] = None
    prob_max: Optional[float] = None

    target_in_text: Optional[bool] = None
    pred_in_text: Optional[bool] = None

def get_field_level_data(target: Any, target_implicit: Any, pred: Any, pred_implicit: Any, target_type: Type = None, field_path: str = "") -> List[FieldData]:
    assert target is not None or target_type is not None, "Either target or target_type must be provided"
    target_type = type(target) if target is not None else target_type

    # Handle case where target type is a simple value (not a Pydantic model)
    if not issubclass(target_type, BaseModel):
        fd = FieldData(
            doc="",  # This should be filled by the caller
            sample=0,  # This should be filled by the caller
            field_path=field_path,
            short_field_path='.'.join([part for part in field_path.split('.') if not all(char.isdigit() for char in part)]),
            target=format_value(target),
            target_implicit=format_value(target_implicit),
            pred=format_value(pred),
            pred_implicit=format_value(pred_implicit),
            correct=1 if format_value(target) == format_value(pred) else 0,
            correct_implicit=1 if format_value(target_implicit) == format_value(pred_implicit) else 0,
            edit_distance=edit_distance(format_value(target), format_value(pred)),
            target_in_text=None,
            pred_in_text=None,
            prob=None,
            prob_min=None,
            prob_max=None
        )
        return [fd]

    field_level_data = []
    for k, field_info in target_type.model_fields.items():
        if k in target_type.get_list_types():
            max_list_size = max([len(getattr(p, k)) if p is not None and getattr(p, k) is not None else 0 for p in [target, pred]])
            for i in range(max_list_size):
                nested_target = getattr(target, k)[i] if target is not None and getattr(target, k) is not None and len(getattr(target, k)) > i else None
                nested_pred = getattr(pred, k)[i] if pred is not None and getattr(pred, k) is not None and len(getattr(pred, k)) > i else None

                nested_target_implicit = getattr(target_implicit, k)[i] if target_implicit is not None and getattr(target_implicit, k) is not None and len(getattr(target_implicit, k)) > i else None
                nested_pred_implicit = getattr(pred_implicit, k)[i] if pred_implicit is not None and getattr(pred_implicit, k) is not None and len(getattr(pred_implicit, k)) > i else None
                
                nested_field_path = f'{field_path}.{k}.{i}' if len(field_path) > 0 else f'{k}.{i}'
                nested_target_type = field_info.annotation.__args__[0].__args__[0]
                nested_field_level_data = get_field_level_data(nested_target, nested_target_implicit, nested_pred, nested_pred_implicit, target_type=nested_target_type, field_path=nested_field_path)
                field_level_data.extend(nested_field_level_data)
        else:
            nested_field_path = f'{field_path}.{k}' if len(field_path) > 0 else k
            field_level_data.extend(get_field_level_data(getattr(target, k) if target is not None else None, 
                                                         getattr(target_implicit, k) if target_implicit is not None else None,
                                                         getattr(pred, k) if pred is not None else None,
                                                         getattr(pred_implicit, k) if pred_implicit is not None else None,
                                                         target_type=get_args(field_info.annotation)[0],
                                                         field_path=nested_field_path))

    return field_level_data

def compute_document_level_results_single(args):
    evaluator = JSONParseEvaluator()
    docid, target, preds, raw_logprobs = args

    target_dict = target.model_dump(exclude_none=True)
    target_implicit_dict = target.infer().model_dump(exclude_none=True)
    
    document_results = []
    for pi, pred in enumerate(preds):
        if pred is None:
            pred_dict = {}
            pred_implicit_dict = {}
        else:
            pred_dict = pred.model_dump(exclude_none=True)
            pred_implicit_dict = pred.infer().model_dump(exclude_none=True)

        sample_data = {'doc': docid, 'sample': pi}

        # Donut metrics
        sample_data['f1'] = evaluator.cal_f1([pred_dict], [target_dict])
        sample_data['nted'] = evaluator.cal_acc(pred_dict, target_dict)

        tp, fn, fp = evaluator.cal_tp_fn_fp([pred_dict], [target_dict])
        sample_data['tp'] = tp
        sample_data['fn'] = fn
        sample_data['fp'] = fp

        sample_data['all_correct'] = fn + fp == 0

        # Implicit metrics
        sample_data['f1_implicit'] = evaluator.cal_f1([pred_implicit_dict], [target_implicit_dict])
        sample_data['nted_implicit'] = evaluator.cal_acc(pred_implicit_dict, target_implicit_dict)

        tp_implicit, fn_implicit, fp_implicit = evaluator.cal_tp_fn_fp([pred_implicit_dict], [target_implicit_dict])
        sample_data['tp_implicit'] = tp_implicit
        sample_data['fn_implicit'] = fn_implicit
        sample_data['fp_implicit'] = fp_implicit

        sample_data['all_correct_implicit'] = fn_implicit + fp_implicit == 0

        sample_data['target_is_valid'] = target.to_implicit().is_valid()
        sample_data['pred_is_valid'] = pred.to_implicit().is_valid() if pred is not None else False

        constraint_details = pred.to_implicit().get_constraint_sample_details() if pred is not None else {}
        # Flatten any sub dicts in the constraint details
        for k, v in constraint_details.items():
            if isinstance(v, list):
                sample_data[f'constraint.{k}.all'] = None #not any(any(False in o.values() for o in v))
            else:
                sample_data[f'constraint.{k}'] = v

        if raw_logprobs is not None:
            sample_logprobs = [np.exp(logprob) for logprob in raw_logprobs[pi] if logprob is not None]
            sample_data['token_prob_avg'] = np.mean(sample_logprobs) if sample_logprobs else None
            sample_data['token_prob_min'] = np.min(sample_logprobs) if sample_logprobs else None
            sample_data['token_prob_max'] = np.max(sample_logprobs) if sample_logprobs else None
        
        document_results.append(sample_data)

    return document_results


def compute_document_level_results(docid2doc: Dict[str, Document[BaseModel]], 
                                   docid2preds: Dict[str, List[BaseModel]], 
                                   docid2logprobs: Dict[str, List[List[float]]] = None,
                                   parallelize: int = None) -> List[Dict[str, Any]]:
    # Prepare the data for multiprocessing
    mp_args = []
    for docid, doc in docid2doc.items():
        raw_logprobs = docid2logprobs.get(docid) if docid2logprobs else None
        
        mp_args.append((docid, doc.target, docid2preds[docid], raw_logprobs))
    
    # Use pmap for parallel processing
    results = pmap(
        compute_document_level_results_single,
        mp_args,
        num_processes=parallelize,
        desc="Computing document-level results",
        use_tqdm=True
    )

    sample_level_data = [item for sublist in results for item in sublist]
    return sample_level_data

def compute_field_level_data_single(args):
    # Instead of passing the Document object directly, we'll pass its components
    docid, target, page_texts, preds = args
    all_text = ' '.join(page_texts)
    field_level_data = []

    for sample_idx, pred in enumerate(preds):
        if pred is None:
            pred = Invoice()
            inferred_pred = Invoice()
        else:
            inferred_pred = pred.infer()
        
        field_data = get_field_level_data(target, target.infer(), pred, inferred_pred, target_type=type(target))
        
        for fd in field_data:
            fd.doc = docid
            fd.sample = sample_idx
            fd.pred_in_text = None if fd.pred is None or fd.pred == '' else fd.pred in all_text
            fd.target_in_text = None if fd.target is None or fd.target == '' else fd.target in all_text
            
            field_dict = dataclasses.asdict(fd)
            field_level_data.append(field_dict)

    return field_level_data

def compute_field_level_data(docid2doc: Dict[str, Document[BaseModel]], 
                           docid2preds: Dict[str, List[BaseModel]], 
                           parallelize: int = None) -> List[Dict[str, Any]]:
    
    # Prepare args for parallel processing - extract the needed components from Document
    mp_args = [
        (docid, doc.target, doc.page_texts, docid2preds[docid])
        for docid, doc in docid2doc.items()
    ]
    
    # Use pmap for parallel processing
    results = pmap(
        compute_field_level_data_single,
        mp_args,
        num_processes=parallelize,
        desc="Computing field-level data",
        use_tqdm=True
    )
    
    # Flatten results
    return [item for sublist in results for item in sublist]
    

if __name__ == '__main__':
    from model.dataset import load_dataset

    docs = load_dataset('cord', split='validation')
    docid2docs = {d.id: d for d in docs}
    docid2preds = {docid: [d.target] for docid, d in docid2docs.items()}
    # s = compute_metrics(docid2docs, docid2preds)
    # print(json.dumps(s, indent=2))
    # print('-' * 20)

    sample_level_data = compute_document_level_results(docid2docs, docid2preds)
    # print(json.dumps(sample_level_data, indent=2))
    print('-' * 20)

    field_level_data = compute_field_level_data(docid2docs, docid2preds)
    # print(json.dumps(field_level_data, indent=2))
