import json
import os
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Union

from tqdm import tqdm

from label_studio.label_studio_model import (EntityType, Task,
                                             line_item_entity_types,
                                             sub_line_item_entity_types)
from model.document import Document
from model.invoice import Invoice, InvoiceLineItem, InvoiceSubLineItem
from utils.config import data_dir
from utils.ocr import load_ocr_result


def parse_task_to_invoice(task: Task) -> Invoice:
    # Initialize invoice fields
    invoice_fields: Dict[EntityType, List[str]] = defaultdict(list)
    line_items: Dict[str, Dict[str, Union[str, List[str]]]] = defaultdict(lambda: defaultdict(list))
    sub_line_items: Dict[str, Dict[str, Dict[str, str]]] = defaultdict(lambda: defaultdict(dict))

    # Group results by their ID
    results_by_id = defaultdict(list)
    for result in task.annotations[0].result:
        results_by_id[result.id].append(result)

    for result_group in results_by_id.values():
        bbox_result = next((r for r in result_group if r.from_name == 'bbox'), None)
        label_result = next((r for r in result_group if r.from_name == 'label'), None)
        text_result = next((r for r in result_group if r.from_name == 'transcription'), None)
        line_item_id = next((r for r in result_group if r.from_name == 'line-item-id'), None)
        sub_line_item_id = next((r for r in result_group if r.from_name == 'sub-line-item-id'), None)

        if not (bbox_result and label_result and text_result):
            continue

        entity_type = EntityType(label_result.value.labels[0])
        text_value = text_result.value.text[0]

        if entity_type in line_item_entity_types:
            # This is a line item
            item_id = line_item_id.value.text[0]
            if entity_type in [EntityType.line_item_net_discounts, EntityType.line_item_gross_discounts]:
                line_items[item_id][entity_type.value].append(text_value)
            else:
                line_items[item_id][entity_type.value] = text_value
        elif entity_type in sub_line_item_entity_types:
            # This is a sub line item
            item_id = line_item_id.value.text[0]
            sub_item_id = sub_line_item_id.value.text[0]
            sub_line_items[item_id][sub_item_id][entity_type.value] = text_value
        else:
            # This is an invoice-level field
            if entity_type in [EntityType.other_payments, EntityType.net_discounts, EntityType.gross_discounts]:
                invoice_fields[entity_type].append(text_value)
            else:
                invoice_fields[entity_type] = text_value

    # Create LineItem and SubLineItem objects
    parsed_line_items = []
    for item_id, item_data in line_items.items():
        sub_items = [
            InvoiceSubLineItem(
                name=sub_item.get(EntityType.sub_line_item_name.value, None),
                tax_rate=sub_item.get(EntityType.sub_line_item_tax_rate.value, None),
                net_unit_price=sub_item.get(EntityType.sub_line_item_net_unit_price.value, None),
                unit_tax=sub_item.get(EntityType.sub_line_item_unit_tax.value, None),
                gross_unit_price=sub_item.get(EntityType.sub_line_item_gross_unit_price.value, None),
                quantity=sub_item.get(EntityType.sub_line_item_quantity.value, None),
                net_price=sub_item.get(EntityType.sub_line_item_net_price.value, None),
                tax_amount=sub_item.get(EntityType.sub_line_item_tax.value, None),
                gross_price=sub_item.get(EntityType.sub_line_item_gross_price.value, None)
            )
            for sub_item in sub_line_items.get(item_id, {}).values()
        ]
        
        parsed_line_items.append(InvoiceLineItem(
            name=item_data.get(EntityType.line_item_name.value, None),
            tax_rate=item_data.get(EntityType.line_item_tax_rate.value, None),
            net_unit_price=item_data.get(EntityType.line_item_net_unit_price.value, None),
            unit_tax=item_data.get(EntityType.line_item_unit_tax.value, None),
            gross_unit_price=item_data.get(EntityType.line_item_gross_unit_price.value, None),
            quantity=item_data.get(EntityType.line_item_quantity.value, None),
            net_price=item_data.get(EntityType.line_item_net_price.value, None),
            tax_amount=item_data.get(EntityType.line_item_tax.value, None),
            gross_price=item_data.get(EntityType.line_item_gross_price.value, None),
            net_sub_items_total=item_data.get(EntityType.line_item_net_sub_items_total.value, None),
            gross_sub_items_total=item_data.get(EntityType.line_item_gross_sub_items_total.value, None),
            net_total=item_data.get(EntityType.line_item_net_total.value, None),
            net_discounts=item_data.get(EntityType.line_item_net_discounts.value, []),
            total_tax=item_data.get(EntityType.line_item_total_tax.value, None),
            gross_discounts=item_data.get(EntityType.line_item_gross_discounts.value, []),
            gross_total=item_data.get(EntityType.line_item_gross_total.value, None),
            sub_items=sub_items
        ))

    # Create and return the Invoice object
    invoice = Invoice(
        base_taxable_amount=invoice_fields.get(EntityType.base_taxable_amount, None),
        net_discounts=invoice_fields.get(EntityType.net_discounts, []),
        net_service_charge=invoice_fields.get(EntityType.net_service_charge, None),
        taxable_amount=invoice_fields.get(EntityType.taxable_amount, None),
        non_taxable_amount=invoice_fields.get(EntityType.non_taxable_amount, None),
        net_total=invoice_fields.get(EntityType.net_total, None),
        tax_rate=invoice_fields.get(EntityType.tax_rate, None),
        tax_amount=invoice_fields.get(EntityType.tax, None),
        base_gross_total=invoice_fields.get(EntityType.base_gross_total, None),
        gross_discounts=invoice_fields.get(EntityType.gross_discounts, []),
        gross_service_charge=invoice_fields.get(EntityType.gross_service_charge, None),
        gross_total=invoice_fields.get(EntityType.gross_total, None),
        rounding_adjustment=invoice_fields.get(EntityType.rounding_adjustment, None),
        commission_fee=invoice_fields.get(EntityType.commission_fee, None),
        due_amount=invoice_fields.get(EntityType.due_amount, None),
        prior_balance=invoice_fields.get(EntityType.prior_balance, None),
        net_due_amount=invoice_fields.get(EntityType.net_due_amount, None),
        paid_amount=invoice_fields.get(EntityType.paid_amount, None),
        change_amount=invoice_fields.get(EntityType.change_amount, None),
        cash_amount=invoice_fields.get(EntityType.cash_amount, None),
        creditcard_amount=invoice_fields.get(EntityType.creditcard_amount, None),
        emoney_amount=invoice_fields.get(EntityType.emoney_amount, None),
        other_payments=invoice_fields.get(EntityType.other_payments, []),
        menutype_count=invoice_fields.get(EntityType.menutype_count, None),
        menuquantity_sum=invoice_fields.get(EntityType.menuquantity_sum, None),
        line_items=parsed_line_items
    )
    return invoice

# Update the load_data function to return the tasks
def import_labelstudion_annotations(dataset_name: str) -> List[Task]:
    path = Path(data_dir) / dataset_name / 'labeled.json'
    with open(path, 'r') as f:
        data = json.load(f)
    
    tasks = []
    for task in data:
        task['predictions'] = []
        try:
            tasks.append(Task(**task))
        except Exception as e:
            print(f"Error parsing task {task['id']}: {e}")

    tot = 0
    invalid = 0

    splits = defaultdict(list)
    for task in tqdm(tasks):
        # For debugging a certain document
        # if task.data.id != 'validation-21-0':
        #     continue
        tot += 1
        try:
            invoice = parse_task_to_invoice(task)
            splits[task.data.split].append((task, invoice))
        except Exception as e:
            print(f"Error parsing invoice {task.data.id}: {e}")
            invalid += 1

        # Code for verifying the invoice is valid
        implicit = invoice.to_implicit()
        if not implicit.is_valid():
            invalid += 1

            constraint_results = implicit.get_constraint_sample_details()
            print(f"Invalid constraints for document {task.data.id}:")
            for constraint_name, result in constraint_results.items():
                if isinstance(result, bool) and result is False:
                    print(f"  - {constraint_name}")
                elif isinstance(result, list):
                    for i, item_result in enumerate(result):
                        if isinstance(item_result, dict):
                            for sub_constraint, sub_result in item_result.items():
                                if isinstance(sub_result, bool) and sub_result is False:
                                    print(f"  - {constraint_name}[{i}].{sub_constraint}")
            print(implicit.model_dump_json(indent=2, exclude_none=True))
            print()

    print(f"{invalid}/{tot} ({invalid/tot*100:.2f}%)")

    for split in ['train', 'validation', 'test']:
        with open(Path(data_dir) / dataset_name / f'{split}-documents.jsonl', 'w') as f:
            for task, invoice in tqdm(splits[split]):
                ocr_path = os.path.join(data_dir, dataset_name, 'ocr', task.data.id + '.jpg.json')
                ocr_result = load_ocr_result(ocr_path)

                document = Document(id=task.data.id, page_texts=[' '.join(ocr_result.words)], target=invoice)
                f.write(document.model_dump_json())
                f.write('\n')

if __name__ == '__main__':
    data = import_labelstudion_annotations('cord') # or sroie