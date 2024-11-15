from enum import Enum
from typing import List, Union

from pydantic import BaseModel


class EntityType(str, Enum):
    # Base amounts
    base_taxable_amount = "Base Taxable Amount"
    taxable_amount = "Taxable Amount"
    non_taxable_amount = "Non-Taxable Amount"
    
    # Net total components
    net_total = "Net Total"
    net_discounts = "Net Discounts"
    net_service_charge = "Net Service Charge"
    
    # Tax components
    tax_rate = "Tax Rate"
    tax = "Tax"
    
    # Gross total components
    base_gross_total = "Base Gross Total"
    gross_discounts = "Gross Discounts"
    gross_service_charge = "Gross Service Charge"
    gross_total = "Gross Total"
    rounding_adjustment = "Rounding Adjustment"
    commission_fee = "Commission Fee"
    
    # Due amounts
    due_amount = "Due Amount"
    prior_balance = "Prior Balance"
    net_due_amount = "Net Due Amount"
    
    # Payment components
    paid_amount = "Paid Amount"
    change_amount = "Change Amount"
    cash_amount = "Cash Amount"
    creditcard_amount = "Credit Card Amount"
    emoney_amount = "E-Money Amount"
    other_payments = "Other Payments"
    
    # Menu statistics
    menutype_count = "Menu Type Count"
    menuquantity_sum = "Menu Quantity Sum"
    
    # Line items
    line_item_name = "Line Item Name"
    line_item_tax_rate = "Line Item Tax Rate"
    line_item_net_unit_price = "Line Item Net Unit Price"
    line_item_unit_tax = "Line Item Unit Tax"
    line_item_gross_unit_price = "Line Item Gross Unit Price"
    line_item_quantity = "Line Item Quantity"
    line_item_net_price = "Line Item Net Price"
    line_item_tax = "Line Item Tax"
    line_item_gross_price = "Line Item Gross Price"
    line_item_net_sub_items_total = "Line Item Net Sub Items Total"
    line_item_gross_sub_items_total = "Line Item Gross Sub Items Total"
    line_item_net_total = "Line Item Net Total"
    line_item_net_discounts = "Line Item Net Discounts"
    line_item_gross_discounts = "Line Item Gross Discounts"
    line_item_total_tax = "Line Item Total Tax"
    line_item_gross_total = "Line Item Gross Total"
    
    # Sub line items
    sub_line_item_name = "Sub Line Item Name"
    sub_line_item_tax_rate = "Sub Line Item Tax Rate"
    sub_line_item_net_unit_price = "Sub Line Item Net Unit Price"
    sub_line_item_unit_tax = "Sub Line Item Unit Tax"
    sub_line_item_gross_unit_price = "Sub Line Item Gross Unit Price"
    sub_line_item_quantity = "Sub Line Item Quantity"
    sub_line_item_net_price = "Sub Line Item Net Price"
    sub_line_item_tax = "Sub Line Item Tax"
    sub_line_item_gross_price = "Sub Line Item Gross Price"

# Mapping from old entity types to potential new entity types
cord_old_to_new_entity_types = {
    "Service": "Net Service Charge",
    "Discount": "Net Discounts", 
    "Rounding": "Rounding Adjustment",
    "Commission": "Commission Fee",
    "Change": "Change Amount",
    "Cash": "Cash Amount",
    "Credit Card": "Credit Card Amount",
    "E-money": "E-Money Amount",
    "Line Item Discounts": "Line Item Net Discounts",
    "Line Item Net Sub Line Item Total": "Line Item Net Sub Items Total",
    "Line Item Gross Sub Line Item Total": "Line Item Gross Sub Items Total", 
    "Line Item Net Item Total": "Line Item Net Total",
    "Line Item Item Tax Total": "Line Item Total Tax",
    "Line Item Gross Item Total": "Line Item Gross Total",
}

sroie_old_to_new_entity_types = {
    "Service": "Gross Service Charge",
    "Discount": "Gross Discounts",
    "Rounding": "Rounding Adjustment",
    "Commission": "Commission Fee",
    "Change": "Change Amount", 
    "Cash": "Cash Amount",
    "Credit Card": "Credit Card Amount",
    "E-money": "E-Money Amount",
    "Line Item Discounts": "Line Item Gross Discounts",
    "Line Item Net Sub Line Item Total": "Line Item Net Sub Items Total",
    "Line Item Gross Sub Line Item Total": "Line Item Gross Sub Items Total",
    "Line Item Net Item Total": "Line Item Net Total",
    "Line Item Item Tax Total": "Line Item Total Tax",
    "Line Item Gross Item Total": "Line Item Gross Total",
}

non_numeric_fields = {
   EntityType.line_item_name,
   EntityType.sub_line_item_name,
}

line_item_entity_types = [
   EntityType.line_item_name,
   EntityType.line_item_tax_rate, 
   EntityType.line_item_net_unit_price,
   EntityType.line_item_unit_tax,
   EntityType.line_item_gross_unit_price,
   EntityType.line_item_quantity,
   EntityType.line_item_net_price,
   EntityType.line_item_tax,
   EntityType.line_item_gross_price,
   EntityType.line_item_net_sub_items_total,
   EntityType.line_item_gross_sub_items_total,
   EntityType.line_item_net_total,
   EntityType.line_item_net_discounts,
   EntityType.line_item_gross_discounts,
   EntityType.line_item_total_tax,
   EntityType.line_item_gross_total
]

sub_line_item_entity_types = [
   EntityType.sub_line_item_name,
   EntityType.sub_line_item_tax_rate,
   EntityType.sub_line_item_net_unit_price, 
   EntityType.sub_line_item_unit_tax,
   EntityType.sub_line_item_gross_unit_price,
   EntityType.sub_line_item_quantity,
   EntityType.sub_line_item_net_price,
   EntityType.sub_line_item_tax,
   EntityType.sub_line_item_gross_price,
]

class Data(BaseModel):
    split: str
    id: str
    page: str
    image: str

class LabelBase(BaseModel):
    x: float
    y: float
    width: float
    height: float
    rotation: float

class TextLabel(LabelBase):
    text: List[str]

class RectangleLabel(LabelBase):
    labels: List[str]

class ResultItem(BaseModel):
    id: str
    original_width: int
    original_height: int
    image_rotation: int = 0
    from_name: str
    to_name: str = "image"
    type: str
    value: Union[RectangleLabel, TextLabel, LabelBase, List[str]]

class Annotations(BaseModel):
    id: int
    result: List[ResultItem]

class Prediction(BaseModel):
   model_version: str
   result: List[ResultItem]

class Task(BaseModel):
    id: int
    data: Data
    annotations: List[Annotations] = []
    predictions: List[Prediction] = []
