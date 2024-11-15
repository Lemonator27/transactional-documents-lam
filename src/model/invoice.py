from decimal import Decimal
from typing import Any, Dict, List, Optional, Set

import numpy as np
from pydantic import BaseModel, Field

from model.constraints import (BaseConstraint, BoolConstraint,
                               ImplicitBaseModel, IsCloseConstraint,
                               solve_equation)
from utils.utils import (format_value, parse_cnt_decimal, parse_decimal,
                         parse_int)


class Invoice(BaseModel):

    # Taxable components
    base_taxable_amount: Optional[str] = Field(None, description='The base amount that is subject to tax')

    net_discounts: Optional[List[str]] = Field(None, description='Discounts applied to taxable amount before tax calculation', json_schema_extra={'unordered': True})
    net_service_charge: Optional[str] = Field(None, description='Service charge applied to taxable amount before tax calculation')
    taxable_amount: Optional[str] = Field(None, description='The amount that is subject to tax. This is the base amount plus net discounts and net service charges')
    
    # Non-taxable component
    non_taxable_amount: Optional[str] = Field(None, description='The base amount that is not subject to tax')
    
    # Combined base total
    net_total: Optional[str] = Field(None, description='Sum of taxable and non-taxable amounts with their modifiers')

    # Tax calculation
    tax_rate: Optional[str] = Field(None, description='Tax rate percentage applied to taxable amount')
    tax_amount: Optional[str] = Field(None, description='Total amount of tax on the invoice')

    # Net total modifiers (applied after tax)
    base_gross_total: Optional[str] = Field(None, description='The base amount that is subject to gross discounts and service charges')
    gross_discounts: Optional[List[str]] = Field(None, description='Discounts applied to entire net total after tax', json_schema_extra={'unordered': True})
    gross_service_charge: Optional[str] = Field(None, description='Service charge applied to entire net total after tax')
    
    # Final amounts
    gross_total: Optional[str] = Field(None, description='Final amount after all taxes and modifications')
    rounding_adjustment: Optional[str] = Field(None, description='Amount added/subtracted to round to desired precision')
    commission_fee: Optional[str] = Field(None, description='Commission amount deducted from total')

    # Due amount
    due_amount: Optional[str] = Field(None, description='The amount due for the transaction before considering prior balance')

    # Prior balance
    prior_balance: Optional[str] = Field(None, description='Previous balance or credit applied to the current transaction')
    net_due_amount: Optional[str] = Field(None, description='The final amount due after applying prior balance')

    # Paid amount
    paid_amount: Optional[str] = Field(None, description='The total amount paid by the customer')

    # Change
    change_amount: Optional[str] = Field(None, description='The amount returned to the customer if overpayment occurred')
    cash_amount: Optional[str] = Field(None, description='The amount paid in cash')
    creditcard_amount: Optional[str] = Field(None, description='The amount paid by credit card')
    emoney_amount: Optional[str] = Field(None, description='The amount paid using electronic money')
    other_payments: Optional[List[str]] = Field(None, description='Amounts paid using other methods (e.g., coupons, vouchers)', json_schema_extra={'unordered': True})

    # NOTE: Order matters
    menutype_count: Optional[str] = Field(None, description='The number of distinct menu item types in the order')
    menuquantity_sum: Optional[str]  = Field(None, description='The total quantity of all menu items ordered')
    line_items: Optional[List['InvoiceLineItem']] = Field(None, description='Detailed list of individual items in the order', json_schema_extra={'identifier_field_name': 'nm'})

    @classmethod
    def get_list_types(cls) -> Set[str]:
        return {'line_items', 'other_payments', 'net_discounts', 'gross_discounts'}
    
    def infer(self) -> 'Invoice':
        # Infer by going back and forth between explicit and implicit
        a = InvoiceImplicit.from_explicit(self)
        return Invoice.from_implicit(a)
    
    @staticmethod
    def from_implicit(x: 'InvoiceImplicit') -> 'Invoice':
        y = Invoice()
        y.base_taxable_amount = format_value(x.base_taxable_amount, value_if_none=None)
        y.non_taxable_amount = format_value(x.non_taxable_amount, value_if_none=None)
        y.taxable_amount = format_value(x.taxable_amount, value_if_none=None)
        y.net_total = format_value(x.net_total, value_if_none=None)
        y.net_discounts = [format_value(d, value_if_none=None) for d in x.net_discounts] if x.net_discounts else None
        y.net_service_charge = format_value(x.net_service_charge, value_if_none=None)
        y.tax_rate = format_value(x.tax_rate, value_if_none=None, decimals=3)
        y.tax_amount = format_value(x.tax_amount, value_if_none=None)
        y.gross_discounts = [format_value(d, value_if_none=None) for d in x.gross_discounts] if x.gross_discounts else None
        y.gross_service_charge = format_value(x.gross_service_charge, value_if_none=None)
        y.gross_total = format_value(x.gross_total, value_if_none=None)
        y.base_gross_total = format_value(x.base_gross_total, value_if_none=None)
        y.rounding_adjustment = format_value(x.rounding_adjustment, value_if_none=None)
        y.commission_fee = format_value(x.commission_fee, value_if_none=None)
        y.due_amount = format_value(x.due_amount, value_if_none=None)
        y.prior_balance = format_value(x.prior_balance, value_if_none=None)
        y.net_due_amount = format_value(x.net_due_amount, value_if_none=None)
        y.paid_amount = format_value(x.paid_amount, value_if_none=None)
        y.change_amount = format_value(x.change_amount, value_if_none=None)
        y.cash_amount = format_value(x.cash_amount, value_if_none=None)
        y.creditcard_amount = format_value(x.creditcard_amount, value_if_none=None)
        y.emoney_amount = format_value(x.emoney_amount, value_if_none=None)
        y.menutype_count = format_value(x.menutype_count, value_if_none=None)
        y.menuquantity_sum = format_value(x.menuquantity_sum, value_if_none=None)
        y.other_payments = [format_value(payment, value_if_none=None) for payment in x.other_payments] if x.other_payments else None
        y.line_items = [InvoiceLineItem.from_implicit(item) for item in x.line_items or []] if x.line_items else None
        return y
    
    def to_implicit(self) -> 'InvoiceImplicit':
        return InvoiceImplicit.from_explicit(self)
    
class InvoiceLineItem(BaseModel):
    # Basic info
    name: Optional[str] = Field(None, description='The name of the menu item')

    # Unit pricing
    net_unit_price: Optional[str] = Field(None, description='The unit price before tax')
    # tax_rate: Optional[str] = Field(None, description='Tax rate percentage applied to this item') # Never present, only infer
    unit_tax: Optional[str] = Field(None, description='Tax amount per unit')
    gross_unit_price: Optional[str] = Field(None, description='Unit price including tax')

    # Quantity
    quantity: Optional[str] = Field(None, description='Quantity ordered (can be decimal for weights/volumes/litres)')

    # Base item totals
    net_price: Optional[str] = Field(None, description='Total price before tax (quantity × net_unit_price)')
    tax_amount: Optional[str] = Field(None, description='Total tax amount (quantity × unit_tax)')
    gross_price: Optional[str] = Field(None, description='Total price including tax')

    # Sub-items (e.g., toppings, modifications)
    sub_items: Optional[List['InvoiceSubLineItem']] = Field(None, description='Additional components or modifications', json_schema_extra={'identifier_field_name': 'nm'})
    net_sub_items_total: Optional[str] = Field(None, description='Total price of all sub-items before tax')
    gross_sub_items_total: Optional[str] = Field(None, description='Total price of all sub-items including tax')

    # Combined totals with modifiers
    net_total: Optional[str] = Field(None, description='Combined net price of item and sub-items before discounts')
    net_discounts: Optional[List[str]] = Field(None, description='Discounts applied to net total of this item', json_schema_extra={'unordered': True})
    total_tax: Optional[str] = Field(None, description='Combined tax amount for item and sub-items')
    gross_discounts: Optional[List[str]] = Field(None, description='Discounts applied to the gross total of this item', json_schema_extra={'unordered': True})
    gross_total: Optional[str] = Field(None, description='Final price including tax and after discounts')

    @classmethod
    def get_list_types(cls) -> Set[str]:
        return {'sub_items', 'net_discounts', 'gross_discounts'}
    
    @staticmethod
    def from_implicit(x: 'InvoiceImplicitLineItem') -> 'InvoiceLineItem':
        y = InvoiceLineItem()
        y.name = format_value(x.name, value_if_none=None)
        y.net_unit_price = format_value(x.net_unit_price, value_if_none=None)
        # y.tax_rate = format_value(x.tax_rate, value_if_none=None, decimals=3)
        y.unit_tax = format_value(x.unit_tax, value_if_none=None)
        y.gross_unit_price = format_value(x.gross_unit_price, value_if_none=None)
        y.quantity = format_value(x.quantity, value_if_none=None)
        y.net_price = format_value(x.net_price, value_if_none=None)
        y.tax_amount = format_value(x.tax_amount, value_if_none=None)
        y.gross_price = format_value(x.gross_price, value_if_none=None)
        y.sub_items = [InvoiceSubLineItem.from_implicit(sub) for sub in x.sub_items or []] if x.sub_items else None
        y.net_sub_items_total = format_value(x.net_sub_items_total, value_if_none=None)
        y.gross_sub_items_total = format_value(x.gross_sub_items_total, value_if_none=None)
        y.net_total = format_value(x.net_total, value_if_none=None)
        y.net_discounts = [format_value(d, value_if_none=None) for d in x.net_discounts] if x.net_discounts else None
        y.total_tax = format_value(x.total_tax, value_if_none=None)
        y.gross_discounts = [format_value(d, value_if_none=None) for d in x.gross_discounts] if x.gross_discounts else None
        y.gross_total = format_value(x.gross_total, value_if_none=None)
        return y

class InvoiceSubLineItem(BaseModel):
    name: Optional[str] = Field(None, description='The name of the sub-item or modification')

    # tax_rate: Optional[str] = Field(None, description='The tax rate applied to the sub-item') # Never present, only infer

    net_unit_price: Optional[str] = Field(None, description='The unit price of the sub-item before tax')
    unit_tax: Optional[str] = Field(None, description='The tax amount per unit of the sub-item')
    gross_unit_price: Optional[str] = Field(None, description='The unit price of the sub-item including tax')

    quantity: Optional[str] = Field(None, description='The quantity of the sub-item (can be a decimal for items sold by weight or volume)')

    net_price: Optional[str] = Field(None, description='The total price of the sub-item before tax')
    tax_amount: Optional[str] = Field(None, description='The total tax amount for the sub-item')
    gross_price: Optional[str] = Field(None, description='The total price of the sub-item including tax')

    @classmethod
    def get_list_types(cls) -> Set[str]:
        return {}
    
    @ staticmethod
    def from_implicit(x: 'InvoiceImplicitSubLineItem') -> 'InvoiceSubLineItem':
        y = InvoiceSubLineItem()
        y.name = format_value(x.name, value_if_none=None)
        # y.tax_rate = format_value(x.tax_rate, value_if_none=None, decimals=3)
        y.net_unit_price = format_value(x.net_unit_price, value_if_none=None)
        y.unit_tax = format_value(x.unit_tax, value_if_none=None)
        y.gross_unit_price = format_value(x.gross_unit_price, value_if_none=None)
        y.quantity = format_value(x.quantity, value_if_none=None)
        y.net_price = format_value(x.net_price, value_if_none=None)
        y.tax_amount = format_value(x.tax_amount, value_if_none=None)
        y.gross_price = format_value(x.gross_price, value_if_none=None)
        return y
    

# ************** IMPLICIT **************
class InvoiceImplicit(ImplicitBaseModel):

    base_taxable_amount: Optional[Decimal] = Field(None, description='The base amount that is subject to tax')

    net_discounts: Optional[List[Decimal]] = Field(None, description='Discounts applied to taxable amount before tax calculation')
    net_service_charge: Optional[Decimal] = Field(None, description='Service charge applied to taxable amount before tax calculation')
    taxable_amount: Optional[Decimal] = Field(None, description='The amount that is subject to tax. This is the base amount plus net discounts and net service charges')
    
    # Non-taxable component
    non_taxable_amount: Optional[Decimal] = Field(None, description='The base amount that is not subject to tax')
    
    # Combined base total
    net_total: Optional[Decimal] = Field(None, description='Sum of taxable and non-taxable amounts with their modifiers')

    # Tax calculation
    tax_rate: Optional[Decimal] = Field(None, description='Tax rate percentage applied to modified taxable amount')
    tax_amount: Optional[Decimal] = Field(None, description='Tax amount calculated from taxable amount and tax rate')

    # Net total modifiers (applied after tax)
    base_gross_total: Optional[Decimal] = Field(None, description='The base amount that is subject to gross discounts and service charges')
    gross_discounts: Optional[List[Decimal]] = Field(None, description='Discounts applied to entire net total after tax', json_schema_extra={'unordered': True})
    gross_service_charge: Optional[Decimal] = Field(None, description='Service charge applied to entire net total after tax')
    gross_total: Optional[Decimal] = Field(None, description='Final amount after all taxes and modifications')

    rounding_adjustment: Optional[Decimal] = Field(None, description='Amount added/subtracted to round to desired precision')
    commission_fee: Optional[Decimal] = Field(None, description='Commission amount deducted from total')
    due_amount: Optional[Decimal] = Field(None, description='The amount due for the transaction before considering prior balance')

    prior_balance: Optional[Decimal] = Field(None, description='Previous balance or credit applied to the current transaction')
    net_due_amount: Optional[Decimal] = Field(None, description='The final amount due after applying prior balance')

    # Paid amount
    paid_amount: Optional[Decimal] = Field(None, description='The total amount paid by the customer')

    # Change
    change_amount: Optional[Decimal] = Field(None, description='The amount returned to the customer if overpayment occurred')
    cash_amount: Optional[Decimal] = Field(None, description='The amount paid in cash')
    creditcard_amount: Optional[Decimal] = Field(None, description='The amount paid by credit card')
    emoney_amount: Optional[Decimal] = Field(None, description='The amount paid using electronic money')
    other_payments: Optional[List[Decimal]] = Field(None, description='Amounts paid using other methods (e.g., coupons, vouchers)', json_schema_extra={'unordered': True})

    menutype_count: Optional[int] = Field(None, description='The total count of the types of menu items')
    menuquantity_sum: Optional[Decimal]  = Field(None, description='The total count of the quantities of menu items')
    line_items: Optional[List['InvoiceImplicitLineItem']] = Field(None, description='The list of menu items', json_schema_extra={'identifier_field_name': 'nm'})

    @staticmethod
    def get_list_types() -> Set[str]:
        return {'line_items', 'other_payments'}

    def get_constraints(self) -> List[BaseConstraint]:
        constraints = [
            # Net & gross amounts and discounts
            IsCloseConstraint("taxable_amount=base_taxable_amount-sum(net_discounts)+net_service_charge",
                      lambda m: (m.taxable_amount, m.base_taxable_amount - sum(m.net_discounts or []) + m.net_service_charge)),
            IsCloseConstraint("net_total=taxable_amount+non_taxable_amount",
                      lambda m: (m.net_total, m.taxable_amount + m.non_taxable_amount)),
            IsCloseConstraint("tax_amount=taxable_amount*tax_rate",
                      lambda m: (m.tax_amount, m.taxable_amount * m.tax_rate)),
            IsCloseConstraint("base_gross_total=net_total+tax_amount",
                      lambda m: (m.base_gross_total, m.net_total + m.tax_amount)),
            IsCloseConstraint("gross_total=base_gross_total-sum(gross_discounts)+gross_service_charge",
                      lambda m: (m.gross_total, m.base_gross_total - sum(m.gross_discounts or []) + m.gross_service_charge)),

            # Due amounts
            IsCloseConstraint("due_amount=gross_total-commission_fee+/-rounding_adjustment", 
                       lambda m: (m.due_amount, min(
                           m.gross_total - (m.commission_fee or 0) - (m.rounding_adjustment or 0),
                           m.gross_total - (m.commission_fee or 0) + (m.rounding_adjustment or 0)
                       ))),
            IsCloseConstraint("net_due_amount=due_amount+prior_balance", 
                       lambda m: (m.net_due_amount, m.due_amount + (m.prior_balance or 0))),
            IsCloseConstraint("paid_amount=net_due_amount", 
                       lambda m: (m.paid_amount, m.net_due_amount) if m.paid_amount != 0 or m.net_due_amount != 0 else (0, 1)),
            IsCloseConstraint("paid_amount=cash_amount-change_amount+creditcard_amount+emoney_amount+sum(other_payments) or 0",
                       lambda m: (m.paid_amount, (m.cash_amount or 0) - (m.change_amount or 0) + (m.creditcard_amount or 0) + 
                                (m.emoney_amount or 0) + sum(m.other_payments or []))
                       if ((m.cash_amount or 0) + (m.change_amount or 0) + (m.creditcard_amount or 0) + 
                           (m.emoney_amount or 0) + sum(m.other_payments or [])) > 0 else (0, 0)),
            BoolConstraint("due_amount!=0", lambda m: m.due_amount != 0),
            BoolConstraint("0<=tax_rate<=1", lambda m: 0 <= m.tax_rate <= 1),

            # Line item constraints
            # All line item tax rates should be 0 or the same
            # Warning: This constraint is not always true, as there can be line items that are not taxable
            BoolConstraint("all(line_items.tax_rate==0 or line_items.tax_rate==line_items[0].tax_rate)",
                       lambda m: all(item.tax_rate == 0 or np.isclose(item.tax_rate, m.line_items[0].tax_rate, rtol=1e-2) for item in m.line_items) if m.line_items else True),
            BoolConstraint("len(line_items)>0", lambda m: len(m.line_items or []) > 0),
            IsCloseConstraint("base_taxable_amount=sum(line_items.net_total)",
                       lambda m: (m.base_taxable_amount, sum(item.net_total for item in m.line_items) if m.line_items else None)),
            IsCloseConstraint("base_gross_total=sum(line_items.gross_total)",
                       lambda m: (m.base_gross_total, sum(item.gross_total for item in m.line_items) if m.line_items else None)),
            IsCloseConstraint("tax_amount=sum(line_items.total_tax)",
                       lambda m: (m.tax_amount, sum(item.total_tax for item in m.line_items) if m.line_items else None)),
            IsCloseConstraint("menutype_count=len(line_items)", 
                       lambda m: (m.menutype_count, len(m.line_items) if m.line_items else None)),
            IsCloseConstraint("menuquantity_sum=sum(line_items.quantity)", 
                       lambda m: (m.menuquantity_sum, sum(item.quantity for item in m.line_items) if m.line_items else None)),
        ]
        return constraints
    
    @staticmethod
    def from_explicit(x: Invoice) -> 'InvoiceImplicit':
        y = InvoiceImplicit()

        y.base_taxable_amount = parse_decimal(x.base_taxable_amount)
        y.taxable_amount = parse_decimal(x.taxable_amount)
        y.non_taxable_amount = parse_decimal(x.non_taxable_amount) or Decimal(0)
        y.net_discounts = [parse_decimal(d) for d in x.net_discounts] if x.net_discounts else []
        y.net_service_charge = parse_decimal(x.net_service_charge) or Decimal(0)
        y.net_total = parse_decimal(x.net_total)
        y.gross_discounts = [parse_decimal(d) for d in x.gross_discounts] if x.gross_discounts else []
        y.gross_service_charge = parse_decimal(x.gross_service_charge) or Decimal(0)
        y.gross_total = parse_decimal(x.gross_total)
        y.base_gross_total = parse_decimal(x.base_gross_total)
        y.tax_amount = parse_decimal(x.tax_amount)
        y.tax_rate = parse_decimal(x.tax_rate, is_percentage=True)
        y.rounding_adjustment = parse_decimal(x.rounding_adjustment) or Decimal(0)
        y.commission_fee = parse_decimal(x.commission_fee) or Decimal(0)
        y.due_amount = parse_decimal(x.due_amount)
        y.prior_balance = parse_decimal(x.prior_balance) or Decimal(0)
        y.net_due_amount = parse_decimal(x.net_due_amount)
        y.paid_amount = parse_decimal(x.paid_amount)
        y.change_amount = parse_decimal(x.change_amount) or Decimal(0)
        y.cash_amount = parse_decimal(x.cash_amount) or Decimal(0)
        y.creditcard_amount = parse_decimal(x.creditcard_amount) or Decimal(0)
        y.emoney_amount = parse_decimal(x.emoney_amount) or Decimal(0)
        y.other_payments = [parse_decimal(p) for p in x.other_payments] if x.other_payments else []
        y.other_payments = [abs(p) for p in y.other_payments if p is not None]
        y.menutype_count = parse_int(x.menutype_count)
        y.menuquantity_sum = parse_cnt_decimal(x.menuquantity_sum)

        if y.creditcard_amount < 0:
            y.creditcard_amount = abs(y.creditcard_amount)
        if y.emoney_amount < 0:
            y.emoney_amount = abs(y.emoney_amount)

        inferred_items = []
        for item in x.line_items or []:
            inferred_items.append(InvoiceImplicitLineItem.from_explicit(item))
        y.line_items = inferred_items

        if y.menutype_count is None:
            y.menutype_count = len(y.line_items)
        if y.menuquantity_sum is None:
            y.menuquantity_sum = Decimal(sum(item.quantity for item in y.line_items)) if all(item.quantity is not None for item in y.line_items) and len(y.line_items) > 0 else None
        
        if y.base_taxable_amount is None:
            y.base_taxable_amount = Decimal(sum(item.net_total or 0 for item in y.line_items)) if any(item.net_total is not None for item in y.line_items) and len(y.line_items) > 0 else None
        
        if y.base_gross_total is None:
            y.base_gross_total = Decimal(sum(item.gross_total or 0 for item in y.line_items)) if any(item.gross_total is not None for item in y.line_items) and len(y.line_items) > 0 else None

        # Make sure these values are positive and not None
        y.change_amount = abs(y.change_amount) if y.change_amount is not None else None
        y.gross_discounts = [abs(d) for d in y.gross_discounts if d is not None] if y.gross_discounts else []
        y.net_discounts = [abs(d) for d in y.net_discounts if d is not None] if y.net_discounts else []
        y.other_payments = [abs(p) for p in y.other_payments if p is not None] if y.other_payments else []
        y.commission_fee = abs(y.commission_fee) if y.commission_fee is not None else Decimal(0)

        i = 0
        values = y.solvable_fields()
        while True:
            i += 1
            y.solve()
            if y.is_complete():
                break

            new_values = y.solvable_fields()
            if new_values == values:
                # Break if we have no changes
                break
            
            values = new_values
        
        # Round the tax rate to two decimals
        # Causes some issues for large amounts. The formatting should take care of this
        # if y.tax_rate is not None:
        #     y.tax_rate = round(y.tax_rate, 3) # 1 decimal when percentage
            
        return y
    
    def solvable_fields(self) -> List[Optional[Decimal]]:
        return [self.base_taxable_amount,
                self.taxable_amount,
                self.tax_rate,
                self.net_total,
                self.tax_amount,
                self.base_gross_total,
                self.gross_total,
                self.due_amount,
                self.net_due_amount,
                self.paid_amount]
    
    def is_complete(self) -> bool:
        return all(x is not None for x in self.solvable_fields())

    def solve(self):
        equations = [
            "taxable_amount = base_taxable_amount - net_discounts + net_service_charge",
            "net_total = taxable_amount + non_taxable_amount",
            "tax_amount = taxable_amount * tax_rate",
            "base_gross_total = net_total + tax_amount",
            "gross_total = base_gross_total - gross_discounts + gross_service_charge",
            "due_amount = gross_total - commission_fee + rounding_adjustment", # TODO: Handle bidirectional rounding
            "net_due_amount = due_amount + prior_balance",
            "paid_amount = net_due_amount",
            "paid_amount = cash_amount - change_amount + creditcard_amount + emoney_amount + other_payments"
        ]

        for eq in equations:
            solve_equation(eq, self)

class InvoiceImplicitLineItem(ImplicitBaseModel):
    name: Optional[str] = Field(None, description='The name of the line item')
    
    tax_rate: Optional[Decimal] = Field(None, description='The tax rate of the line item')

    net_unit_price: Optional[Decimal] = Field(None, description='The unit price of the line item without tax')
    unit_tax: Optional[Decimal] = Field(None, description='The tax amount of the line item per unit')
    gross_unit_price: Optional[Decimal] = Field(None, description='The unit price of the line item including tax')

    quantity: Optional[Decimal] = Field(None, description='The quantity of the line item. Can be a decimal when the quantity is for example liters')

    net_price: Optional[Decimal] = Field(None, description='The net price of the line item excluding subitems and discounts')
    tax_amount: Optional[Decimal] = Field(None, description='The tax amount of the line item')
    gross_price: Optional[Decimal] = Field(None, description='The gross price of the line item excluding subitems and discounts')

    sub_items: Optional[List['InvoiceImplicitSubLineItem']] = Field(None, description='The sub elements of this line item such as toppings or ingredients')

    net_sub_items_total: Optional[Decimal] = Field(None, description='The net total of all the sub items')
    gross_sub_items_total: Optional[Decimal] = Field(None, description='The gross total of all the sub items including tax')
    tax_sub_items_total: Optional[Decimal] = Field(None, description='The total tax amount of all the sub items')

    net_discounts: Optional[List[Decimal]] = Field(None, description='Any discounts applied to the net total of the line item')
    net_total: Optional[Decimal] = Field(None, description='The net total of the line item, including sub items and discounts')

    total_tax: Optional[Decimal] = Field(None, description='The tax amount of the line item and its sub items')
    gross_discounts: Optional[List[Decimal]] = Field(None, description='Any discounts applied to the gross total of the line item')
    gross_total: Optional[Decimal] = Field(None, description='The gross total of the line item and its sub items, including tax and discounts')
    
    def get_constraints(self) -> List[BaseConstraint]:
        return [
            # Basic price calculations
            IsCloseConstraint("net_price=net_unit_price*quantity", 
                      lambda m: (m.net_price, m.net_unit_price * m.quantity)),
            IsCloseConstraint("tax_amount=unit_tax*quantity", 
                      lambda m: (m.tax_amount, m.unit_tax * m.quantity)),
            IsCloseConstraint("gross_price=gross_unit_price*quantity", 
                      lambda m: (m.gross_price, m.gross_unit_price * m.quantity)),
            IsCloseConstraint("gross_unit_price=net_unit_price+unit_tax",
                      lambda m: (m.gross_unit_price, m.net_unit_price + m.unit_tax)),
            IsCloseConstraint("gross_price=net_price+tax_amount",
                      lambda m: (m.gross_price, m.net_price + m.tax_amount)),
            IsCloseConstraint("tax_rate=unit_tax/net_unit_price",
                      lambda m: (m.tax_rate, m.unit_tax / m.net_unit_price) if m.net_unit_price != 0 else (None, None)),
            IsCloseConstraint("unit_tax=net_unit_price*tax_rate",
                      lambda m: (m.unit_tax, m.net_unit_price * m.tax_rate)),
            BoolConstraint("tax_rate < 1", lambda m: m.tax_rate < 1),
            
            # Total calculations
            IsCloseConstraint("net_total=net_price+net_sub_items_total-sum(net_discounts)", 
                      lambda m: (m.net_total, m.net_price + m.net_sub_items_total - sum(m.net_discounts or []))),
            IsCloseConstraint("total_tax=tax_amount+tax_sub_items_total", 
                      lambda m: (m.total_tax, m.tax_amount + m.tax_sub_items_total)),
            IsCloseConstraint("gross_total=net_total+total_tax-sum(gross_discounts)",
                      lambda m: (m.gross_total, m.net_total + m.total_tax - sum(m.gross_discounts or []))),
            BoolConstraint("gross_total or net_total not None", lambda m: m.gross_total is not None or m.net_total is not None),

            # Sub item totals
            IsCloseConstraint("net_sub_items_total=sum(sub_items.net_price)",
                      lambda m: (m.net_sub_items_total, sum(sub.net_price for sub in m.sub_items or []))),
            IsCloseConstraint("gross_sub_items_total=sum(sub_items.gross_price)", 
                      lambda m: (m.gross_sub_items_total, sum(sub.gross_price for sub in m.sub_items or []))),
            IsCloseConstraint("tax_sub_items_total=sum(sub_items.tax_amount)",
                      lambda m: (m.tax_sub_items_total, sum(sub.tax_amount for sub in m.sub_items or [])))
        ]
    
    @staticmethod
    def get_list_types() -> Set[str]:
        return {'sub_items', 'net_discounts', 'gross_discounts'}  # Updated to match field names

    @staticmethod
    def from_explicit(x: InvoiceLineItem) -> 'InvoiceImplicitLineItem':
        y = InvoiceImplicitLineItem()

        # String fields
        y.name = x.name

        # Parse fields
        # y.tax_rate = parse_decimal(x.tax_rate, is_percentage=True)  # Never present, only inferred
        y.net_unit_price = parse_decimal(x.net_unit_price)
        y.unit_tax = parse_decimal(x.unit_tax)
        y.gross_unit_price = parse_decimal(x.gross_unit_price)
        y.quantity = parse_cnt_decimal(x.quantity)
        y.net_price = parse_decimal(x.net_price)
        y.tax_amount = parse_decimal(x.tax_amount)  # Fixed field name
        y.gross_price = parse_decimal(x.gross_price)
        y.net_sub_items_total = parse_decimal(x.net_sub_items_total)
        y.gross_sub_items_total = parse_decimal(x.gross_sub_items_total)
        y.net_total = parse_decimal(x.net_total)
        y.total_tax = parse_decimal(x.total_tax)
        y.gross_total = parse_decimal(x.gross_total)

        # List fields
        y.sub_items = [InvoiceImplicitSubLineItem.from_explicit(sub) for sub in x.sub_items or []] if x.sub_items else []
        y.net_discounts = [parse_decimal(d) for d in x.net_discounts] if x.net_discounts else []
        y.gross_discounts = [parse_decimal(d) for d in x.gross_discounts] if x.gross_discounts else []

        # Make sure discounts are positive
        if y.net_discounts:
            y.net_discounts = [abs(d) for d in y.net_discounts if d is not None]
        if y.gross_discounts:
            y.gross_discounts = [abs(d) for d in y.gross_discounts if d is not None]

        # Calculate sub item totals if not provided
        if y.net_sub_items_total is None:
            y.net_sub_items_total = Decimal(sum(sub.net_price for sub in y.sub_items)) if all(sub.net_price is not None for sub in y.sub_items) and len(y.sub_items) > 0 else None
            y.net_sub_items_total = Decimal(0) if y.net_sub_items_total is None else y.net_sub_items_total
        
        if y.gross_sub_items_total is None:
            y.gross_sub_items_total = Decimal(sum(sub.gross_price for sub in y.sub_items)) if all(sub.gross_price is not None for sub in y.sub_items) and len(y.sub_items) > 0 else None
            y.gross_sub_items_total = Decimal(0) if y.gross_sub_items_total is None else y.gross_sub_items_total
        
        if y.tax_sub_items_total is None:
            y.tax_sub_items_total = Decimal(sum(sub.tax_amount for sub in y.sub_items)) if all(sub.tax_amount is not None for sub in y.sub_items) and len(y.sub_items) > 0 else None

        # Solve constraints
        i = 0
        prev_values = None
        while True:
            i += 1
            y.solve()
            if y.is_complete():
                break
            
            # Check if values have changed
            current_values = y.solvable_fields()
            if current_values == prev_values:
                if y.quantity is None:
                    y.quantity = Decimal(1)
                    continue
                # Break when we tried assuming some values but nothing changed
                break
            
            prev_values = current_values

        return y
    
    def solvable_fields(self) -> List[Any]:
        """Return current values of all fields that can be solved"""
        return [
            self.net_unit_price, self.unit_tax, self.gross_unit_price,
            self.net_price, self.tax_amount, self.gross_price,
            self.tax_rate,
            self.net_total, self.total_tax, self.gross_total,
            self.net_sub_items_total, self.gross_sub_items_total
        ]

    def is_complete(self) -> bool:
        return all(x is not None for x in [self.net_unit_price, 
                                           self.net_price,
                                           self.tax_rate,
                                           self.quantity, 
                                           self.gross_price, 
                                           self.tax_amount, 
                                           self.gross_unit_price, 
                                           self.net_total, 
                                           self.gross_total])
    def solve(self):
        equations = [
            "net_price = net_unit_price * quantity",
            "gross_price = gross_unit_price * quantity", 
            "tax_amount = unit_tax * quantity",
            "unit_tax = net_unit_price * tax_rate",
            "tax_rate = unit_tax / net_unit_price",

            "gross_unit_price = net_unit_price + unit_tax",
            "gross_price = net_price + tax_amount",

            "total_tax = tax_amount + tax_sub_items_total",
            "net_total = net_price + net_sub_items_total - net_discounts",
            "gross_total = gross_price + gross_sub_items_total - gross_discounts",
        ]

        for eq in equations:
            solve_equation(eq, self)

class InvoiceImplicitSubLineItem(ImplicitBaseModel):
    name: Optional[str] = Field(None, description='The name of the subline item')

    tax_rate: Optional[Decimal] = Field(None, description='The tax rate of the subline item')

    net_unit_price: Optional[Decimal] = Field(None, description='The unit price of the subline item without tax')
    unit_tax: Optional[Decimal] = Field(None, description='The tax amount of the subline item per unit')
    gross_unit_price: Optional[Decimal] = Field(None, description='The unit price of the subline item including tax')

    quantity: Optional[Decimal] = Field(None, description='The quantity of the subline item')

    net_price: Optional[Decimal] = Field(None, description='The net price of the subline item')
    tax_amount: Optional[Decimal] = Field(None, description='The tax amount of the subline item')
    gross_price: Optional[Decimal] = Field(None, description='The gross price of the subline item including tax')

    @classmethod
    def get_list_types(cls) -> Set[str]:
        return set()

    def get_constraints(self) -> List[BaseConstraint]:
        return [
            IsCloseConstraint("net_price=net_unit_price*quantity", 
                       lambda m: (m.net_price, m.net_unit_price * m.quantity)),
            IsCloseConstraint("tax_amount=unit_tax*quantity", 
                       lambda m: (m.tax_amount, m.unit_tax * m.quantity)),
            IsCloseConstraint("gross_price=gross_unit_price*quantity", 
                       lambda m: (m.gross_price, m.gross_unit_price * m.quantity)),
            IsCloseConstraint("gross_unit_price=net_unit_price+unit_tax", 
                       lambda m: (m.gross_unit_price, m.net_unit_price + m.unit_tax)),
            IsCloseConstraint("gross_price=net_price+tax_amount", 
                       lambda m: (m.gross_price, m.net_price + m.tax_amount)),
            IsCloseConstraint("unit_tax=net_unit_price*tax_rate", 
                       lambda m: (m.unit_tax, m.net_unit_price * m.tax_rate))
        ]

    @staticmethod
    def from_explicit(x: InvoiceSubLineItem) -> 'InvoiceImplicitSubLineItem':
        y = InvoiceImplicitSubLineItem()

        # String fields
        y.name = x.name

        # Parse fields
        # y.tax_rate = parse_decimal(x.tax_rate, is_percentage=True)  # Never present, only inferred
        y.net_unit_price = parse_decimal(x.net_unit_price)
        y.unit_tax = parse_decimal(x.unit_tax)
        y.gross_unit_price = parse_decimal(x.gross_unit_price)
        y.quantity = parse_cnt_decimal(x.quantity) or Decimal(1)
        y.net_price = parse_decimal(x.net_price)
        y.tax_amount = parse_decimal(x.tax_amount)
        y.gross_price = parse_decimal(x.gross_price)

        # Solve fields
        i = 0
        prev_values = None
        while True:
            i += 1
            y.solve()
            if y.is_complete():
                break
            
            current_values = y.solvable_fields()
            if current_values == prev_values:
                if y.gross_price is None and y.net_price is None:
                    y.gross_price = Decimal(0)
                    y.net_price = Decimal(0)
                    continue
                # Break when we tried assuming some values but nothing changed
                break
            
            prev_values = current_values
                
        return y
    
    def solvable_fields(self) -> List[Optional[Decimal]]:
        return [self.net_unit_price, self.unit_tax, self.gross_unit_price,
                self.net_price, self.tax_amount, self.gross_price]
    
    def is_complete(self) -> bool:
        return all(x is not None for x in [self.net_unit_price, 
                                           self.net_price, 
                                           self.quantity, 
                                           self.gross_price, 
                                           self.tax_amount, 
                                           self.gross_unit_price])
    
    def solve(self):
        equations = [
            "net_price = net_unit_price * quantity",
            "tax_amount = unit_tax * quantity",
            "gross_price = gross_unit_price * quantity",
            "gross_unit_price = net_unit_price + unit_tax",
            "gross_price = net_price + tax_amount",
            "unit_tax = net_unit_price * tax_rate"
        ]
        for eq in equations:
            solve_equation(eq, self)
    
def flatten_dict(d: Dict[str, Any]) -> Dict[str, Any]:
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out.update({f'{k}.{k2}': v2 for k2, v2 in flatten_dict(v).items()})
        elif isinstance(v, list):
            for i, item in enumerate(v):
                if isinstance(item, dict):
                    out.update({f'{k}.{k2}': v2 for k2, v2 in flatten_dict(item).items()})
                else:
                    out[f'{k}'] = item
        else:
            out[k] = v
    return out
