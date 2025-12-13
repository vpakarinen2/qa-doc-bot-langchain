import operator
import ast

from langchain.tools import tool
from typing import Union


_ALLOWED_BINARY_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
}


def _eval_node(node: ast.AST) -> Union[int, float]:
    if isinstance(node, ast.Expression):
        return _eval_node(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp):
        left = _eval_node(node.left)
        right = _eval_node(node.right)
        op_type = type(node.op)
        if op_type not in _ALLOWED_BINARY_OPERATORS:
            raise ValueError("Unsupported operator")
        return _ALLOWED_BINARY_OPERATORS[op_type](left, right)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        operand = _eval_node(node.operand)
        return -operand
    raise ValueError("Unsupported expression")


def evaluate_math_expression(expression: str) -> Union[int, float]:
    try:
        parsed = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise ValueError(f"Invalid expression: {expression}") from exc
    result = _eval_node(parsed)
    return result


@tool
def math(expression: str) -> str:
    """Evaluate basic math expressions."""
    result = evaluate_math_expression(expression)
    return str(result)
