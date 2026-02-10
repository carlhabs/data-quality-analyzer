from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
import yaml


@dataclass(frozen=True)
class ColumnRule:
    type: str | None = None
    required: bool = False
    min: float | None = None
    max: float | None = None
    regex: str | None = None
    allowed: list[Any] | None = None
    not_future: bool = False


@dataclass(frozen=True)
class RowRule:
    name: str
    expr: str


@dataclass(frozen=True)
class Rules:
    columns: dict[str, ColumnRule]
    unique_keys: list[list[str]]
    row_rules: list[RowRule]


class RulesError(ValueError):
    pass


def load_rules(path: Path) -> Rules:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise RulesError("Rules file must be a mapping at root")

    columns_payload = payload.get("columns", {}) or {}
    if not isinstance(columns_payload, dict):
        raise RulesError("columns must be a mapping")

    columns: dict[str, ColumnRule] = {}
    for col, config in columns_payload.items():
        if not isinstance(config, dict):
            raise RulesError(f"Column rule for {col} must be a mapping")
        rule = ColumnRule(
            type=config.get("type"),
            required=bool(config.get("required", False)),
            min=config.get("min"),
            max=config.get("max"),
            regex=config.get("regex"),
            allowed=config.get("allowed"),
            not_future=bool(config.get("not_future", False)),
        )
        columns[col] = rule

    unique_keys_payload = payload.get("unique_keys", []) or []
    if not isinstance(unique_keys_payload, list):
        raise RulesError("unique_keys must be a list")
    unique_keys: list[list[str]] = []
    for entry in unique_keys_payload:
        if isinstance(entry, list):
            unique_keys.append([str(c) for c in entry])
        else:
            unique_keys.append([str(entry)])

    row_rules_payload = payload.get("row_rules", []) or []
    if not isinstance(row_rules_payload, list):
        raise RulesError("row_rules must be a list")
    row_rules: list[RowRule] = []
    for rule in row_rules_payload:
        if not isinstance(rule, dict) or "name" not in rule or "expr" not in rule:
            raise RulesError("Each row_rule must include name and expr")
        row_rules.append(RowRule(name=str(rule["name"]), expr=str(rule["expr"])))

    return Rules(columns=columns, unique_keys=unique_keys, row_rules=row_rules)


Token = tuple[str, str]


_TOKEN_REGEX = re.compile(
    r"("
    r"(?P<NUMBER>\d+(?:\.\d+)?)|"
    r"(?P<STRING>'[^']*'|\"[^\"]*\")|"
    r"(?P<OP><=|>=|==|!=|<|>)|"
    r"(?P<LPAREN>\()|"
    r"(?P<RPAREN>\))|"
    r"(?P<COMMA>,)|"
    r"(?P<IDENT>[A-Za-z_][A-Za-z0-9_]*)"
    r")"
)


def tokenize(expr: str) -> list[Token]:
    tokens: list[Token] = []
    pos = 0
    while pos < len(expr):
        if expr[pos].isspace():
            pos += 1
            continue
        match = _TOKEN_REGEX.match(expr, pos)
        if not match:
            raise RulesError(f"Invalid token near: {expr[pos:pos+10]}")
        kind = match.lastgroup
        if kind is None:
            raise RulesError("Invalid token in expression")
        value = match.group(kind)
        if kind == "IDENT" and value.lower() in {"and", "or"}:
            kind = value.upper()
        tokens.append((kind, value))
        pos = match.end()
    return tokens


class Parser:
    def __init__(self, tokens: list[Token]):
        self.tokens = tokens
        self.index = 0

    def _peek(self) -> Token | None:
        if self.index >= len(self.tokens):
            return None
        return self.tokens[self.index]

    def _consume(self, kind: str | None = None) -> Token:
        token = self._peek()
        if token is None:
            raise RulesError("Unexpected end of expression")
        if kind and token[0] != kind:
            raise RulesError(f"Expected {kind} but got {token[0]}")
        self.index += 1
        return token

    def parse(self) -> Any:
        expr = self._parse_or()
        if self._peek() is not None:
            raise RulesError("Unexpected token at end of expression")
        return expr

    def _parse_or(self) -> Any:
        left = self._parse_and()
        while self._peek() and self._peek()[0] == "OR":
            self._consume("OR")
            right = self._parse_and()
            left = ("or", left, right)
        return left

    def _parse_and(self) -> Any:
        left = self._parse_comparison()
        while self._peek() and self._peek()[0] == "AND":
            self._consume("AND")
            right = self._parse_comparison()
            left = ("and", left, right)
        return left

    def _parse_comparison(self) -> Any:
        left = self._parse_term()
        if self._peek() and self._peek()[0] == "OP":
            op = self._consume("OP")[1]
            right = self._parse_term()
            return ("cmp", op, left, right)
        return left

    def _parse_term(self) -> Any:
        token = self._peek()
        if token is None:
            raise RulesError("Unexpected end of expression")
        kind, value = token
        if kind == "LPAREN":
            self._consume("LPAREN")
            expr = self._parse_or()
            self._consume("RPAREN")
            return expr
        if kind == "IDENT":
            self._consume("IDENT")
            if self._peek() and self._peek()[0] == "LPAREN":
                self._consume("LPAREN")
                args: list[Any] = []
                if self._peek() and self._peek()[0] != "RPAREN":
                    args.append(self._parse_or())
                    while self._peek() and self._peek()[0] == "COMMA":
                        self._consume("COMMA")
                        args.append(self._parse_or())
                self._consume("RPAREN")
                return ("call", value, args)
            return ("ident", value)
        if kind == "NUMBER":
            self._consume("NUMBER")
            return ("number", float(value))
        if kind == "STRING":
            self._consume("STRING")
            return ("string", value.strip("\"'") )
        raise RulesError(f"Unexpected token {kind}")


def parse_expression(expr: str) -> Any:
    tokens = tokenize(expr)
    parser = Parser(tokens)
    return parser.parse()


_ALLOWED_FUNCS = {"is_null", "not_null"}


def _as_series(value: Any, df: pd.DataFrame) -> pd.Series | Any:
    if isinstance(value, tuple):
        return eval_ast(value, df)
    return value


def eval_ast(node: Any, df: pd.DataFrame) -> pd.Series:
    kind = node[0]
    if kind == "ident":
        name = node[1]
        if name not in df.columns:
            raise RulesError(f"Unknown column in expression: {name}")
        return df[name]
    if kind == "number":
        return node[1]
    if kind == "string":
        return node[1]
    if kind == "call":
        func = node[1]
        args = node[2]
        if func not in _ALLOWED_FUNCS:
            raise RulesError(f"Function not allowed: {func}")
        if len(args) != 1:
            raise RulesError("Functions require a single argument")
        value = _as_series(args[0], df)
        if func == "is_null":
            return pd.isna(value)
        return ~pd.isna(value)
    if kind == "cmp":
        op, left, right = node[1], node[2], node[3]
        left_val = _as_series(left, df)
        right_val = _as_series(right, df)
        if op == "<":
            return left_val < right_val
        if op == "<=":
            return left_val <= right_val
        if op == ">":
            return left_val > right_val
        if op == ">=":
            return left_val >= right_val
        if op == "==":
            return left_val == right_val
        if op == "!=":
            return left_val != right_val
        raise RulesError(f"Unknown operator: {op}")
    if kind == "and":
        left = _as_series(node[1], df)
        right = _as_series(node[2], df)
        return left & right
    if kind == "or":
        left = _as_series(node[1], df)
        right = _as_series(node[2], df)
        return left | right
    raise RulesError("Invalid AST node")


def evaluate_row_rule(expr: str, df: pd.DataFrame) -> pd.Series:
    ast = parse_expression(expr)
    result = eval_ast(ast, df)
    if isinstance(result, pd.Series):
        return result
    return pd.Series([bool(result)] * len(df), index=df.index)


def validate_column_rules(columns: Iterable[str], rules: Rules) -> list[str]:
    missing = []
    for rule_col in rules.columns:
        if rule_col not in columns:
            missing.append(rule_col)
    return missing
