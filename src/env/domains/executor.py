from __future__ import annotations

from typing import Any

from env.core.types import ToolExecutionResult
from env.core.utils import normalize_runtime_value


def _canonicalize_arguments(tool_spec: dict[str, Any], arguments: dict[str, Any]) -> dict[str, Any]:
    mapping = tool_spec.get("surface_argument_to_datatype")
    if not isinstance(mapping, dict) or not mapping:
        return dict(arguments)

    canonical_arguments: dict[str, Any] = {}
    for argument_name, argument_value in arguments.items():
        canonical_name = mapping.get(argument_name, argument_name)
        canonical_arguments[canonical_name] = argument_value
    return canonical_arguments


def _tool_identifier(tool_spec: dict[str, Any]) -> str:
    return str(tool_spec.get("name") or tool_spec.get("tool_name") or "")


def _resolve_untrusted_source_type(tool_spec: dict[str, Any]) -> str | None:
    tool_type = str(tool_spec.get("tool_type") or "").strip().lower()
    if tool_type.startswith("noisy"):
        return "noisy"
    if tool_type.startswith("blocker"):
        return "blocker"
    return None


def _resolve_untrusted_output_value(tool_spec: dict[str, Any]) -> Any:
    if "return_value" in tool_spec and tool_spec.get("return_value") is not None:
        return tool_spec.get("return_value")

    description = tool_spec.get("description")
    if isinstance(description, str) and description.strip():
        return description

    tool_name = _tool_identifier(tool_spec)
    if tool_name:
        return tool_name
    return None


def _execute_untrusted_tool(tool_spec: dict[str, Any]) -> ToolExecutionResult | None:
    source_type = _resolve_untrusted_source_type(tool_spec)
    if source_type is None:
        return None

    noise_type = str(tool_spec.get("noise_type") or "").strip()
    output_value = _resolve_untrusted_output_value(tool_spec)
    success = noise_type not in {"explicit failures", "deprecated", "condition_limited"}
    output_provenance = "untrusted" if output_value is not None else "none"
    untrusted_source_type = source_type if output_value is not None else None

    return ToolExecutionResult(
        tool_name=_tool_identifier(tool_spec),
        success=success,
        output_datatype=tool_spec["output_datatype"],
        output_value=output_value,
        tool_type=str(tool_spec.get("tool_type") or f"{source_type}_misleading"),
        output_provenance=output_provenance,
        untrusted_source_type=untrusted_source_type,
    )


class FinanceToolExecutor:
    def __init__(self, database: dict[str, Any]) -> None:
        self.database = database
        self.rows = self._build_rows()

    def _company(self, tax_id: str) -> dict[str, Any]:
        return self.database["companies"][tax_id]

    def _account(self, account_id: str) -> dict[str, Any]:
        return self.database["accounts"][account_id]

    def _card_by_account(self, account_id: str) -> tuple[str | None, dict[str, Any] | None]:
        for card_token, card in self.database["cards"].items():
            if card["Account_ID_Internal"] == account_id:
                merged = dict(card)
                merged["Card_Token"] = card_token
                return card_token, merged
        return None, None

    def _fx(self, fx_quote_id: str) -> dict[str, Any]:
        return self.database["fx_quotes"][fx_quote_id]

    def _payment(self, payment_intent_id: str) -> dict[str, Any]:
        return self.database["payments"][payment_intent_id]

    def _ledger_by_payment(self, payment_intent_id: str) -> tuple[str | None, dict[str, Any] | None]:
        for trace_id, event in self.database["ledger_events"].items():
            if event["Payment_Intent_ID"] == payment_intent_id:
                merged = dict(event)
                merged["Trace_ID"] = trace_id
                return trace_id, merged
        return None, None

    def _build_rows(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []

        for tax_id, company in self.database["companies"].items():
            row = dict(company)
            row["Tax_ID"] = tax_id
            rows.append(row)

        for account_id, account in self.database["accounts"].items():
            row = dict(account)
            row["Account_ID_Internal"] = account_id
            rows.append(row)

            company = self._company(account["Tax_ID"])
            row = dict(company)
            row["Tax_ID"] = account["Tax_ID"]
            row.update(account)
            row["Account_ID_Internal"] = account_id
            rows.append(row)

        for card_token, card in self.database["cards"].items():
            row = dict(card)
            row["Card_Token"] = card_token
            rows.append(row)

            account = self._account(card["Account_ID_Internal"])
            company = self._company(account["Tax_ID"])
            row = dict(company)
            row["Tax_ID"] = account["Tax_ID"]
            row.update(account)
            row["Account_ID_Internal"] = card["Account_ID_Internal"]
            row.update(card)
            row["Card_Token"] = card_token
            rows.append(row)

        for fx_quote_id, fx_quote in self.database["fx_quotes"].items():
            row = dict(fx_quote)
            row["FX_Quote_ID"] = fx_quote_id
            rows.append(row)

        for payment_intent_id, payment in self.database["payments"].items():
            account = self._account(payment["Account_ID_Internal"])
            company = self._company(account["Tax_ID"])
            fx_quote = self._fx(payment["FX_Quote_ID"])
            _, card = self._card_by_account(payment["Account_ID_Internal"])
            trace_id, ledger = self._ledger_by_payment(payment_intent_id)

            payment_row = dict(payment)
            payment_row["Payment_Intent_ID"] = payment_intent_id
            rows.append(payment_row)

            row = dict(account)
            row["Account_ID_Internal"] = payment["Account_ID_Internal"]
            row.update(payment)
            row["Payment_Intent_ID"] = payment_intent_id
            rows.append(row)

            row = dict(company)
            row["Tax_ID"] = account["Tax_ID"]
            row.update(account)
            row["Account_ID_Internal"] = payment["Account_ID_Internal"]
            row.update(payment)
            row["Payment_Intent_ID"] = payment_intent_id
            rows.append(row)

            row = dict(company)
            row["Tax_ID"] = account["Tax_ID"]
            row.update(account)
            row["Account_ID_Internal"] = payment["Account_ID_Internal"]
            if card:
                row.update(card)
            row.update(payment)
            row["Payment_Intent_ID"] = payment_intent_id
            row.update(fx_quote)
            row["FX_Quote_ID"] = payment["FX_Quote_ID"]
            if ledger:
                row.update(ledger)
            if trace_id:
                row["Trace_ID"] = trace_id
            rows.append(row)

        for trace_id, event in self.database["ledger_events"].items():
            row = dict(event)
            row["Trace_ID"] = trace_id
            rows.append(row)

        return rows

    def execute_tool(self, tool_spec: dict[str, Any], arguments: dict[str, Any]) -> ToolExecutionResult:
        untrusted_result = _execute_untrusted_tool(tool_spec)
        if untrusted_result is not None:
            return untrusted_result

        canonical_arguments = _canonicalize_arguments(tool_spec, arguments)
        input_datatypes = tool_spec["input_datatypes"]
        output_datatype = tool_spec["output_datatype"]

        matched_outputs = []
        for row in self.rows:
            if output_datatype not in row or row[output_datatype] is None:
                continue

            ok = True
            for input_datatype in input_datatypes:
                if input_datatype not in row:
                    ok = False
                    break
                if normalize_runtime_value(row[input_datatype]) != normalize_runtime_value(canonical_arguments[input_datatype]):
                    ok = False
                    break
            if ok:
                matched_outputs.append(row[output_datatype])

        distinct_outputs = []
        seen = set()
        for output in matched_outputs:
            normalized = normalize_runtime_value(output)
            if normalized in seen:
                continue
            distinct_outputs.append(output)
            seen.add(normalized)

        if not distinct_outputs:
            output_value = None
        elif len(distinct_outputs) == 1:
            output_value = distinct_outputs[0]
        else:
            raise RuntimeError(
                f"Ambiguous finance tool execution for {_tool_identifier(tool_spec)} with arguments {canonical_arguments}: "
                f"{distinct_outputs}"
            )

        return ToolExecutionResult(
            tool_name=_tool_identifier(tool_spec),
            success=True,
            output_datatype=output_datatype,
            output_value=output_value,
            tool_type=str(tool_spec.get("tool_type") or "baseline"),
            output_provenance="trusted" if output_value is not None else "none",
            untrusted_source_type=None,
        )


class RetailToolExecutor:
    def __init__(self, database: dict[str, Any]) -> None:
        self.database = database
        self.rows = self._build_rows()

    def _build_rows(self) -> list[dict[str, Any]]:
        order_cases = self.database.get("order_cases", {})
        if not isinstance(order_cases, dict):
            return []

        rows: list[dict[str, Any]] = []
        for case in order_cases.values():
            if isinstance(case, dict):
                rows.append(case)
        return rows

    def execute_tool(self, tool_spec: dict[str, Any], arguments: dict[str, Any]) -> ToolExecutionResult:
        untrusted_result = _execute_untrusted_tool(tool_spec)
        if untrusted_result is not None:
            return untrusted_result

        canonical_arguments = _canonicalize_arguments(tool_spec, arguments)
        input_datatypes = tool_spec["input_datatypes"]
        output_datatype = tool_spec["output_datatype"]

        matched_outputs = []
        for row in self.rows:
            if output_datatype not in row or row[output_datatype] is None:
                continue

            ok = True
            for input_datatype in input_datatypes:
                if input_datatype not in row:
                    ok = False
                    break
                if normalize_runtime_value(row[input_datatype]) != normalize_runtime_value(canonical_arguments[input_datatype]):
                    ok = False
                    break
            if ok:
                matched_outputs.append(row[output_datatype])

        distinct_outputs = []
        seen = set()
        for output in matched_outputs:
            normalized = normalize_runtime_value(output)
            if normalized in seen:
                continue
            distinct_outputs.append(output)
            seen.add(normalized)

        if not distinct_outputs:
            output_value = None
        elif len(distinct_outputs) == 1:
            output_value = distinct_outputs[0]
        else:
            raise RuntimeError(
                f"Ambiguous retail tool execution for {_tool_identifier(tool_spec)} with arguments {canonical_arguments}: "
                f"{distinct_outputs}"
            )

        return ToolExecutionResult(
            tool_name=_tool_identifier(tool_spec),
            success=True,
            output_datatype=output_datatype,
            output_value=output_value,
            tool_type=str(tool_spec.get("tool_type") or "baseline"),
            output_provenance="trusted" if output_value is not None else "none",
            untrusted_source_type=None,
        )


class DomainToolExecutor:
    def __init__(self, databases: dict[str, dict[str, Any]]) -> None:
        self.executors: dict[str, Any] = {}
        for domain, database in databases.items():
            if domain == "finance":
                self.executors[domain] = FinanceToolExecutor(database)
            elif domain == "retail":
                self.executors[domain] = RetailToolExecutor(database)

    def execute_tool(self, tool_spec: dict[str, Any], arguments: dict[str, Any]) -> ToolExecutionResult:
        domain = tool_spec["domain"]
        executor = self.executors.get(domain)
        if executor is None:
            raise NotImplementedError(f"Tool execution for domain {domain!r} is not implemented yet.")
        return executor.execute_tool(tool_spec, arguments)
