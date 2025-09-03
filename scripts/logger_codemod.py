#!/usr/bin/env python3
from __future__ import annotations
import sys
import pathlib
from typing import Optional, List

import libcst as cst
import libcst.matchers as m


def _param_name(p: cst.Param | None) -> Optional[str]:
    return p.name.value if isinstance(p, cst.Param) and isinstance(p.name, cst.Name) else None


def _has_logging_import(mod: cst.Module) -> bool:
    for stmt in mod.body:
        if isinstance(stmt, cst.SimpleStatementLine):
            for small in stmt.body:
                if isinstance(small, cst.Import):
                    for alias in small.names:
                        if isinstance(alias, cst.ImportAlias) and isinstance(alias.name, cst.Name):
                            if alias.name.value == "logging":
                                return True
                if isinstance(small, cst.ImportFrom):
                    if isinstance(small.module, cst.Name) and small.module.value == "logging":
                        return True
    return False


def _insert_import_logging(mod: cst.Module) -> cst.Module:
    if _has_logging_import(mod):
        return mod
    imp = cst.SimpleStatementLine(body=[cst.Import(names=[cst.ImportAlias(name=cst.Name("logging"))])])
    body = list(mod.body)
    if body and isinstance(body[0], cst.SimpleStatementLine) and body[0].body and isinstance(body[0].body[0], cst.Expr) and isinstance(body[0].body[0].value, cst.SimpleString):
        return mod.with_changes(body=[body[0], imp] + body[1:])
    return mod.with_changes(body=[imp] + body)


def _has_self_logger_assignment(func_body: List[cst.BaseStatement]) -> bool:
    assign_match = m.SimpleStatementLine(
        body=[m.Assign(targets=[m.AssignTarget(target=m.Attribute(value=m.Name("self"), attr=m.Name("logger")))])]
    )
    return any(m.matches(stmt, assign_match) for stmt in func_body)


def _strip_logger_from_params(params: cst.Parameters) -> cst.Parameters:
    new_pos = [p for p in params.params if _param_name(p) != "logger"]
    new_kwonly = [p for p in params.kwonly_params if _param_name(p) != "logger"]

    # star_arg may be ParamStar (bare *) or Param (*args)
    star_arg = params.star_arg
    if isinstance(star_arg, cst.Param) and _param_name(star_arg) == "logger":
        star_arg = None  # extremely rare, but just in case

    # modern LibCST uses star_kwarg for **kwargs
    star_kwarg = params.star_kwarg
    if isinstance(star_kwarg, cst.Param) and _param_name(star_kwarg) == "logger":
        star_kwarg = None

    return params.with_changes(params=new_pos, kwonly_params=new_kwonly, star_arg=star_arg, star_kwarg=star_kwarg)


class _StripLoggerFromSuper(cst.CSTTransformer):
    def leave_Call(self, original_node: cst.Call, updated_node: cst.Call) -> cst.Call:
        func = updated_node.func
        if isinstance(func, cst.Attribute) and isinstance(func.attr, cst.Name) and func.attr.value == "__init__":
            new_args = []
            for a in updated_node.args:
                if a.keyword is None and isinstance(a.value, cst.Name) and a.value.value == "logger":
                    continue
                if a.keyword and isinstance(a.keyword, cst.Name) and a.keyword.value == "logger":
                    continue
                new_args.append(a)
            if len(new_args) != len(updated_node.args):
                return updated_node.with_changes(args=new_args)
        return updated_node


class LoggerTransformer(cst.CSTTransformer):
    def __init__(self) -> None:
        self.modified = False

    def leave_Module(self, original_node: cst.Module, updated_node: cst.Module) -> cst.Module:
        return _insert_import_logging(updated_node) if self.modified else updated_node

    def leave_FunctionDef(self, original_node: cst.FunctionDef, updated_node: cst.FunctionDef) -> cst.FunctionDef:
        if not (isinstance(updated_node.name, cst.Name) and updated_node.name.value == "__init__"):
            return updated_node

        params = updated_node.params

        # detect any logger parameter among positional, kw-only, *args, **kwargs
        has_logger = any(_param_name(p) == "logger" for p in params.params)
        has_logger = has_logger or any(_param_name(p) == "logger" for p in params.kwonly_params)
        if isinstance(params.star_arg, cst.Param) and _param_name(params.star_arg) == "logger":
            has_logger = True
        if isinstance(params.star_kwarg, cst.Param) and _param_name(params.star_kwarg) == "logger":
            has_logger = True

        if not has_logger:
            return updated_node

        # 1) remove from signature
        new_params = _strip_logger_from_params(params)

        # 2) process body
        body_stmts = list(updated_node.body.body)
        cleaned: List[cst.BaseStatement] = []
        for stmt in body_stmts:
            # drop `self.logger = logger`
            if m.matches(
                stmt,
                m.SimpleStatementLine(
                    body=[m.Assign(
                        targets=[m.AssignTarget(target=m.Attribute(value=m.Name("self"), attr=m.Name("logger")))],
                        value=m.Name("logger"),
                    )]
                ),
            ):
                self.modified = True
                continue
            # strip logger from super().__init__(...)
            stmt = stmt.visit(_StripLoggerFromSuper())
            cleaned.append(stmt)

        # 3) inject default logger if missing
        if not _has_self_logger_assignment(cleaned):
            cleaned.insert(0, cst.parse_statement("self.logger = logging.getLogger(__name__)\n"))
            self.modified = True

        self.modified = True
        return updated_node.with_changes(params=new_params, body=updated_node.body.with_changes(body=cleaned))


def transform_file(path: pathlib.Path) -> None:
    try:
        src = path.read_text(encoding="utf-8")
        mod = cst.parse_module(src)
        new = mod.visit(LoggerTransformer())
        if new.code != src:
            path.write_text(new.code, encoding="utf-8")
            print(f"✔ Rewrote: {path}")
        else:
            print(f"· No change: {path}")
    except Exception as e:
        print(f"⚠️  Skipped (parse/transform error): {path} -> {e}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python tools/logger_codemod.py <path-or-file>")
        sys.exit(2)
    target = pathlib.Path(sys.argv[1]).resolve()
    if target.is_file():
        transform_file(target)
    else:
        for py in target.rglob("*.py"):
            transform_file(py)


if __name__ == "__main__":
    main()
