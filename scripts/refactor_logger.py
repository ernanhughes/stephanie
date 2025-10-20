#!/usr/bin/env python3
"""
Refactor constructors to drop `logger` param and use stdlib logging.

Two modes:
  - keep-self (default): keeps `self.logger` alive by assigning
      self.logger = logging.getLogger(__name__)
    at the top of __init__.
  - module: inserts a module-level `logger = logging.getLogger(__name__)`
    and replaces all `self.logger` references with `logger`.

Also removes:
  - keyword callsites: `logger=...`
  - super().__init__(..., logger) trailing arg

Optionally removes trailing positional arg named `logger` at callsites with --drop-positional.

Requires: libcst (`pip install libcst`)
"""

from __future__ import annotations
import argparse
import os
from pathlib import Path
from typing import Optional

import libcst as cst
import libcst.matchers as m


def _rm_param(params: cst.Parameters, name: str) -> cst.Parameters:
    def keep(p: cst.Param) -> bool:
        return p.name.value != name

    return params.with_changes(
        posonly_params=[p for p in params.posonly_params if keep(p)],
        params=[p for p in params.params if keep(p)],
        kwonly_params=[p for p in params.kwonly_params if keep(p)],
        # star_arg / star_kwarg: only remove if the *name* matches exactly
        star_arg=params.star_arg if not (params.star_arg and params.star_arg.name and params.star_arg.name.value == name) else None,
        star_kwarg=params.star_kwarg if not (params.star_kwarg and params.star_kwarg.name and params.star_kwarg.name.value == name) else None,
    )


def _import_logging_stmt() -> cst.SimpleStatementLine:
    return cst.SimpleStatementLine([cst.Import(names=[cst.ImportAlias(name=cst.Name("logging"))])])


def _module_logger_stmt() -> cst.SimpleStatementLine:
    return cst.SimpleStatementLine(
        [cst.Assign(
            targets=[cst.AssignTarget(target=cst.Name("logger"))],
            value=cst.Call(
                func=cst.Attribute(value=cst.Name("logging"), attr=cst.Name("getLogger")),
                args=[cst.Arg(value=cst.Name("__name__"))],
            ),
        )]
    )


def _self_logger_assign_stmt() -> cst.SimpleStatementLine:
    return cst.SimpleStatementLine(
        [cst.Assign(
            targets=[cst.AssignTarget(target=cst.Attribute(value=cst.Name("self"), attr=cst.Name("logger")))],
            value=cst.Call(
                func=cst.Attribute(value=cst.Name("logging"), attr=cst.Name("getLogger")),
                args=[cst.Arg(value=cst.Name("__name__"))],
            ),
        )]
    )


class LoggerRefactor(cst.CSTTransformer):
    def __init__(self, mode: str = "keep-self", drop_positional_callsites: bool = False):
        assert mode in ("keep-self", "module")
        self.mode = mode
        self.drop_positional_callsites = drop_positional_callsites

        # module-level flags
        self.seen_import_logging = False
        self.need_import_logging = False
        self.need_module_logger = False
        self.need_self_logger_assign = False

    # ---- Track imports -------------------------------------------------

    def visit_Import(self, node: cst.Import) -> Optional[bool]:
        if any(isinstance(alias, cst.ImportAlias) and alias.name.value == "logging" for alias in node.names):
            self.seen_import_logging = True
        return True

    def visit_ImportFrom(self, node: cst.ImportFrom) -> Optional[bool]:
        if isinstance(node.module, cst.Name) and node.module.value == "logging":
            self.seen_import_logging = True
        return True

    # ---- Replace self.logger usages in module mode ---------------------

    def leave_Attribute(self, original_node: cst.Attribute, updated_node: cst.Attribute) -> cst.BaseExpression:
        if self.mode == "module" and m.matches(original_node, m.Attribute(value=m.Name("self"), attr=m.Name("logger"))):
            # Replace `self.logger` -> `logger`
            self.need_import_logging = True
            self.need_module_logger = True
            return cst.Name("logger")
        return updated_node

    # ---- Rewrite __init__ signatures and bodies -----------------------

    def leave_FunctionDef(self, original_node: cst.FunctionDef, updated_node: cst.FunctionDef) -> cst.FunctionDef:
        if original_node.name.value != "__init__":
            return updated_node

        # Remove `logger` from parameters
        params = _rm_param(updated_node.params, "logger")
        func = updated_node.with_changes(params=params)

        # In keep-self mode: ensure self.logger assignment exists at top of body
        if self.mode == "keep-self":
            # If body already assigns `self.logger = ...`, leave as is. Otherwise insert it first.
            body_stmts = list(func.body.body)

            # Remove lines like `self.logger = logger` or `self.logger = JSONLogger(...)`
            new_body = []
            removed_one = False
            for stmt in body_stmts:
                # Detect a simple `self.logger = logger` assignment
                if m.matches(
                    stmt,
                    m.SimpleStatementLine(
                        body=[m.Assign(
                            targets=[m.AssignTarget(target=m.Attribute(value=m.Name("self"), attr=m.Name("logger")))],
                            value=m.Name("logger"),
                        )]
                    ),
                ):
                    removed_one = True
                    continue
                new_body.append(stmt)

            # Insert our default at the very top if no explicit assignment exists
            if not any(
                m.matches(
                    s,
                    m.SimpleStatementLine(
                        body=[m.Assign(
                            targets=[m.AssignTarget(target=m.Attribute(value=m.Name("self"), attr=m.Name("logger")))],
                        )]
                    ),
                ) for s in new_body
            ):
                self.need_import_logging = True
                new_body.insert(0, _self_logger_assign_stmt())

            func = func.with_changes(body=func.body.with_changes(body=new_body))

        # In module mode: remove any `self.logger = ...` assignment lines
        else:
            body_stmts = list(func.body.body)
            new_body = []
            for stmt in body_stmts:
                if m.matches(
                    stmt,
                    m.SimpleStatementLine(
                        body=[m.Assign(
                            targets=[m.AssignTarget(target=m.Attribute(value=m.Name("self"), attr=m.Name("logger")))]
                        )]
                    ),
                ):
                    # drop the assignment in module mode
                    continue
                new_body.append(stmt)
            func = func.with_changes(body=func.body.with_changes(body=new_body))

        # Remove `logger` arg from super().__init__(...) if present
        class SuperArgStripper(cst.CSTTransformer):
            def leave_Call(self, o: cst.Call, u: cst.Call) -> cst.Call:
                if m.matches(
                    o.func,
                    m.Attribute(
                        value=m.Call(func=m.Name("super")),
                        attr=m.Name("__init__"),
                    ),
                ):
                    # strip keyword logger=...
                    new_args = [a for a in u.args if not (a.keyword and a.keyword.value == "logger")]
                    # optionally strip trailing positional Name("logger")
                    if new_args:
                        last = new_args[-1]
                        if not last.keyword and isinstance(last.value, cst.Name) and last.value.value == "logger":
                            new_args = new_args[:-1]
                    return u.with_changes(args=new_args)
                return u

        func = func.visit(SuperArgStripper())

        return func

    # ---- Remove keyword callsites: logger=... --------------------------

    def leave_Call(self, original_node: cst.Call, updated_node: cst.Call) -> cst.Call:
        # Remove keyword logger=... everywhere
        new_args = [a for a in updated_node.args if not (a.keyword and a.keyword.value == "logger")]

        # Optionally remove trailing positional arg that is `logger`
        if self.drop_positional_callsites and new_args:
            last = new_args[-1]
            if not last.keyword and isinstance(last.value, cst.Name) and last.value.value == "logger":
                new_args = new_args[:-1]

        return updated_node.with_changes(args=new_args)

    # ---- Module finalize: ensure imports / module-level logger ---------

    def leave_Module(self, original_node: cst.Module, updated_node: cst.Module) -> cst.Module:
        body = list(updated_node.body)

        # Ensure `import logging`
        if (self.mode == "keep-self" and self.need_import_logging and not self.seen_import_logging) or \
           (self.mode == "module" and (self.need_module_logger or self.need_import_logging) and not self.seen_import_logging):
            body.insert(0, _import_logging_stmt())

        # Ensure module-level `logger = logging.getLogger(__name__)` in module mode
        if self.mode == "module" and self.need_module_logger:
            # Insert after imports
            insert_idx = 0
            for i, stmt in enumerate(body):
                if m.matches(stmt, m.SimpleStatementLine(body=[m.Import() | m.ImportFrom()])):
                    insert_idx = i + 1
                else:
                    # stop on the first non-import
                    if i == 0:
                        insert_idx = 0
                    break
            body.insert(insert_idx, _module_logger_stmt())

        return updated_node.with_changes(body=body)


def process_file(path: Path, mode: str, drop_positional: bool, dry_run: bool) -> bool:
    code = path.read_text(encoding="utf-8")
    try:
        module = cst.parse_module(code)
        transformer = LoggerRefactor(mode=mode, drop_positional_callsites=drop_positional)
        new_module = module.visit(transformer)
        new_code = new_module.code
        if new_code != code:
            if not dry_run:
                path.write_text(new_code, encoding="utf-8")
            return True
    except Exception as e:
        print(f"⚠️  Skipped (parse/transform error): {path}  -> {e}")
    return False


def main():
    ap = argparse.ArgumentParser(description="Refactor logger param -> stdlib logging")
    ap.add_argument("root", type=str, help="Project root to traverse")
    ap.add_argument("--mode", choices=["keep-self", "module"], default="keep-self",
                    help="Refactor strategy (default: keep-self)")
    ap.add_argument("--drop-positional", action="store_true",
                    help="Also drop trailing positional `logger` args at call sites")
    ap.add_argument("--dry-run", action="store_true", help="Do not write changes")
    ap.add_argument("--include", nargs="*", default=[".py"], help="File extensions to include")
    ap.add_argument("--exclude-dirs", nargs="*", default=[".git", ".venv", "venv", "__pycache__"],
                    help="Directories to skip")
    args = ap.parse_args()

    root = Path(args.root)
    changed = 0
    total = 0
    for dirpath, dirnames, filenames in os.walk(root):
        # prune excluded dirs
        dirnames[:] = [d for d in dirnames if d not in set(args.exclude_dirs)]
        for fn in filenames:
            if not any(fn.endswith(ext) for ext in args.include):
                continue
            p = Path(dirpath) / fn
            total += 1
            if process_file(p, args.mode, args.drop_positional, args.dry_run):
                changed += 1

    print(f"Done. Scanned {total} files. {'Would modify' if args.dry_run else 'Modified'} {changed} file(s).")


if __name__ == "__main__":
    main()
