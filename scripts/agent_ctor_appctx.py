#!/usr/bin/env python3
import sys
import pathlib
import libcst as cst
from libcst import matchers as m

APPCTX_ANNOT = cst.Annotation(cst.Name("AppContext"))
APP_PARAM = cst.Param(name=cst.Name("app"), annotation=APPCTX_ANNOT)

class CtorTransform(cst.CSTTransformer):
    def __init__(self):
        super().__init__()
        self.in_agent = False

    def visit_ClassDef(self, node: cst.ClassDef) -> bool:
        # Is subclass of BaseAgent?
        if any(getattr(b, "value", None) and getattr(b.value, "value", "") == "BaseAgent" for b in node.bases):
            self.in_agent = True
        return True

    def leave_ClassDef(self, node: cst.ClassDef, updated: cst.ClassDef) -> cst.ClassDef:
        self.in_agent = False
        return updated

    def leave_FunctionDef(self, orig: cst.FunctionDef, upd: cst.FunctionDef) -> cst.FunctionDef:
        if not self.in_agent or orig.name.value != "__init__":
            return upd

        # change signature to (self, app: AppContext)
        params = upd.params
        if len(params.params) >= 2 and params.params[1].name.value in ("cfg","app"):
            new_params = params.with_changes(
                params=[params.params[0], APP_PARAM],
                posonly_params=[], kwonly_params=[], star_arg=None, star_kwarg=None
            )
        else:
            # already migrated or different shape
            new_params = params

        # fix super().__init__(...)
        def replace_super_call(stmt):
            if m.matches(stmt,
                m.SimpleStatementLine(
                    body=[m.Expr(m.Call(func=m.Attribute(value=m.Call(func=m.Name("super")), attr=m.Name("__init__"))))]
                )
            ):
                call = cst.Call(
                    func=cst.Attribute(value=cst.Call(func=cst.Name("super"), args=[]), attr=cst.Name("__init__")),
                    args=[cst.Arg(cst.Name("app"))]
                )
                return cst.SimpleStatementLine([cst.Expr(call)])
            return stmt

        new_body = []
        for s in upd.body.body:
            new_body.append(replace_super_call(s))
        return upd.with_changes(params=new_params, body=upd.body.with_changes(body=new_body))

def process(root: pathlib.Path):
    for py in root.rglob("*.py"):
        code = py.read_text(encoding="utf-8")
        mod = cst.parse_module(code)
        new = mod.visit(CtorTransform())
        if new.code != code:
            py.write_text(new.code, encoding="utf-8")
            print("updated", py)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: agent_ctor_appctx.py <repo-root>", file=sys.stderr); sys.exit(2)
    process(pathlib.Path(sys.argv[1]))
