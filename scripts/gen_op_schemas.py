#!/usr/bin/env python3
"""Generate the torch.ops schema lists of the fish-scales-ops API docs.

The schemas are read from the source, not from a built extension, so the script
needs neither torch nor a GPU:

* every ``m.def("...")`` inside a ``TORCH_LIBRARY`` / ``TORCH_LIBRARY_FRAGMENT``
  block of ``csrc/gemm/bindings.cpp`` and ``csrc/attention/csrc/flash_attn_ext.cpp``
  (adjacent C string literals are concatenated, exactly as the compiler does);
* every ``torch.library.custom_op("fish_scales_ops::<name>", schema="...")``
  under ``python/fish_scales_ops/`` (today: ``dense_linear`` and ``moe_layer``).

Usage, from anywhere (the repository root defaults to the parent of ``scripts/``)::

    python scripts/gen_op_schemas.py                   # print the GEMM and MoE block of docs/api/compat.md
    python scripts/gen_op_schemas.py --doc attention   # print the attention block
    python scripts/gen_op_schemas.py --check           # compare with docs/api/*.md, exit 1 on drift
    python scripts/gen_op_schemas.py --write           # rewrite the blocks in docs/api/*.md in place
    python scripts/gen_op_schemas.py <repo-root> ...   # another checkout

Each generated block sits between an ``<!-- BEGIN GENERATED ... -->`` and an
``<!-- END GENERATED ... -->`` line in its document; ``--check`` and ``--write``
work on the text between those two markers and touch nothing else.

GEMM ops are grouped as docs/api/compat.md groups them. A name that is not in the
explicit lists below is placed by the name rules in ``_group_of`` (and reported on
stderr) so that a newly registered op always appears in the output; move it into
the right explicit list when the rule guesses wrong. Within a group the ops keep
their registration order.
"""
from __future__ import annotations

import argparse
import ast
import re
import sys
from pathlib import Path

GEMM_BINDINGS = Path("csrc/gemm/bindings.cpp")
ATTN_BINDINGS = Path("csrc/attention/csrc/flash_attn_ext.cpp")
PY_PACKAGE = Path("python/fish_scales_ops")
NAMESPACE = "fish_scales_ops"

DOCS = {
    "compat": Path("docs/api/compat.md"),
    "attention": Path("docs/api/attention.md"),
}

# Group titles, in output order, for docs/api/compat.md.
GEMM_GROUPS = (
    ("dense_fp8", "Dense block-FP8 (1x128 activation, 128x128 weight) and the bf16 convenience ops"),
    ("dense_mxfp8", "Dense MXFP8 (1x32)"),
    ("moe_mxfp8", "MoE: the MXFP8 grouped GEMMs of sm_100/103 and sm_120/121, their route queries, "
                  "and the router, routing builders and combines of both layouts"),
    ("moe_sm90", "MoE: the sm_90 block-FP8 grouped GEMMs and quantizers"),
    ("python", "Registered from Python (torch.library.custom_op)"),
)

# Explicit membership. Anything not listed falls to the name rules in _group_of.
GEMM_MEMBERS = {
    "dense_fp8": {
        "linear_bf16", "linear_fp8", "linear_qx", "quantize_1x128", "quantize_1x128_packed",
        "quantize_128x128", "repack_fp8_act_scales", "repack_fp8_wgt_scales",
    },
    "dense_mxfp8": {
        "quantize_1x32", "quantize_1x32_packed", "silu_chunk_mul_quantize_1x32", "repack_mxfp8_scales",
        "linear_mxfp8_raw",
    },
    "moe_mxfp8": {
        "linear_mxfp8_grouped_masked", "linear_mxfp8_grouped_masked_swiglu",
        "linear_mxfp8_grouped_masked_combine", "quantize_1x32_grouped_gather",
        "silu_chunk_mul_quantize_1x32_grouped", "mxfp8_grouped_swiglu_available",
        "mxfp8_grouped_swiglu_fused_route", "mxfp8_grouped_slot_possible",
        "mxfp8_grouped_problem_shapes_consumed", "moe_build_routing", "moe_combine",
        "moe_topk_from_logits", "moe_build_sorted", "moe_combine_sorted",
    },
    "moe_sm90": {
        "linear_fp8_grouped_masked", "quantize_1x128_grouped_gather_sm90",
        "silu_chunk_mul_quantize_1x128_grouped_sm90", "linear_fp8_grouped_contiguous",
        "linear_fp8_grouped_contiguous_swapab", "linear_fp8_grouped_contiguous_swapab_pair",
        "linear_fp8_grouped_contiguous_swapab_swiglu", "linear_fp8_grouped_contiguous_2wg",
        "linear_fp8_grouped_contiguous_swiglu", "quantize_1x128_sorted_gather_sm90",
        "silu_chunk_mul_quantize_1x128_sorted_sm90",
    },
}


def _group_of(name: str, origin: str) -> tuple[str, bool]:
    """(group key, guessed). ``guessed`` is True when no explicit list names the op."""
    if origin == "python":
        return "python", False
    for key, members in GEMM_MEMBERS.items():
        if name in members:
            return key, False
    if name.endswith("_sm90") or name.startswith("linear_fp8_grouped"):
        return "moe_sm90", True
    if name.startswith("moe_") or "grouped" in name:
        return "moe_mxfp8", True
    if "1x32" in name or "mxfp8" in name:
        return "dense_mxfp8", True
    return "dense_fp8", True


# --------------------------------------------------------------------------
# C++ parsing
# --------------------------------------------------------------------------
_TOKEN_RE = re.compile(
    r"""
      (?P<lcomment>//[^\n]*)
    | (?P<bcomment>/\*.*?\*/)
    | (?P<string>"(?:[^"\\\n]|\\.)*")
    | (?P<char>'(?:[^'\\\n]|\\.)*')
    | (?P<other>.)
    """,
    re.VERBOSE | re.DOTALL,
)


def _unescape(lit: str) -> str:
    body = lit[1:-1]
    return bytes(body, "utf-8").decode("unicode_escape") if "\\" in body else body


def _cpp_tokens(text: str):
    """Yield (kind, value, offset) with comments dropped and whitespace collapsed."""
    for m in _TOKEN_RE.finditer(text):
        kind = m.lastgroup
        if kind in ("lcomment", "bcomment"):
            continue
        val = m.group(kind)
        if kind == "other" and val.isspace():
            continue
        yield kind, val, m.start()


def cpp_schemas(path: Path) -> list[str]:
    """The schema strings of every m.def(...) inside a TORCH_LIBRARY[_FRAGMENT](fish_scales_ops, m)
    block of ``path``, in source order. TORCH_LIBRARY_IMPL blocks register no schema and are skipped."""
    text = path.read_text()
    toks = list(_cpp_tokens(text))
    out: list[str] = []
    for blk in re.finditer(r"TORCH_LIBRARY(?:_FRAGMENT)?\s*\(\s*(\w+)\s*,\s*(\w+)\s*\)", text):
        ns, var = blk.group(1), blk.group(2)
        if ns != NAMESPACE:
            continue
        # The block body: brace matching on the comment- and string-free token stream.
        body: list[tuple[str, str, int]] = []
        depth = 0
        for tok in toks:
            kind, val, off = tok
            if off < blk.end():
                continue
            if kind == "other" and val == "{":
                depth += 1
                if depth == 1:
                    continue
            elif kind == "other" and val == "}":
                depth -= 1
                if depth == 0:
                    break
            if depth >= 1:
                body.append(tok)
        # `<var>.def(` followed by one or more adjacent string literals: the schema.
        call = f"{var}.def"
        j = 0
        while j < len(body):
            kind, val, off = body[j]
            starts_call = (kind == "other" and text.startswith(call, off)
                           and not (off > 0 and (text[off - 1].isalnum() or text[off - 1] == "_")))
            if not starts_call:
                j += 1
                continue
            k = j
            while k < len(body) and not (body[k][0] == "other" and body[k][1] == "("):
                k += 1
            k += 1
            parts = []
            while k < len(body) and body[k][0] == "string":
                parts.append(_unescape(body[k][1]))
                k += 1
            if parts:
                out.append(" ".join("".join(parts).split()))
            j = k
    return out


# --------------------------------------------------------------------------
# Python parsing
# --------------------------------------------------------------------------
def _const_str(node) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def python_schemas(root: Path) -> list[tuple[str, Path]]:
    """(full schema without namespace, file) for every torch.library.custom_op registered in the
    fish_scales_ops namespace with an explicit ``schema=`` string."""
    found: list[tuple[str, Path]] = []
    for path in sorted((root / PY_PACKAGE).rglob("*.py")):
        if "_vendor" in path.parts or "build" in path.parts:
            continue
        try:
            tree = ast.parse(path.read_text(), filename=str(path))
        except SyntaxError as e:  # pragma: no cover - a broken file is reported, not fatal
            print(f"warning: cannot parse {path}: {e}", file=sys.stderr)
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            f = node.func
            fname = f.attr if isinstance(f, ast.Attribute) else (f.id if isinstance(f, ast.Name) else "")
            if fname != "custom_op" or not node.args:
                continue
            qual = _const_str(node.args[0])
            if not qual or not qual.startswith(NAMESPACE + "::"):
                continue
            schema = None
            for kw in node.keywords:
                if kw.arg == "schema":
                    schema = _const_str(kw.value)
            name = qual.split("::", 1)[1]
            if schema is None:
                print(f"warning: {qual} in {path} has no literal schema=; listed by name only",
                      file=sys.stderr)
                found.append((f"{name}(...)", path))
                continue
            found.append((name + " ".join(schema.split()), path))
    return found


def _op_name(schema: str) -> str:
    return schema.split("(", 1)[0].strip()


# --------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------
def _marker(doc: str, which: str) -> str:
    if which == "begin":
        return (f"<!-- BEGIN GENERATED torch.ops schemas ({doc}): do not edit by hand; regenerated "
                f"from the m.def strings and the Python custom_op registrations -->")
    return f"<!-- END GENERATED torch.ops schemas ({doc}) -->"


def render(root: Path, doc: str) -> str:
    lines = [_marker(doc, "begin"), ""]
    if doc == "attention":
        schemas = cpp_schemas(root / ATTN_BINDINGS)
        lines.append(f"Registered in `{ATTN_BINDINGS.as_posix()}`:")
        lines.append("")
        lines.append("```")
        lines.extend(schemas)
        lines.append("```")
    else:
        groups: dict[str, list[str]] = {k: [] for k, _ in GEMM_GROUPS}
        for s in cpp_schemas(root / GEMM_BINDINGS):
            key, guessed = _group_of(_op_name(s), "cpp")
            if guessed:
                print(f"note: {_op_name(s)} is in no explicit group of gen_op_schemas.py; placed in "
                      f"'{key}' by its name", file=sys.stderr)
            groups[key].append(s)
        py = python_schemas(root)
        for s, _path in py:
            groups["python"].append(s)
        first = True
        for key, title in GEMM_GROUPS:
            if not groups[key]:
                continue
            if not first:
                lines.append("")
            first = False
            if key == "python":
                files = [f"`{f}`" for f in sorted({p.relative_to(root).as_posix() for _s, p in py})]
                where = files[0] if len(files) == 1 else ", ".join(files[:-1]) + " and " + files[-1]
                lines.append(f"{title}, in {where}:")
            else:
                lines.append(f"{title}, registered in `{GEMM_BINDINGS.as_posix()}`:")
            lines.append("")
            lines.append("```")
            lines.extend(groups[key])
            lines.append("```")
    lines.append("")
    lines.append(_marker(doc, "end"))
    return "\n".join(lines) + "\n"


def _split_doc(text: str, doc: str) -> tuple[str, str, str] | None:
    b, e = _marker(doc, "begin"), _marker(doc, "end")
    i = text.find(b)
    j = text.find(e)
    if i < 0 or j < 0 or j < i:
        return None
    j_end = j + len(e)
    if j_end < len(text) and text[j_end] == "\n":
        j_end += 1
    return text[:i], text[i:j_end], text[j_end:]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("root", type=Path, nargs="?", default=Path(__file__).resolve().parent.parent,
                    help="repository root, the directory holding csrc/ and python/ (default: the parent of "
                         "the scripts/ directory this file is in)")
    ap.add_argument("--doc", choices=sorted(DOCS), default="compat", help="which block to print (default: compat)")
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--check", action="store_true",
                      help="compare every generated block with docs/api/*.md; exit 1 on a difference")
    mode.add_argument("--write", action="store_true",
                      help="rewrite every generated block of docs/api/*.md in place")
    args = ap.parse_args(argv)
    root = args.root.resolve()
    for p in (GEMM_BINDINGS, ATTN_BINDINGS):
        if not (root / p).is_file():
            print(f"error: {root / p} not found; pass the repository root", file=sys.stderr)
            return 2

    if not (args.check or args.write):
        sys.stdout.write(render(root, args.doc))
        return 0

    status = 0
    for doc, rel in DOCS.items():
        path = root / rel
        text = path.read_text()
        parts = _split_doc(text, doc)
        if parts is None:
            print(f"{rel}: no generated block (markers missing)", file=sys.stderr)
            status = 1
            continue
        want = render(root, doc)
        if parts[1] == want:
            print(f"{rel}: up to date")
            continue
        if args.write:
            path.write_text(parts[0] + want + parts[2])
            print(f"{rel}: rewritten")
        else:
            import difflib
            sys.stdout.writelines(difflib.unified_diff(
                parts[1].splitlines(True), want.splitlines(True), f"{rel} (current)", f"{rel} (generated)"))
            status = 1
    return status


if __name__ == "__main__":
    sys.exit(main())
