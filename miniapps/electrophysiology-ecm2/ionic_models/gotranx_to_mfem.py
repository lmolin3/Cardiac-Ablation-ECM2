#!/usr/bin/env python3
"""Turn a gotranx ``ode2c`` header into an MFEM EP miniapp ionic-model header.

The workflow documented in README.md ends with "some minimal modifications are
required to integrate it in the MFEM API".  Those modifications are mechanical
but touch every line of a 60 kB generated file, so they are done here instead of
by hand:

  * the free functions ``rhs`` / ``monitor_values`` / ``<scheme>`` become
    ``MFEM_HOST_DEVICE static`` members of a nested ``Kernel`` struct, so the
    reaction kernels can call them from device code with no virtual dispatch;
  * the index and initialisation helpers become host-side member functions of
    the model class;
  * ``NUM_STATES`` / ``NUM_PARAMS`` / ``NUM_MONITORED`` become ``constexpr``
    members of ``Kernel``, alongside the model metadata (potential index,
    stimulus index, ...) that the reaction kernel needs at compile time.

Usage:
    python3 gotranx_to_mfem.py --config models.json

See ``models.json`` for the per-model metadata.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

# Functions that must end up inside the device-callable Kernel struct.
KERNEL_FUNCS = (
    "rhs",
    "monitor_values",
    "explicit_euler",
    "generalized_rush_larsen",
    "forward_explicit_euler",
    "forward_generalized_rush_larsen",
    "hybrid_rush_larsen",
)

# Functions that stay on the host as members of the model class.
HOST_FUNCS = (
    "parameter_index",
    "state_index",
    "monitor_index",
    "init_parameter_values",
    "init_state_values",
)


def split_functions(src: str) -> dict[str, str]:
    """Return {name: full function text} for every top-level function.

    gotranx emits every function at column 0, so a definition starts at a line
    matching ``<type> <name>(`` and ends at the matching closing brace.
    """
    out: dict[str, str] = {}
    pattern = re.compile(r"^(?:int|void|double)\s+(\w+)\s*\(", re.MULTILINE)
    for m in pattern.finditer(src):
        name = m.group(1)
        # Walk braces from the first '{' after the signature.
        i = src.index("{", m.end() - 1)
        depth, j = 0, i
        while j < len(src):
            if src[j] == "{":
                depth += 1
            elif src[j] == "}":
                depth -= 1
                if depth == 0:
                    break
            j += 1
        out[name] = src[m.start(): j + 1]
    return out


def indent(text: str, prefix: str) -> str:
    return "\n".join(prefix + ln if ln.strip() else ln for ln in text.split("\n"))


def scalar(src: str, name: str) -> int:
    m = re.search(rf"^int\s+{name}\s*=\s*(\d+)\s*;", src, re.MULTILINE)
    if m is None:
        raise SystemExit(f"could not find {name} in generated source")
    return int(m.group(1))


def lookup_index(funcs: dict[str, str], which: str, name: str) -> int:
    """Resolve a state/parameter/monitor name to its index.

    The generated ``*_index`` functions are a chain of ``strcmp`` tests, so the
    index is read straight out of the generated source.  This keeps the emitted
    ``constexpr`` metadata and the runtime name lookup from drifting apart --
    the model class also asserts they agree at construction.
    """
    body = funcs[which]
    m = re.search(rf'strcmp\(name,\s*"{re.escape(name)}"\)\s*==\s*0\)\s*\{{\s*return\s+(\d+)\s*;',
                  body)
    if m is None:
        raise SystemExit(f"'{name}' not found in {which}()")
    return int(m.group(1))


def make_kernel_struct(funcs: dict[str, str], consts: list[str]) -> str:
    parts = ["struct Kernel : IonicKernelDefaults", "{"]
    parts.append(indent("\n".join(consts), "    "))
    for fname in KERNEL_FUNCS:
        if fname not in funcs:
            continue
        body = funcs[fname]
        # Prefix the definition with MFEM_HOST_DEVICE static.
        body = re.sub(r"^(int|void|double)\s+", r"MFEM_HOST_DEVICE static \1 ", body, count=1)
        parts.append("")
        parts.append(indent(body, "    "))
    parts.append("};")
    return "\n".join(parts)


def build_header(cfg: dict, generated: Path) -> str:
    src = generated.read_text()
    funcs = split_functions(src)

    missing = [f for f in HOST_FUNCS if f not in funcs]
    if missing:
        raise SystemExit(f"{generated}: missing generated functions {missing}")

    nstates = scalar(src, "NUM_STATES")
    nparams = scalar(src, "NUM_PARAMS")
    nmonitored = scalar(src, "NUM_MONITORED")

    consts = [
        f"static constexpr int nstates = {nstates};",
        f"static constexpr int nparams = {nparams};",
        f"static constexpr int nmonitored = {nmonitored};",
        "",
    ]

    # Resolve every name the model config asks for into a compile-time index.
    resolved: dict[str, int] = {}
    for key, (which, name) in cfg.get("indices", {}).items():
        idx = lookup_index(funcs, which, name)
        resolved[key] = idx
        consts.append(f'static constexpr int {key} = {idx};   // "{name}"')

    consts.append("")
    consts.append(f'static constexpr bool dimensionless = {str(cfg["dimensionless"]).lower()};')
    consts.append(f'static constexpr real_t stim_sign = {cfg["stim_sign"]};')

    kernel = make_kernel_struct(funcs, consts)

    host_bodies = "\n\n".join(indent(funcs[f], "            ") for f in HOST_FUNCS)

    ctor = "\n".join(indent(ln, "                ") for ln in cfg["ctor_body"])

    # Interface overrides supplied per model.  These are emitted before the
    # nested Kernel struct but may refer to it: a member function body is a
    # complete-class context, so Kernel::* is visible there.
    extra = "\n".join(indent(ln, "            ") for ln in cfg.get("extra_members", []))

    cls = cfg["class"]
    base = cfg["base"]

    return f"""#pragma once

// GENERATED FILE -- do not edit by hand.
//
// Produced by gotranx_to_mfem.py from
//     {cfg["ode"]}  ->  gotranx ode2c  ->  {generated.name}
// Regenerate with:
//     python3 gotranx_to_mfem.py --config models.json
//
// {cfg["doc"]}

#include "gotranx_wrapper.hpp"

namespace mfem
{{
    namespace electrophysiology
    {{

        /** @brief {cfg["doc"]} */
        class {cls} : public {base}
        {{
        public:
            {cls}() : {base}()
            {{
{ctor}
            }}

            std::string GetName() const override {{ return "{cfg["name"]}"; }}

{extra}

{host_bodies}

            /**
             * @brief Device-callable kernel for this model.
             *
             * Holds the model math as MFEM_HOST_DEVICE *static* functions and the
             * model metadata as compile-time constants, so that the reaction
             * kernels can instantiate a templated mfem::forall over it with no
             * virtual dispatch and full inlining.
             */
{indent(kernel, "            ")}
        }};

    }} // namespace electrophysiology
}} // namespace mfem
"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, type=Path)
    args = ap.parse_args()

    cfg_path = args.config
    root = cfg_path.parent
    models = json.loads(cfg_path.read_text())

    for cfg in models:
        generated = root / cfg["generated"]
        out = root / cfg["output"]
        out.write_text(build_header(cfg, generated))
        print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
