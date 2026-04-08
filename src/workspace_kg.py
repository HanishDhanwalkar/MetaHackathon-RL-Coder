from __future__ import annotations

import ast
import re
import hashlib
from dataclasses import dataclass, field
from typing import Any, List, Optional, Dict, Set, Tuple

def _stable_struct_signature(source: str) -> str:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        raw = re.sub(r"\s+", "", source[:8000])
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:32]
    
    parts: List[str] = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            parts.append(f"fn{node.name}:{node.lineno})")
        elif isinstance(node, ast.ClassDef):
            n = ",".join(
                sorted(
                    n.name
                    for n in node.body
                    if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
                )
            )
            parts.append(f"class{node.name}:{n}")
        elif isinstance(node, ast.Import):
            for a in node.names:
                parts.append(f"im:{a.name}")
        elif isinstance(node, ast.ImportFrom):
            mod  = node.module or ""
            names = ",".join(sorted(n.name for n in node.names))
            parts.append(f"if:{mod}:{names}")
        elif isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name):
                    parts.append(f"as:{t.id}")
    parts.sort()
    return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()[:32]

def _line_at_offset(source: str, offset: int) -> int:
    return source.count("\n", 0, max(0, min(offset, len(source)))) + 1

@dataclass
class WorkspaceKG:
    """AST KG for single python buffer"""
    
    _struct_sig: Optional[str] = None
    _symbols: Dict[str, str] = field(default_factory=dict)
    _imports: Set[str] = field(default_factory=set)
    _calls: List[Tuple[str, int]] = field(default_factory=list)
    _last_source: str = ""
    
    def update(self, source: str)-> Dict[str, Any]:
        """rebuild KG"""
        
        prev = self._struct_sig
        new_sig = _stable_struct_signature(source)
        line_delta = 0
        
        if self._last_source:
            line_delta = abs(source.count("\n") - self._last_source.count("\n"))
            
        char_delta = abs(len(source) - len(self._last_source))
        
        major = False
        
        if prev is None:
            major = True
        elif new_sig is None:
            major = True
        elif line_delta >= max(8, int(0.12 * max(1, self._last_source.count("\n")))):
            major = True
        elif char_delta >= 400:
            major = True
        
        self._struct_sig = new_sig
        self._last_source = source
        self._symbols.clear()
        self._imports.clear()
        self._calls.clear()
        
        try:
            tree = ast.parse(source)
        except SyntaxError:
            return {
                "major_changed": major,
                "symbol_count": 0,
                "import_count": 0,
                "valid_parse": False,
            }
        
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                self._symbols[node.name] = f"fn (line {node.lineno})"
            elif isinstance(node, ast.ClassDef):
                self._symbols[node.name] = f"class (line {node.lineno})"
            elif isinstance(node, ast.Import):
                for a in node.names:
                    self._imports.add(f"import {a.name}")
            elif isinstance(node, ast.ImportFrom):
                mod  = node.module or ""
                for a in node.names:
                    self._imports.add(f"from {mod} import {a.name}")
            elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                self._calls.append((node.func.id, _line_at_offset(source, node.lineno)))
        return {
            "major_changed": major,
            "symbol_count": len(self._symbols),
            "import_count": len(self._imports),
            "valid_parse": True,
        }
                
    def context_lines(
        self,
        source: str,
        c_offset: int,
        max_items: int = 14
    ) -> List[str]:
        """model friendly slices of KG"""
        if source != self._last_source:
            self.update(source)
            
        cur_line = _line_at_offset(source, c_offset)
        lines: List[str] = []
        
        if self._imports:
            imp = sorted(self._imports)
            lines.append("imports: " + "; ".join(imp[:6]))
            if len(imp) > 6:
                lines.append(f"... + {len(imp) - 6} more imports")
                
        nearby_calls: List[str] = []
        for name, ln in self._calls:
            if ln and abs(ln - cur_line) <= 6:
                nearby_calls.append(name)
                
        if nearby_calls:
            uniq = list(dict.fromkeys(nearby_calls))
            lines.append("calls_near_cursor: " + "; ".join(uniq))
                
        near_scope: List[str] = []
        for n, desc in self._symbols.items():
            m = re.search(r"line (\d+)", desc)
            if m and abs(int(m.group(1)) - cur_line) <= 4:
                near_scope.append(n)
                
        if near_scope:
            lines.append(
                "symbols_this_region: " + ", ".join(f"{n} ({self._symbols[n]})" for n in near_scope[:6])
            )
            
        if len(self._symbols) > 0 and len(lines) < max_items:
            defs = [f"{n} - {d}" for n, d in list(self._symbols.items())[:max_items]]
            lines.append(
                "definitions: " + " | ".join(defs[:6])
            )            
        return lines[:max_items]