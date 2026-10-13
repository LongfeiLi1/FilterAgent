"""Bounded synthetic validation entry point; zero SPICE/LLM calls."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import time
import unittest

import numpy as np

from . import test_interface
from .core import ContractError, local_modes
from .sk_reference import CORE_NODES


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--reference-root', type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    files = ('src/rc_agent/linear_stability.py', 'src/rc_agent/physical.py',
             'src/rc_agent/baseline.py', 'src/rc_agent/baseline_full.py',
             'src/rc_agent/stability.py', 'configs/t1_baseline_batch.json',
             'configs/t1_baseline_full.json')

    def hashes():
        return {name: hashlib.sha256((args.reference_root/name).read_bytes()).hexdigest() for name in files}

    before = hashes() if args.reference_root else {}
    references = {}
    if args.reference_root:
        config = json.loads((args.reference_root/'configs/t1_baseline_batch.json').read_text())
        references['tasks'] = [{k: row[k] for k in ('id', 'f0_hz', 'q')} for row in config['tasks']]
        assert [t['f0_hz'] for t in references['tasks']] == [12648.820123726657, 37946.46037117997]
        assert all(t['q'] == .7071067811865476 for t in references['tasks'])
        tree = ast.parse((args.reference_root/'src/rc_agent/stability.py').read_text())
        old_core = next(ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign)
                        and any(isinstance(t, ast.Name) and t.id == 'CORE_NODES' for t in n.targets))
        assert old_core == CORE_NODES
        references['legacy_core_node_match'] = True
    start = time.monotonic()
    suite = unittest.defaultTestLoader.loadTestsFromModule(test_interface)
    with (args.output/'unittest.log').open('w') as stream:
        result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    examples = []
    for name, hidden_g in [('stable_hidden', .0005), ('unstable_hidden', -.0005), ('zero_mode', 0.)]:
        g, c, layout, _ = test_interface.hidden_fixture(hidden_g)
        examples.append({'fixture': name, **local_modes(g, c, layout, layout.names)})
    g, c, layout, _ = test_interface.hidden_fixture()
    c[2, 2] = 0
    try:
        local_modes(g, c, layout, layout.names)
        raise AssertionError('rank-deficient charge incorrectly accepted')
    except ContractError as exc:
        examples.append({'fixture': 'unexplained_algebraic_state', 'status': 'rejected', 'reason': str(exc)})
    after = hashes() if args.reference_root else {}
    report = {'scope': 'synthetic interface checks, not nine-Case benchmark implementation',
              'tests_run': result.testsRun, 'failures': len(result.failures), 'errors': len(result.errors),
              'skipped': len(result.skipped), 'success': result.wasSuccessful() and before == after,
              'wall_seconds': time.monotonic()-start, 'spice_processes': 0, 'llm_calls': 0,
              'case_gates_passed': [], 'real_topology_simulations': [],
              'reference_checks': references, 'legacy_sha256_before': before,
              'legacy_sha256_after': after, 'legacy_files_unchanged': before == after,
              'synthetic_evidence': examples}
    (args.output/'result.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps({'output': str(args.output), 'success': report['success'], 'tests_run': result.testsRun,
                      'failures': len(result.failures), 'errors': len(result.errors)}))
    return 0 if report['success'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
