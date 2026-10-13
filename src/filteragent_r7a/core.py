"""Instance-owned refinement policy. All physical observations stay in CommonExecution."""
import copy
import math
from dataclasses import asdict
from types import MethodType
from .policy import BASE_SHA, digest, sha

def valid(row):
    return bool(row and row.get('evidence') and row.get('measurement_status', 'valid') == 'valid')

def context(backend):
    from rc_agent import physical
    # Per-instance only. Immutable PDK dependency closure is sealed at freeze.
    from .policy import HERE, read
    dependencies = read(HERE/'pdk_manifest.json')['files']
    return {'case': backend.case, 'topology_binding': getattr(backend, 'binding', None),
        'task': getattr(backend, 'task', None), 'bounds': backend.bounds,
        'acceptance': backend.common, 'protocol': backend.engine.protocol,
        'optimizer': backend.optimizer_settings, 'base_source': BASE_SHA,
        'PDK_dependencies': dependencies, 'PDK_path': str(physical.PDK),
        'execution_seed':0, 'simulator':read(HERE/'pdk_manifest.json')['simulator'],
        'conditions': 'TT; VDD1.8; VinVref1.2; unchanged source/load and analysis decks'}

def key(backend):
    row = backend.current
    # Every RC/device coordinate, candidate version and saved actual measurement
    # are included. Descriptor/model/target versions prevent stale math reuse.
    return digest({'context': backend._refinement_context, 'observation': row,
        'descriptor_revision': len(backend.auxiliary),
        'descriptor_anchor': backend.descriptor_anchor,
        'current_paid_descriptor': None if not row or descriptor_raw_key(backend,row['parameters']) not in backend.descriptors else asdict(backend.descriptors[descriptor_raw_key(backend,row['parameters'])][0]),
        'plan_eta': None if backend.plan_eta is None else asdict(backend.plan_eta),
        'model_instance_revision': id(backend.plan_calibration),
        'calibration_revision': backend.calibrations,
        'fixed_targets': getattr(backend, 'fixed_allocation_targets', None),
        'adaptive_target_feedback': {k:getattr(backend, 'adaptive_target_feedback', {}).get(k) for k in ('revision','eta_penalty_weights')}})

def model_from_bench(backend, x, eta, evidence):
    if backend.case['type_id'] != 'T1':
        return backend.calibration_models[backend.descriptor_key(x)]
    from rc_agent import physical
    return physical.calibration_from_evidence(eta, evidence)

def descriptor_raw_key(backend, x):
    if backend.case['type_id'] != 'T1':
        return backend.descriptor_key(x)
    from filteragent.backend import value_hash, DEVICE_NAMES
    return value_hash({n:float(x[n]) for n in DEVICE_NAMES})

def descriptor_deck(backend, x):
    if backend.case['type_id'] == 'T1':
        from orchestration.filteragent_v3.m1_type_v3.characterization import deck
        return deck(backend.devices(x))  # exact native T1 wide-grid bytes
    from filteragent_type_v3.characterization import deck
    return deck(backend.devices(x),backend.case,x)

def descriptor_render_key(backend, x):
    import hashlib
    return digest({'context':backend._refinement_context,
        'exact_rendered_deck_sha256':hashlib.sha256(descriptor_deck(backend,x).encode()).hexdigest()})

def install_rendered_descriptor_cache(backend):
    native=backend.characterize
    cache={}
    backend._rendered_descriptor_cache=cache
    def characterize(self,x,*,fresh=False):
        rendered=descriptor_render_key(self,x)
        raw=descriptor_raw_key(self,x)
        if not fresh and rendered in cache:
            result,model=cache[rendered]
            self.descriptors[raw]=result
            if self.case['type_id']!='T1':
                self.calibration_models[raw]=model
            self.refinement_audit['rendered_OTA_cache_hits']+=1
            self.event('exact_rendered_OTA_cache_reuse',{'render_key':rendered,
                'source_evidence':result[1]['evidence'],'new_physical_candidate':False,
                'new_SPICE':0,'fresh':False,'no_float_rounding':True})
            return result
        result=native(x,fresh=fresh)
        model=None if self.case['type_id']=='T1' else self.calibration_models[raw]
        cache[rendered]=(result,model)
        return result
    backend.characterize=MethodType(characterize,backend)

def install_bootstrap(backend):
    from filteragent.backend import analytic_initial
    from filteragent_fallback_v2.backend import initial_pass
    audit = {'enabled': True, 'triggered': False, 'skipped_OTA_requests': 0,
        'endpoint_still_requires_fresh_measurement': True,
        'planning_bootstrap_not_available_when_skipped': False, 'reason': None}
    backend.bootstrap_optimization_audit = audit
    def initialize(self):
        typed = self.case['type_id'] != 'T1'
        if not self.room(3, 13 if typed else 12, 2):
            from filteragent.backend import ToolUnavailable
            raise ToolUnavailable('initial_shared_allowance_unavailable')
        x = dict(self.binding['initial']) if typed else analytic_initial(self.task, self.bounds)
        self.current = self.evaluate(x)
        self.event('unchanged_analytic_initial_full_measurement', self.current)
        if initial_pass(self):
            audit.update(triggered=True, skipped_OTA_requests=1,
                planning_bootstrap_not_available_when_skipped=True,
                reason='unchanged complete initial_PASS; fresh endpoint remains mandatory')
            self.event('planning_bootstrap_deferred_after_full_initial_PASS', audit)
        elif valid(self.current):
            bootstrap(self, self.current)
        return self.current
    backend.initialize = MethodType(initialize, backend)
    native_accept = backend.accept_trial
    def accept_trial(self, row, *, planned_eta=None, planned_model=None, force=False):
        from filteragent_fallback_v2.backend import initial_pass
        from types import SimpleNamespace
        # Frozen native snapshots admit repair tools only for status != success.
        # The original complete-PASS predicate additionally requires complete
        # valid evidence and no failed constraints. Hence the next legal action
        # is finish, with no further design decision or model needed.
        if self.engine.status!='paused' and initial_pass(SimpleNamespace(current=row)):
            previous=self.current
            cached=(descriptor_raw_key(self,row['parameters']) in self.descriptors or
                descriptor_render_key(self,row['parameters']) in self._rendered_descriptor_cache)
            self.current=row
            self.plan_eta=planned_eta if planned_eta is not None else self.plan_eta
            self.plan_calibration=planned_model if planned_model is not None else self.plan_calibration
            self.last_diagnostic=None  # Never claim an old-point diagnostic for the new point.
            reason='complete valid full PASS and frozen native next-actions exactly finish; unused planning-only diagnosis deferred; fresh endpoint mandatory'
            self.refinement_audit['terminal_PASS_diagnostic_deferrals']+=1
            self.event('complete_PASS_trial_diagnosis_deferred',{
                'candidate_id':row['candidate_id'],'descriptor_request_skipped':True,
                'paid_descriptor_already_cached':cached,'actual_new_OTA':0,
                'skipped_reason':reason,'previous_costs_not_modified':True,
                'full_filter_evaluation_already_complete':True,'fresh_endpoint_required':True})
            return {'selected':True,'rollback':False,'previous_candidate':previous['candidate_id'],
                'trial':row,'trial_diagnostics':None,'trial_diagnostic_error':None,
                'diagnostic_deferred':True,'diagnostic_skipped_reason':reason,
                'next_legal_actions':['finish']}
        # A first valid probe after an invalid initial point may establish the
        # model from its independently measured bench. No full-response fit.
        if valid(row) and planned_model is None:
            self._capture_bootstrap_target = self.plan_calibration is None
            self._bootstrap_capture_row = row
            try:
                eta, evidence = self.characterize(row['parameters'])
                planned_model = model_from_bench(self, row['parameters'], eta, evidence)
                planned_eta = planned_eta or eta
            finally:
                self._capture_bootstrap_target = False
                self._bootstrap_capture_row = None
        return native_accept(row, planned_eta=planned_eta, planned_model=planned_model, force=force)
    backend.accept_trial = MethodType(accept_trial, backend)

def bootstrap(backend, row):
    backend._capture_bootstrap_target = True
    backend._bootstrap_capture_row = row
    try:
        eta, evidence = backend.characterize(row['parameters'])
        backend.plan_eta = eta
        backend.plan_calibration = model_from_bench(backend, row['parameters'], eta, evidence)
    finally:
        backend._capture_bootstrap_target = False
        backend._bootstrap_capture_row = None
    backend.last_diagnostic = None
    backend.event('first_valid_independent_model_bootstrap', {
        'candidate_id': row['candidate_id'], 'independent_OTA_evidence': evidence['evidence'],
        'full_filter_fit': False, 'extra_SPICE': 0,
        'one_pole_diagnostic_before_known_A_Z_removed': True})
    return backend.diagnose()

def install_diagnosis_cache(backend):
    native = backend.diagnose
    cache = {}
    def diagnose(self):
        k = key(self)
        if k in cache:
            self.refinement_audit['diagnostic_math_cache_hits'] += 1
            self.last_diagnostic = copy.deepcopy(cache[k])
            return self.last_diagnostic
        value = native()
        # characterize may append a newly charged descriptor: use post-call key.
        cache[key(self)] = copy.deepcopy(value)
        self.refinement_audit['diagnostic_math_computations'] += 1
        return value
    backend.diagnose = MethodType(diagnose, backend)

def install_sizing_stop(backend):
    from filteragent_provider_stream_v1.provider import injected
    from filteragent.backend import SizingBlockComplete
    native = backend.size_devices.__func__
    powell = native.__globals__['native_powell']
    def bounded_powell(objective, z0, *args, **kwargs):
        import numpy as np
        from .action_preview import paid_power_seed
        raw=descriptor_raw_key(backend,backend.current['parameters'])
        preview=paid_power_seed(backend,backend._active_size_plan,raw,z0=z0)
        initial=np.asarray(preview['seed_log_parameters'],float)
        if preview['power_ratio_rule_available']:
            backend.refinement_audit['power_ratio_seeds']+=int(preview['power_ratio_rule_applied'])
            backend.event('generic_paid_power_ratio_device_seed',preview)
        def measured_objective(z):
            value = objective(z)
            # Native objective has already updated its measured best. Stop only
            # if that selected best is ready, never merely a tied nonselected
            # objective point (zero loss alone lacks a DC gate).
            import numpy as np
            cells = dict(zip(objective.__code__.co_freevars, objective.__closure__))
            best = cells['best'].cell_contents
            plan = backend._active_size_plan
            ratio = np.abs(np.log(np.array([best['eta'][k] for k in ('a0','wp_rad_s','ro_ohm')])/
                np.array([plan['ota_target'][k] for k in ('a0','wp_rad_s','ro_ohm')])))
            ready = bool(np.all(ratio <= [.25,.25,.35]) and
                best['power_w'] <= plan['power_target_w'] and
                best['minimum_saturation_margin_v'] >= plan['saturation_target_v'] and
                backend.common['dc_out_range_v'][0] <= best['dc_out'] <= backend.common['dc_out_range_v'][1])
            if ready:
                backend.refinement_audit['device_target_early_stops'] += 1
                raise SizingBlockComplete('independently_measured_device_target_ready_full_filter_still_required')
            return value
        return powell(measured_objective, initial, *args, **kwargs)
    sized = injected(native, native_powell=bounded_powell).__get__(backend, type(backend))
    def size_devices(self, plan, max_objectives=12):
        self._active_size_plan = plan
        try:
            return sized(plan, max_objectives=max_objectives)
        finally:
            self._active_size_plan = None
    backend.size_devices = MethodType(size_devices, backend)

def install_allocation(backend):
    native = backend.allocation
    cache = {}
    def allocation(self, coupled, radius='normal'):
        k = (key(self), bool(coupled), radius)
        if k in cache:
            self.refinement_audit['allocation_math_cache_hits'] += 1
            return copy.deepcopy(cache[k])
        result = native(coupled, radius)
        cache[(key(self), bool(coupled), radius)] = copy.deepcopy(result)
        self.refinement_audit['allocation_math_computations'] += 1
        return result
    backend.allocation = MethodType(allocation, backend)

def guidance(backend, state):
    from filteragent.backend import ToolUnavailable
    diagnostic = state.get('diagnostics') or {}
    plans = []
    if diagnostic.get('model_trusted'):
        for name, coupled in (('rc_step', False), ('coupled_step', True)):
            if name not in state['allowed_actions']:
                continue
            for radius in ('small', 'normal'):
                try:
                    p = backend.allocation(coupled, radius)
                    from .action_preview import compare_allocation, paid_power_seed
                    comparison=compare_allocation(p)
                    implementation=None if not coupled else paid_power_seed(backend,p,descriptor_raw_key(backend,backend.current['parameters']))
                    n = len(p['passives'])
                    jac = p['model_jacobian']
                    plans.append({'action': name, 'radius': radius,
                        **comparison,
                        'RC_certified_margin': p['rc_only_feasible'] if not coupled else None,
                        'coupled_device_implementation_preview':implementation,
                        'coupled_legality_is_model_domain_budget_not_RC_feasibility':coupled,
                        'passive_column_support': [sum(abs(row[k]) for row in jac) for k in range(n)],
                        'OTA_power_column_support': [sum(abs(row[k]) for row in jac) for k in range(n, len(jac[0]))],
                        'candidate_passives': p['passives'], 'OTA_target': p['ota_target'],
                        'power_target_w': p['power_target_w'],
                        'local_delta_z': p['proposal']['delta_z'],
                        'unsupported_gates_require_real_SPICE': True})
                except (ValueError, RuntimeError, OSError, ToolUnavailable) as exc:
                    plans.append({'action': name, 'radius': radius, 'available': False,
                                  'reason': type(exc).__name__})
    components = diagnostic.get('trust_band_component_errors')
    dominant = None if not components else ['model', 'realization', 'planning'][max(range(3), key=lambda i: components[i])]
    ranked = sorted((p for p in plans if 'predicted_reduction' in p),
        key=lambda p: (-p['predicted_reduction'], p['action'] != 'rc_step', p['radius'] != 'small'))
    return {'dominant_trust_band_residual': dominant, 'alternatives': plans,
        'predicted_candidate_order': [{'action': p['action'], 'radius': p['radius']} for p in ranked],
        'controller_instructions': 'Use measured residuals and alternative sensitivities to choose RC versus OTA and small versus normal. Ranking compares the same route margin before/after. Coupled linear allocation and the device realization goal/first seed are separate stages: an allocated power goal and proportional seed estimate are not a measured OTA or complete circuit prediction. RC certification applies only to rc_step. The seed may fail and the native device block may use up to 12 objective calls; no extra SPICE was run to preview it. Predictions are not acceptance. Prefer RC when its margin is certified. Characterize reuses only the exact already paid independent bench; calibrate means fresh independent OTA plus same-point full repeat and should address mismatch, not a missing initial model. A device target-ready stop still requires actual full circuit and fresh endpoint.',
        'rejected_last_step': state.get('last_tool_result', {}).get('rollback') is True}

def install_snapshot(backend):
    native = backend.snapshot
    def snapshot(self, step):
        state = native(step)
        state['repair_guidance'] = guidance(self, state)
        from .action_preview import measured_probe_preview
        state['shared_measured_probe_preview']=measured_probe_preview(self)
        state.pop('state_id', None)
        state['state_id'] = digest(state)
        return state
    backend.snapshot = MethodType(snapshot, backend)

def install(backend):
    from .adaptive import install as install_adaptive
    install_adaptive(backend, enabled=True)
    backend._refinement_context = context(backend)
    backend.refinement_audit = {'diagnostic_math_cache_hits': 0, 'diagnostic_math_computations': 0,
        'allocation_math_cache_hits': 0, 'allocation_math_computations': 0,
        'device_target_early_stops': 0, 'rendered_OTA_cache_hits':0, 'power_ratio_seeds':0,
        'terminal_PASS_diagnostic_deferrals':0, 'cache_scope': 'new worker instance; no historical/cross-case observations',
        'physical_cache_changed': 'exact rendered OTA deck equivalence, fresh bypass', 'accounting_changed':False}
    install_rendered_descriptor_cache(backend)
    install_bootstrap(backend)
    install_diagnosis_cache(backend)
    install_sizing_stop(backend)
    install_allocation(backend)
    install_snapshot(backend)
    from .scheduler import install as install_scheduler
    install_scheduler(backend)
    return backend
