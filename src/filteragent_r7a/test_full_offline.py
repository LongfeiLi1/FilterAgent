"""Existing Full regression fixtures; external I/O is denied."""
import copy,json,math,socket,subprocess,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
from . import core as iv
from .policy import *
from .loop import run_loop as full_loop
from filteragent_fallback_v2.backend import initial_pass
from filteragent_fallback_v2.evidence import save
from filteragent_release.paths import CODE_ROOT as ROOT
from filteragent_release.cli import construct as public_construct,load_cases
OUTPUT=ROOT/'output'
R8=ROOT/'src/filteragent_bootstrap_optimization_v1'
CONFIGS=('full_on',)
class NoProvider:
    is_mock=True
    actual_retries=0
    def decide(self,*a,**k):raise AssertionError('No paid provider allowed in fixture')
def deny(*a,**k):raise AssertionError('offline_external_IO_forbidden')
def check(arm,cid,cohort):
    assert arm=='full_on'
    case=next(c for c in load_cases() if c['case_id']==cid)
    return {'case':case,'paths':{'out':OUTPUT/'offline'},'config':{'factors':{}},'protocol':json.loads((ROOT/'src/filteragent_bootstrap_optimization_v1/protocols'/('T1.json' if case['type_id']=='T1' else 'Type.json')).read_text())}
def construct(cfg,executor=None):return public_construct(cfg['case'],cfg['paths']['out'],'ngspice',executor)
class FixtureClient(NoProvider):
    """Explicitly mock decision, neither transport nor measured-result replay."""
    is_mock = True
    def __init__(self, actions, engine=None):
        self.actions = iter(actions)
        self.calls = 0
        self.engine = engine
    def decide(self, state, **kwargs):
        from filteragent_provider_v2.diagnostic import contract
        self.calls += 1
        name, args = next(self.actions, ('finish', {}))
        if self.engine is not None:
            # Synthetic API accounting only. No request/credential/transport.
            if len(self.engine.api_calls) >= self.engine.protocol['max_llm_calls_per_case']:
                raise AssertionError('synthetic_API_budget_exhausted')
            self.engine.api_calls.append({'status': 'completed', 'model': 'qwen3.8-flash',
                'prompt_tokens': 1, 'completion_tokens': 2, 'total_tokens': 3,
                'wall_seconds': 0., 'is_mock': True})
            self.engine.save()
        return contract(state['case']['type_id']).scripted_decision(state, name, args)

class Tests(unittest.TestCase):
    def setUp(self):
        OUTPUT.mkdir(exist_ok=True)
        self.tmp=tempfile.TemporaryDirectory(dir=OUTPUT,prefix='offline_fixture_')
        self.folder=Path(self.tmp.name);self.addCleanup(self.tmp.cleanup)
        dummy=self.folder/'offline_model_placeholder.spice';dummy.write_text('* Rendering fixture only; never simulated.\n')
        guard=patch('rc_agent.physical.PDK',dummy);guard.start();self.addCleanup(guard.stop)
        for target in ['socket.socket','socket.create_connection','subprocess.Popen','rc_agent.qwen_adapter.QwenConnectivityAdapter.from_existing_config']:
            guard=patch(target,side_effect=deny);guard.start();self.addCleanup(guard.stop)
    def backend(self, type_id='T1', arm='full_on', tag='b'):
            cid = 'V3-'+type_id+'-L1-C01'
            cfg = check(arm, cid, 'formal')
            cfg['paths'] = {**cfg['paths'], 'out': self.folder/(tag+'_'+type_id+'_'+arm)}
            return construct(cfg, executor=deny)

    def synthetic_row(self, b, params, status='success'):
            evidence = {'performance': 'synthetic_no_files', 'matrix': 'synthetic_no_files'} if b.case['type_id'] != 'T1' else 'synthetic_no_files'
            return {'candidate_id': 0, 'parameters': copy.deepcopy(params), 'endpoint_check': False,
                'measurement_status': 'valid', 'status': status,
                'failed_constraints': [] if status == 'success' else ['synthetic_gate'],
                'loss': 0. if status == 'success' else 1., 'fom': 1., 'evidence': evidence,
                'metrics': {'power_w': 1e-6, 'stable': True, 'full_stability_status': 'pass_numerical_local',
                            'full_local_stability': 'stable'}, 'errors': []}

    def fixture_measurements(self, b, status='success', endpoint_status='success'):
            calls = []
            def evaluate(params, endpoint=False):
                calls.append((copy.deepcopy(params), endpoint))
                row = self.synthetic_row(b, params, endpoint_status if endpoint else status)
                if endpoint:
                    c, _ = b.engine.candidate(dict(params), endpoint=True)
                    b.engine.endpoint = c['id']
                    c.update(measured=True, accepted=row['status']=='success', finalized=True, status='measured')
                    row['candidate_id'], row['endpoint_check'] = c['id'], True
                    b.endpoint = row
                    b.engine.status = 'success' if row['status']=='success' else 'circuit_fail'
                else:
                    b.rows.append(row)
                    b.best = row
                    if row['status'] == 'success':
                        b.first = row
                return row
            b.evaluate = evaluate
            return calls

    def test_endpoint_failure_never_uses_initial_PASS(self):
            for arm in CONFIGS:
                b = self.backend('T2', arm)
                calls = self.fixture_measurements(b, endpoint_status='circuit_fail')
                b.characterize = deny
                loop = full_loop
                r = loop(b, NoProvider())
                self.assertEqual(len(calls), 2)
                self.assertFalse(r['method_success'])
                self.assertFalse(r['measurement_shortcut_success'])

    def test_native_engine_initial_characterization_retry_endpoint_all_charged(self):
            from common_execution import CommonExecution, ExecutionPaused
            protocol = read(R8/'protocols/T1.json')
            calls = []
            def mock_executor(argv, cwd, timeout):
                calls.append((argv, str(cwd), timeout))
                if len(calls) <= 2:
                    raise subprocess.TimeoutExpired(argv, timeout, output='synthetic_timeout')
                return SimpleNamespace(returncode=0, stdout='synthetic_mock_executor', stderr='')
            e = CommonExecution(self.folder/'ledger', protocol, 'L1', 'NO_REAL_SPICE', executor=mock_executor)
            initial, _ = e.candidate({'role': 'synthetic_initial'})
            deck = 'synthetic_fixture\n'+'op\n'*5
            first = e.run_spice(initial, 'performance', deck)
            self.assertEqual(first['attempts'], 3)
            e.run_spice(initial, 'matrix', deck)
            e.finish_candidate(initial, measured=True)
            ota, _ = e.candidate({'role': 'synthetic_OTA_characterization'})
            e.run_spice(ota, 'OTA', 'synthetic_fixture\nop\nac dec 1 1 10\n')
            e.finish_candidate(ota, measured=True)
            endpoint, _ = e.candidate({'role': 'synthetic_initial'}, endpoint=True)
            e.run_spice(endpoint, 'performance', deck)
            e.run_spice(endpoint, 'matrix', deck)
            e.finish_candidate(endpoint, measured=True, accepted=True, endpoint_evidence='synthetic_fixture')
            ledger = read(e.directory/'budget.json')
            self.assertEqual(ledger['new_unique_candidate_attempts'], 2)
            self.assertEqual(ledger['actual_started_spice_processes'], 7)  # mock only
            self.assertEqual(ledger['attempted_explicit_analysis_commands'], 32)
            self.assertEqual(sum(not s['endpoint'] for s in e.slots), 3)
            self.assertEqual(sum(s['endpoint'] for s in e.slots), 2)
            self.assertTrue(all(call[2] == 20 for call in calls))
            with self.assertRaisesRegex(ExecutionPaused, 'already attempted'):
                e.candidate({'role': 'synthetic_initial'}, endpoint=True)

    def test_native_candidate_caps_all_levels_preserve_endpoint(self):
            from common_execution import CommonExecution, CandidateBudgetExhausted
            for level, cap in (('L1', 120), ('L2', 240), ('L3', 360)):
                e = CommonExecution(self.folder/level, read(R8/'protocols/T1.json'), level, 'NO_REAL_SPICE', executor=deny)
                for i in range(cap):
                    e.candidate({'synthetic_unique': i})
                with self.assertRaises(CandidateBudgetExhausted):
                    e.candidate({'synthetic_unique': cap})
                endpoint, cached = e.candidate({'synthetic_unique': 0}, endpoint=True)
                self.assertFalse(cached)
                self.assertEqual(len(e.cache), cap)
                self.assertEqual(endpoint['role'], 'endpoint')

    def test_frozen_recovery_and_accounting_limits_unchanged(self):
            from filteragent_fallback_interface_repair_v1.recovery import MAX_CORRECTIONS, MAX_RULE_ACTIONS, budget_view
            self.assertEqual(MAX_CORRECTIONS, 3)
            self.assertEqual(MAX_RULE_ACTIONS, 1)
            for arm in CONFIGS:
                b = self.backend('T1', arm)
                b.engine.api_calls = [dict(prompt_tokens=1, completion_tokens=2, total_tokens=3) for i in range(32)]
                view = budget_view(b, NoProvider())
                self.assertEqual(view['remaining']['API_calls'], 0)
                self.assertEqual(view['usage']['total_tokens']['known_subtotal'], 96)
                self.assertTrue(view['usage']['total_tokens']['complete'])
                self.assertTrue(view['endpoint_reserve_preserved'])
                self.assertEqual(b.engine.protocol['api_timeout_seconds'], 300)

    def test_full_allocation_math_equal_R8_for_identical_inputs(self):
            from filteragent.backend import T1Backend
            from filteragent_type_v3.backend import TypeBackend
            for t in ('T1','T2','T3'):
                b=self.backend(t);self.numerical_fixture(b);b.initialize()
                native=(T1Backend if t=='T1' else TypeBackend).allocation
                for coupled in (False,True):
                    for radius in ('small','normal'):
                        expected=native(b,coupled,radius)
                        actual=b.allocation(coupled,radius)
                        self.assertIn('target_penalty_feedback',actual)
                        self.numeric_equal({k:v for k,v in actual.items() if k!='target_penalty_feedback'},expected)
                        self.assertEqual(b.allocation(coupled,radius),actual)
                self.assertGreater(b.refinement_audit['allocation_math_cache_hits'],0)

    def test_cache_key_invalidates_all_semantic_coordinates_and_versions(self):
            from rc_agent.models import OTA
            b=self.backend('T3');self.numerical_fixture(b);b.initialize()
            original=iv.key(b)
            for name in b.bounds:
                previous=b.current['parameters'][name]
                b.current['parameters'][name]*=math.exp(.001)
                self.assertNotEqual(iv.key(b),original)
                b.current['parameters'][name]=previous
            for change,restore in (
                (lambda:setattr(b,'plan_eta',OTA(1001.,1e7,100.)),lambda:setattr(b,'plan_eta',OTA(1000.,1e7,100.))),
                (lambda:setattr(b,'calibrations',1),lambda:setattr(b,'calibrations',0)),
                (lambda:b.auxiliary.append({'fixture':1}),lambda:b.auxiliary.pop()),
                (lambda:b.current.update(candidate_id=999),lambda:b.current.update(candidate_id=1))):
                change();self.assertNotEqual(iv.key(b),original);restore()
            before=iv.key(b);b.plan_calibration=object();self.assertNotEqual(iv.key(b),before)
            before=iv.key(b);b.fixed_allocation_targets={'eta':{'a0':1000.}};self.assertNotEqual(iv.key(b),before)
            self.assertIn('PDK_dependencies',b._refinement_context)
            self.assertEqual(b._refinement_context['case'],b.case)
            self.assertEqual(b._refinement_context['protocol'],b.engine.protocol)

    def test_same_diagnostic_cache_numerically_equal_and_fresh_calibration_bypasses(self):
            for t in ('T1','T2','T3'):
                b=self.backend(t);self.numerical_fixture(b);b.initialize()
                first=copy.deepcopy(b.diagnose());count=b.refinement_audit['diagnostic_math_computations']
                self.assertEqual(b.diagnose(),first)
                self.assertEqual(b.refinement_audit['diagnostic_math_computations'],count)
                b.calibrate()
                self.assertGreater(b.refinement_audit['diagnostic_math_computations'],count)

    def test_first_valid_probe_uses_independent_model_without_calibration_repeat(self):
            for t in ('T1','T2','T3'):
                b=self.backend(t);self.numerical_fixture(b)
                b.current=self.synthetic_row(b,self.initial_params(b),'unknown')
                b.current.update(evidence=None,measurement_status='error',loss=1e9)
                row=b.evaluate(self.initial_params(b))
                result=b.accept_trial(row)
                self.assertTrue(result['selected']);self.assertIsNotNone(b.plan_calibration)
                self.assertTrue(result['trial_diagnostics']['model_trusted'])
                self.assertEqual(b.calibrations,0)

    def test_native_physical_evaluate_finish_execute_and_calibrate_unchanged(self):
            from filteragent.backend import T1Backend
            from filteragent_type_v3.backend import TypeBackend
            for arm in CONFIGS:
                for t in ('T1','T2','T3'):
                    b=self.backend(t,arm);cls=T1Backend if t=='T1' else TypeBackend
                    self.assertIs(b.scheduler_original_execute.__func__,cls.execute)
                    for name in ('evaluate','finish','repeat_full'):
                        self.assertIs(getattr(b,name).__func__,getattr(cls,name))
                    if True:self.assertIs(b.calibrate.__func__,cls.calibrate)

    def test_sizing_earlystop_only_actual_selected_best_ready_tie_DC(self):
            from types import MethodType
            from filteragent.backend import T1Backend
            from filteragent_type_v3.backend import TypeBackend
            from rc_agent.models import OTA
            import numpy as np
            for t in ('T1','T2','T3'):
                for first_bad in (False,True):
                    b=self.backend(t,tag='tie'+str(first_bad));self.numerical_fixture(b)
                    b.current=self.synthetic_row(b,self.initial_params(b),'circuit_fail')
                    eta=OTA(1000.,1e7,100.);seen=[]
                    def measured(self,x,fresh=False):
                        seen.append(copy.deepcopy(x))
                        return eta,{'evidence':'synthetic','power_w':1e-6,
                            'minimum_saturation_margin_v':.2,
                            'dc_out':1.25 if first_bad and len(seen)==1 else 1.2}
                    b.characterize=MethodType(measured,b)
                    cls=T1Backend if t=='T1' else TypeBackend
                    b.size_devices=MethodType(cls.size_devices,b)
                    def driver(objective,z,*a,**k):
                        objective(z)
                        objective(z+np.array([.01,0,0]))
                        return None
                    module='filteragent.backend' if t=='T1' else 'filteragent_type_v3.backend'
                    with patch(module+'.native_powell',side_effect=driver):
                        iv.install_sizing_stop(b)
                        passives={k:v for k,v in b.current['parameters'].items() if k not in ('current_scale','output_scale','cc_pf')}
                        returned=b.size_devices({'passives':passives,'ota_target':dict(a0=1000.,wp_rad_s=1e7,ro_ohm=100.),
                            'power_target_w':1e-6,'saturation_target_v':.1})
                    self.assertEqual(returned,seen[0])
                    self.assertEqual(len(seen),2 if first_bad else 1)
                    self.assertEqual(b.refinement_audit['device_target_early_stops'],0 if first_bad else 1)

    def test_diagnostic_cache_matches_native_math_all_three_types(self):
            from filteragent.backend import T1Backend
            from filteragent_type_v3.backend import TypeBackend
            for t in ('T1','T2','T3'):
                b=self.backend(t);self.numerical_fixture(b);b.initialize()
                expected=(T1Backend if t=='T1' else TypeBackend).diagnose(b)
                self.assertEqual(b.diagnose(),expected)

    def test_exact_rendered_descriptor_key_no_float_rounding_and_DC_preserved(self):
            import numpy as np
            for t in ('T1','T2','T3'):
                b=self.backend(t);x=self.initial_params(b);z=copy.deepcopy(x)
                z['current_scale']=float(np.nextafter(x['current_scale'],math.inf))
                self.assertNotEqual(iv.descriptor_raw_key(b,x),iv.descriptor_raw_key(b,z))
                self.assertEqual(iv.descriptor_deck(b,x),iv.descriptor_deck(b,z))
                self.assertEqual(iv.descriptor_render_key(b,x),iv.descriptor_render_key(b,z))
                z['current_scale']=x['current_scale']*1.000001
                self.assertNotEqual(iv.descriptor_render_key(b,x),iv.descriptor_render_key(b,z))
                if t=='T3':
                    z=copy.deepcopy(x);z['rin']=float(np.nextafter(x['rin'],math.inf))
                    self.assertNotEqual(iv.descriptor_render_key(b,x),iv.descriptor_render_key(b,z))

    def test_T1_renderer_byte_exact_actual_native_wide_deck(self):
            from rc_agent import physical
            from filteragent_provider_stream_v1.provider import injected
            b=self.backend('T1');x=self.initial_params(b);captured=[]
            class Captured(Exception):pass
            def capture(deck,*a):
                captured.append(deck.replace('ac dec 40 1 1e9','ac dec 40 0.001 1e9'))
                raise Captured()
            with self.assertRaises(Captured):
                injected(physical.characterize,run_deck=capture)(b.devices(x),self.folder,None)
            self.assertEqual(captured,[iv.descriptor_deck(b,x)])

    def test_rendered_cache_charges_real_mock_ledger_and_fresh_bypasses_all_types(self):
            import numpy as np
            from types import MethodType
            from rc_agent.models import OTA
            for arm in CONFIGS:
                for t in ('T1','T2','T3'):
                    b=self.backend(t,arm,tag='deckcache');x=self.initial_params(b)
                    seen=[]
                    b.engine.executor=lambda *a,**k:SimpleNamespace(returncode=0,stdout='synthetic_fixture',stderr='')
                    def measured(self,p,*,fresh=False):
                        seen.append(copy.deepcopy(p))
                        c,_=self.engine.candidate({'synthetic_OTA':p,'fresh_repeat':len(self.auxiliary) if fresh else None})
                        out=self.engine.run_spice(c,'ota_descriptor',iv.descriptor_deck(self,p))
                        self.engine.finish_candidate(c,measured=True)
                        eta=OTA(1000.,1e7,100.);e={'evidence':out['directory'],'power_w':1e-6}
                        self.auxiliary.append({'candidate_id':c['id'],'status':'measured'})
                        self.descriptors[iv.descriptor_raw_key(self,p)]=(eta,e)
                        if t!='T1':self.calibration_models[iv.descriptor_raw_key(self,p)]=object()
                        return eta,e
                    b.characterize=MethodType(measured,b);iv.install_rendered_descriptor_cache(b)
                    first=b.characterize(x);z=copy.deepcopy(x)
                    z['current_scale']=float(np.nextafter(x['current_scale'],math.inf))
                    self.assertEqual(b.characterize(z),first)
                    self.assertEqual(len(seen),1);self.assertEqual(len(b.engine.attempts),1)
                    b.characterize(z,fresh=True)
                    self.assertEqual(len(seen),2);self.assertEqual(len(b.engine.attempts),2)
                    ledger=read(b.out/'budget.json')
                    self.assertEqual(ledger['actual_started_spice_processes'],2)
                    self.assertEqual(ledger['new_unique_candidate_attempts'],2)
                    self.assertEqual(len(b.auxiliary),2)

    def test_same_margin_zero_update_zero_and_improvement_sign(self):
            from .action_preview import compare_allocation
            for t in ('T1','T2','T3'):
                actual=[.082,-.5,-1.];margin=[.02,.02,0.]
                zero={'actual_constraints':actual,'linearized_hard_constraints':actual,
                    'predicted_constraints_with_margin':[a+m for a,m in zip(actual,margin)]}
                self.assertEqual(compare_allocation(zero)['predicted_reduction'],0.)
                self.assertEqual(compare_allocation(zero)['predicted_hard_reduction'],0.)
                for proposed,sign in ((.078,1),(.086,-1)):
                    p={**zero,'linearized_hard_constraints':[proposed,-.5,-1.],
                        'predicted_constraints_with_margin':[proposed+.02,-.48,-1.]}
                    value=compare_allocation(p)
                    self.assertGreater(sign*value['predicted_reduction'],0.)
                    self.assertGreater(sign*value['predicted_hard_reduction'],0.)
                # Mixed-margin r5 arithmetic would falsely report negative gain.
                p={**zero,'linearized_hard_constraints':[.078,-.5,-1.],
                    'predicted_constraints_with_margin':[.098,-.48,-1.]}
                self.assertLess(actual[0]**2-.098**2,0.)
                self.assertGreater(compare_allocation(p)['predicted_reduction'],0.)

    def test_preview_and_actual_Powell_seed_match_without_new_bench(self):
            from .action_preview import paid_power_seed
            from filteragent.backend import T1Backend
            from filteragent_type_v3.backend import TypeBackend
            from rc_agent.models import OTA
            from types import MethodType
            import numpy as np
            for arm in CONFIGS:
                for t in ('T1','T2','T3'):
                    b=self.backend(t,arm,tag='preview');self.numerical_fixture(b)
                    b.current=self.synthetic_row(b,self.initial_params(b),'circuit_fail')
                    raw=iv.descriptor_raw_key(b,b.current['parameters'])
                    b.descriptors[raw]=(OTA(1000.,1e7,100.),{'power_w':2e-6,'evidence':'synthetic_paid'})
                    plan={'passives':{},'ota_target':dict(a0=1000.,wp_rad_s=1e7,ro_ohm=100.),
                        'power_target_w':1e-6,'saturation_target_v':.1}
                    before=len(b.engine.attempts)
                    preview=paid_power_seed(b,plan,raw)
                    self.assertEqual(len(b.engine.attempts),before)
                    for field in ('predicted_full_filter_power_w','predicted_full_filter_phase_margin_deg',
                        'predicted_full_filter_response','predicted_complete_circuit_PASS','achieved_seed_eta'):
                        self.assertIsNone(preview[field])
                    cls=T1Backend if t=='T1' else TypeBackend
                    b.size_devices=MethodType(cls.size_devices,b);seen=[]
                    def driver(objective,z,*a,**k):
                        seen.append(list(map(float,z)));return objective(z)
                    module='filteragent.backend' if t=='T1' else 'filteragent_type_v3.backend'
                    with patch(module+'.native_powell',side_effect=driver):
                        iv.install_sizing_stop(b);b.size_devices(plan)
                    self.assertEqual(seen,[preview['seed_log_parameters']])
                    event=next(e['data'] for e in b.events if e['kind']=='generic_paid_power_ratio_device_seed')
                    self.assertEqual(event,preview)
                    self.assertEqual(len(b.engine.attempts),before)

    def test_power_seed_preview_missing_or_invalid_data_never_invents_reachability(self):
            from .action_preview import paid_power_seed
            from rc_agent.models import OTA
            b=self.backend('T3');self.numerical_fixture(b)
            b.current=self.synthetic_row(b,self.initial_params(b),'circuit_fail')
            raw=iv.descriptor_raw_key(b,b.current['parameters'])
            for value in (None,True,0.,-1.,float('nan')):
                b.descriptors[raw]=(OTA(1000.,1e7,100.),{'power_w':value,'evidence':'synthetic'})
                p=paid_power_seed(b,{'power_target_w':1e-6},raw)
                self.assertFalse(p['power_ratio_rule_available'])
                self.assertIsNone(p['approximate_seed_OTA_power_w'])
                self.assertFalse(p['power_ratio_rule_applied'])
                self.assertEqual(p['seed_parameters'],b.current['parameters'])

    def test_post_complete_PASS_no_provider_snapshot_or_OTA_fresh_still_executes(self):
            from types import MethodType
            for arm in CONFIGS:
                for t in ('T1','T2','T3'):
                    for endpoint_status in ('success','circuit_fail'):
                        b=self.backend(t,arm,tag='terminal'+endpoint_status)
                        self.numerical_fixture(b)
                        seen={'pass':False,'OTA_before_PASS':0,'provider_calls':0,'evaluations':[]}
                        b.engine.executor=lambda *a,**k:SimpleNamespace(returncode=0,stdout='synthetic_mock_only',stderr='')
                        native_char=b.characterize
                        def characterize(self,*a,**k):
                            if seen['pass']:deny('post_PASS_OTA_forbidden')
                            seen['OTA_before_PASS']+=1
                            return native_char(*a,**k)
                        b.characterize=MethodType(characterize,b)
                        b.diagnose=MethodType(lambda self:{'model_trusted':False,'matched_operating_point':True},b)
                        def evaluate(self,params,endpoint=False):
                            initial=not seen['evaluations']
                            status=endpoint_status if endpoint else 'circuit_fail' if initial else 'success'
                            row=selftest.synthetic_row(self,params,status)
                            c,_=self.engine.candidate(dict(params),endpoint=endpoint)
                            # Actual common ledger path with an in-memory executor,
                            # never a physical SPICE subprocess. Both full slots and
                            # fresh reservation remain enforced by the native engine.
                            for name in ('performance','matrix'):
                                analyses=6 if name=='matrix' and self.case['type_id']!='T1' else 5
                                self.engine.run_spice(c,name,'synthetic_'+name+'\n'+'op\n'*analyses)
                            self.engine.finish_candidate(c,measured=True,accepted=status=='success',
                                endpoint_evidence='synthetic_mock_only' if endpoint else None)
                            row.update(candidate_id=c['id'],endpoint_check=endpoint)
                            row['metrics']['power_w']=self.case['power_max_uW']*1e-6*(1.2 if initial else .95)
                            seen['evaluations'].append(endpoint)
                            self.rows.append(row)
                            if endpoint:
                                self.endpoint=row;self.engine.endpoint=c['id']
                                self.engine.status='success' if status=='success' else 'circuit_fail'
                            else:
                                self.best=row
                                if status=='success':self.first=row;seen['pass']=True
                            return row
                        selftest=self;b.evaluate=MethodType(evaluate,b)
                        def snapshot(self,step):
                            if seen['pass']:deny('post_PASS_snapshot_forbidden')
                            state={'step':step,'case':self.case,'observation':self.current,
                                'diagnostics':{'model_trusted':False},'allowed_actions':['finish','probe'],
                                'evidence_ids':['observation:'+str(self.current['candidate_id'])],
                                'knowledge_ids':['K01'],'last_tool_result':{},'tool_history':[],
                                'knowledge':{'cards':[{'id':'K01'}]}}
                            if t!='T1':state['parameter_names']=self.names
                            state['state_id']=digest(state);return state
                        b.snapshot=MethodType(snapshot,b)
                        client=FixtureClient([('probe',{'axis':'current_scale','direction':-1})],b.engine)
                        if True:
                            decide=client.decide
                            def checked_decide(*a,**k):
                                if seen['pass']:deny('post_PASS_provider_forbidden')
                                seen['provider_calls']+=1;return decide(*a,**k)
                            client.decide=checked_decide
                        result=full_loop(b,client)
                        self.assertEqual(seen['evaluations'],[False,False,True])
                        self.assertEqual(seen['provider_calls'],1)
                        self.assertGreater(seen['OTA_before_PASS'],0)
                        self.assertEqual(b.refinement_audit['terminal_PASS_diagnostic_deferrals'],1)
                        self.assertEqual(result['early_stop_path'],'current_full_measurement_pass')
                        self.assertTrue(result['independent_endpoint_checked'])
                        self.assertEqual(result['method_success'],endpoint_status=='success')
                        self.assertEqual(sum(c['role']=='endpoint' for c in b.engine.candidates),1)
                        self.assertEqual(len(b.engine.attempts),6)  # mock executor only
                        event=next(e['data'] for e in b.events if e['kind']=='complete_PASS_trial_diagnosis_deferred')
                        self.assertTrue(event['previous_costs_not_modified']);self.assertTrue(event['fresh_endpoint_required'])
                        self.assertIn('exactly finish',event['skipped_reason'])

    def test_nonPASS_or_incomplete_PASS_retains_trial_diagnostics(self):
            for t in ('T1','T2','T3'):
                b=self.backend(t,tag='nonPASSdiag');self.numerical_fixture(b);b.initialize()
                native_char=b.characterize;seen=[]
                from types import MethodType
                def characterized(self,*a,**k):
                    seen.append((a,k));return native_char(*a,**k)
                b.characterize=MethodType(characterized,b)
                for status,failed,validity in (('circuit_fail',['power'],'valid'),('success',['power'],'valid'),('success',[],'error')):
                    row=b.evaluate(self.initial_params(b))
                    row.update(status=status,failed_constraints=failed,measurement_status=validity)
                    count=len(seen)
                    b.accept_trial(row,planned_eta=b.plan_eta,planned_model=b.plan_calibration,force=True)
                    self.assertGreater(len(seen),count)
                    self.assertEqual(b.refinement_audit['terminal_PASS_diagnostic_deferrals'],0)

    def test_full_PASS_keeps_one_fresh_native_endpoint_all_types(self):
            for arm in CONFIGS:
                for t in ('T1', 'T2', 'T3'):
                    b = self.backend(t, arm)
                    calls = self.fixture_measurements(b)
                    b.characterize = deny
                    loop = full_loop
                    r = loop(b, NoProvider(), identity={'case_id': b.case['case_id']})
                    self.assertEqual([c[1] for c in calls], [False, True])
                    self.assertTrue(r['method_success'])
                    self.assertTrue(r['independent_endpoint_checked'])
                    self.assertEqual(r['decision_count'], 0)
                    self.assertEqual(b.engine.api_calls, [])
                    self.assertEqual(b.engine.attempts, [])

    def test_trace_attach_all_types_and_fresh_endpoint(self):
            from filteragent_bootstrap_optimization_v1.telemetry import Trace, attach
            for arm in CONFIGS:
                for t in ('T1', 'T2', 'T3'):
                    b = self.backend(t, arm)
                    trace = Trace(b.out)
                    attach(b, trace)  # Real R8 reflection/instance-wrapper path.
                    calls = self.fixture_measurements(b)
                    b.characterize = deny
                    loop = full_loop
                    with trace.span('case.worker'):
                        result = loop(b, NoProvider())
                    trace.summarize({'queue_is_outside_native_case_wall': True})
                    self.assertTrue(result['method_success'])
                    self.assertTrue(result['independent_endpoint_checked'])
                    self.assertEqual([r[1] for r in calls], [False, True])
                    self.assertTrue(any(s['name'] == 'initialization' for s in trace.spans))
                    self.assertTrue(any(s['name'] == 'endpoint_and_finalization' for s in trace.spans))
                    self.assertFalse(trace.stack)
                    self.assertEqual(b.engine.api_calls, [])
                    self.assertEqual(b.engine.attempts, [])

    def initial_params(self, b):
            if b.case['type_id'] != 'T1':
                return copy.deepcopy(b.binding['initial'])
            from filteragent.backend import analytic_initial
            return analytic_initial(b.task, b.bounds)

    def numerical_fixture(self, b):
            import numpy as np
            from types import MethodType
            from rc_agent.models import OTA, sk_response
            from orchestration.filteragent_v3.m1_type_v3.model import network_response
            from filteragent import numerics as t1n
            from filteragent_type_v3 import numerics as tn
            typed = b.case['type_id'] != 'T1'
            class Model:
                def response(self, p, eta, f, load):
                    if not typed:
                        return sk_response(p, eta, f, load)
                    gain = eta.a0/(1+2j*np.pi*f/eta.wp_rad_s)
                    return network_response(b.case['type_id'], p, f, gain, np.full(len(f), eta.ro_ohm), load)
            model = Model(); eta = OTA(1000., 1e7, 100.)
            specrows = tn.rows(b.case,b.common) if typed else t1n.constraint_rows(b.task,b.common)
            grouped = {}
            for name, side, value, scale in specrows:
                grouped.setdefault(name, {})[side] = value
            metrics = {k: (sum(v.values())/len(v)) for k,v in grouped.items()}
            metrics.update(f0_hz=b.case['f0_hz']*1.06,
                q=b.case['q_target'] if typed else b.task['q'], dc_out=1.2, power_w=1e-6,
                stable=True, full_local_stability='stable', full_stability_status='pass_numerical_local',
                polarity_pass=True, step_status='pass', sampled_nyquist_turns=0.,
                minimum_saturation_margin_v=.2, phase_margin_deg=70.)
            sequence = [0]
            def evaluate(self, params, endpoint=False):
                row = Tests.synthetic_row(selftest, self, params, 'circuit_fail')
                sequence[0] += 1
                row.update(candidate_id=sequence[0], metrics=copy.deepcopy(metrics))
                self.rows.append(row); self.best = row
                return row
            selftest = self
            b.evaluate = MethodType(evaluate,b)
            def characterize(self, x, fresh=False):
                nonlocal model
                if fresh:
                    model = Model()
                if typed:
                    self.calibration_models[self.descriptor_key(x)] = model
                if self.descriptor_anchor is None:
                    self.descriptor_anchor = eta.values
                return eta, {'evidence':'synthetic_no_files', 'power_w':metrics['power_w'],
                    'minimum_saturation_margin_v':.2, 'dc_out':1.2}
            b.characterize = MethodType(characterize,b)
            b.repeat_full = MethodType(lambda self,x:self.evaluate(x),b)
            def wave(self,row):
                f=np.geomspace(.001,1e9,481)
                if typed:
                    from orchestration.filteragent_v3.type_runtime_v2.adapter import PASSIVE_NAMES
                    p={k:row['parameters'][k] for k in PASSIVE_NAMES[b.case['type_id']]}
                    load=b.case['load_cap_pF']*1e-12
                else:
                    from rc_agent.relations import Passives
                    p=Passives(*(row['parameters'][k] for k in t1n.PASSIVE_NAMES))
                    load=b.task['load_f']
                return f,model.response(p,eta,f,load)
            b.wave=MethodType(wave,b)
            if not typed:
                guard=patch('rc_agent.physical.calibration_from_evidence',side_effect=lambda *a:model)
                guard.start();self.addCleanup(guard.stop)
            return eta, model

    def numeric_equal(self, actual, expected):
            # Native Jacobians divide solver-fit floating differences by 1e-4;
            # repeated native least-squares fits may differ in the last 1e-9. Keep
            # structure/flags exact and numeric agreement far below .02 margins.
            import numbers
            if isinstance(expected,bool) or expected is None or isinstance(expected,str):
                self.assertEqual(actual,expected)
            elif isinstance(expected,numbers.Real):
                self.assertLessEqual(abs(actual-expected),1e-8+1e-9*abs(expected))
            elif isinstance(expected,dict):
                self.assertEqual(set(actual),set(expected))
                for k in expected:self.numeric_equal(actual[k],expected[k])
            elif isinstance(expected,(list,tuple)):
                self.assertEqual(len(actual),len(expected))
                for a,e in zip(actual,expected):self.numeric_equal(a,e)
            else:self.assertEqual(actual,expected)
