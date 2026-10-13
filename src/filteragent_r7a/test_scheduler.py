"""Synthetic target feedback and retained decision evidence; no physical/API I/O."""
import copy, math, socket, subprocess, unittest
from unittest.mock import patch
import numpy as np
from rc_agent.allocation import LogDomain, bounded_update
from .adaptive import feedback_weights, weighted_update
from .scheduler import compact

class Tests(unittest.TestCase):
    def setUp(self):
        def deny(*a, **k): raise AssertionError('offline_external_IO_forbidden')
        for target in ['socket.socket', 'socket.create_connection', 'subprocess.Popen', 'rc_agent.qwen_adapter.QwenConnectivityAdapter.from_existing_config']:
            guard = patch(target, side_effect=deny); guard.start(); self.addCleanup(guard.stop)
    def fixture(self):
        return dict(q=np.ones(4), domain=LogDomain((1,)*4,(.5,)*4,(2,)*4), c=np.array([.02,.015]),
            jacobian=np.array([[1.,0.,-1.,0.],[0.,1.,0.,-1.]]), trusted=True,
            scale=np.ones(4), delta=.04, margin=np.array([.02,.02]), regularization=.05, iterations=160)
    def test_without_feedback_exact_original(self):
        k=self.fixture();self.assertEqual(bounded_update(**k),weighted_update(**k,weights=[1.]*4))
    def test_penalty_moves_to_more_realizable_coordinates(self):
        k=self.fixture();base=bounded_update(**k);new=weighted_update(**k,weights=[1,1,4,4])
        self.assertLess(abs(new.delta_z[2]),abs(base.delta_z[2]))
        self.assertLess(abs(new.delta_z[3]),abs(base.delta_z[3]))
        self.assertGreater(abs(new.delta_z[0]),abs(base.delta_z[0]))
        self.assertTrue(all(abs(v)<=.04+1e-12 for v in new.delta_z))
        self.assertTrue(all(.5<=v<=2 for v in new.values))
    def test_domains_and_radius_on_large_violation(self):
        k=self.fixture();k['c']=np.array([50.,50.]);k['q']=np.array([1.99,1.99,.51,.51])
        new=weighted_update(**k,weights=[1,1,4,4])
        self.assertTrue(all(.5<=v<=2 for v in new.values));self.assertTrue(all(abs(v)<=.04+1e-12 for v in new.delta_z))
    def test_invalid_weights_fail_closed(self):
        for w in ([1,1,1],[1,1,1,0],[1,1,1,5],[1,1,1,float('nan')]):
            with self.assertRaises(ValueError):weighted_update(**self.fixture(),weights=w)
    def test_untrusted_nonfinite_inputs_rejected(self):
        for name,value in [('trusted',False),('delta',float('nan')),('c',[float('nan'),1])]:
            k=self.fixture();k[name]=value
            with self.assertRaises(ValueError):weighted_update(**k,weights=[4]*4)
    def event(self):
        return {'target':dict(a0=100.,wp_rad_s=1e6,ro_ohm=1000.),
                'best':{'eta':dict(a0=10.,wp_rad_s=1e6,ro_ohm=1000.)},'target_realized_within_tolerances':False}
    def test_feedback_only_mismatched_ETA_capped(self):
        self.assertEqual(feedback_weights(self.event()),[4.,1.,1.])
    def test_success_resets_penalty(self):
        e=self.event();e['target_realized_within_tolerances']=True;self.assertEqual(feedback_weights(e),[1.]*3)
    def test_power_only_failure_never_invents_ETA_gap(self):
        e=self.event();e['best']['eta']=copy.deepcopy(e['target']);self.assertEqual(feedback_weights(e),[1.]*3)
    def test_missing_invalid_paid_feedback_neutral(self):
        for e in ({},{'best':None},self.event()):
            if e.get('best'):e['best']['eta']['a0']=float('nan')
            self.assertEqual(feedback_weights(e),[1.]*3)
    def state(self):
        return {'case':{'type_id':'T3','power_max_uW':100},'observation':{'measurement_status':'error',
            'failed_constraints':['power','measurement_invalid'],'parameters':{'rin':1},'status':'circuit_fail',
            'metrics':{'power_w':None,'step_status':'fail','stability_check':{'status':'fail','max_pole_real_rad_s':2,
                'poles_rad_s':[[2,0]],'matrix_details':[{'matrix':list(range(300))}]}}},
            'diagnostics':{'model_trusted':False,'model_error':.1,'realization_error':.2,'planning_error':.3},
            'allowed_actions':['finish','probe'],'knowledge':{'cards':[{'id':'safe','content':'all original text'}]},
            'evidence_ids':['obs:0'],'knowledge_ids':['safe'],'limits':{'API':32},'tool_history':[{'huge':list(range(500))}]}
    def test_projection_preserves_all_gates_and_knowledge(self):
        s=self.state();before=copy.deepcopy(s);c=compact(s)
        self.assertEqual(s,before)
        for k in ['case','allowed_actions','knowledge','diagnostics','limits','evidence_ids','knowledge_ids']:self.assertEqual(c[k],s[k])
        self.assertEqual(c['observation']['failed_constraints'],s['observation']['failed_constraints'])
        self.assertEqual(c['observation']['metrics']['step_status'],'fail')
        self.assertEqual(c['observation']['metrics']['stability_check']['status'],'fail')
        self.assertEqual(c['observation']['metrics']['stability_check']['max_pole_real_rad_s'],2)
    def test_invalid_measurement_never_becomes_PASS(self):
        c=compact(self.state());self.assertEqual(c['observation']['measurement_status'],'error');self.assertEqual(c['observation']['status'],'circuit_fail')

if __name__=='__main__':unittest.main(verbosity=2)
