import unittest
import numpy as np
from rc_agent.native_powell import native_powell
class NativePowellTests(unittest.TestCase):
 def test_native_complete_iterations_reachable_for_all_seven_coordinates(self):
  n=7;state={};observed=[];history=[]
  settings={'log_coordinate_probe_step':.08,'max_native_function_calls':5000,'max_native_iterations':50,'xtol':.02,'ftol':.01}
  def loss(z):observed.append(np.array(z));return float(np.sum((z-np.linspace(-.3,.3,n))**2))
  r=native_powell(loss,np.zeros(n),np.array([[-1.,1.]]*n),settings,state,lambda s,z:history.append(s))
  self.assertTrue(r.success);self.assertGreaterEqual(r.nit,1);self.assertGreaterEqual(state['completed_iterations'],1)
  self.assertLess(r.fun,1e-8);self.assertTrue(all(any(abs(z[k])>1e-8 for z in observed) for k in range(n)))
  self.assertEqual(state['coordinate_design_calls'],8)
