from filteragent_release.paths import CODE_ROOT
"""Self-authored bounded benches for a reviewed MIT SKY130 OTA topology.

Topology source: nilaymastaadmi/analog-pmic-sky130, commit 66a09ae0039bb3f2965a46426b9e657cf9294f5e.
See circuits/LICENSE.nilay. Adaptations: explicit parameterization, micrometre W/L
for installed open_pdks scale=1u, own benches and no upstream scripts executed.
"""
from dataclasses import dataclass,asdict
from pathlib import Path
import json,subprocess,time,re,hashlib,math
import numpy as np
from scipy.optimize import least_squares
from .relations import Passives
from .models import OTA,OTACalibration
from .stability import port_commands,two_port_return
from .linear_stability import matrix_deck,analyze_dut

ROOT=CODE_ROOT
from filteragent_release.paths import PDK

@dataclass(frozen=True)
class Devices:
    current_scale: float=1.
    output_scale: float=1.
    cc_pf: float=2.
    rz_ohm: float=4604.

    def __post_init__(self):
        if not all(math.isfinite(v) and v>0 for v in asdict(self).values()):raise ValueError('Positive finite device parameters required')

def ota_netlist(d):
    s=d.current_scale;o=d.output_scale
    return f'''
* Adapted from MIT Nilay Toshniwal OTA; dimensions below are micrometres.
.subckt ota vinp vinn vout vdd vss vbias
XM1 s1o vinp tail vss sky130_fd_pr__nfet_01v8 W={3.469*s:.12g} L=1
XM2 s1x vinn tail vss sky130_fd_pr__nfet_01v8 W={3.469*s:.12g} L=1
XM3 s1x s1x vdd vdd sky130_fd_pr__pfet_01v8 W={25.676*s:.12g} L=1
XM4 s1o s1x vdd vdd sky130_fd_pr__pfet_01v8 W={25.676*s:.12g} L=1
XM5 tail vbias vss vss sky130_fd_pr__nfet_01v8 W={4.117*s:.12g} L=1
XM6 vout s1o vdd vdd sky130_fd_pr__pfet_01v8 W={46.478*s*o:.12g} L=1 m=3
XM7 vout vbias vss vss sky130_fd_pr__nfet_01v8 W={11.178*s*o:.12g} L=1
Rz s1o z {d.rz_ohm:.12g}
Cc z vout {d.cc_pf*1e-12:.12g}
.ends ota
.subckt bias vdd vss vbias
Iref vdd vbias {10e-6*s:.12g}
XM8 vbias vbias vss vss sky130_fd_pr__nfet_01v8 W={2.058*s:.12g} L=1
.ends bias
'''

def header(d):
    assert PDK.is_file()
    return f'* Self-authored real OTA development bench\n.lib "{PDK}" tt\n.options reltol=1e-5 abstol=1e-13 vntol=1e-8\n'+ota_netlist(d)

def op_commands():
    lines=['op','let dc_out=v(out)','let power_w=-i(Vdd)*1.8','print dc_out power_w']
    for i in range(1,8):
        kind='nfet' if i in (1,2,5,7) else 'pfet'
        dev=f'@m.xu.xm{i}.msky130_fd_pr__{kind}_01v8'
        lines.extend([f'let margin{i}=abs({dev}[vds])-abs({dev}[vdsat])',f'print margin{i}'])
    return '\n'.join(lines)

def run_deck(deck,directory,budget,tag):
    directory=Path(directory).resolve()
    if not directory.is_relative_to(ROOT) or directory.is_relative_to(ROOT/'backup'):raise ValueError('Run output outside writable new project')
    directory.mkdir(parents=True,exist_ok=False)
    (directory/'circuit.cir').write_text(deck)
    commands=re.findall(r'(?im)^[ \t]*(op|ac|tran|pz|tf)\b','\n'.join(deck.splitlines()[1:]))  # SPICE first line is always a title
    row,timeout=budget.reserve(tag,analysis_count=len(commands));start=time.monotonic()
    try:
        p=subprocess.run(['ngspice','-n','-b','circuit.cir'],cwd=directory,capture_output=True,text=True,timeout=timeout)
        log=p.stdout+'\n'+p.stderr;(directory/'ngspice.log').write_text(log)
        bad=p.returncode!=0 or re.search(r'(?im)^\s*(error|fatal)\b',log) is not None
        budget.finish(row,status='failed' if bad else 'completed',returncode=p.returncode,wall_seconds=time.monotonic()-start,evidence=str(directory.relative_to(ROOT)),netlist_sha256=hashlib.sha256(deck.encode()).hexdigest())
        if bad:raise RuntimeError('SPICE failed; inspect '+str(directory/'ngspice.log'))
        return log
    except subprocess.TimeoutExpired as e:
        def decoded(v):return v.decode(errors='replace') if isinstance(v,bytes) else (v or '')
        (directory/'ngspice.partial.log').write_text(decoded(e.stdout)+'\n'+decoded(e.stderr))
        (directory/'timeout.txt').write_text('Invocation timed out; no metrics accepted')
        budget.finish(row,status='timeout',wall_seconds=time.monotonic()-start,evidence=str(directory.relative_to(ROOT)))
        raise RuntimeError('SPICE timeout') from e
    except OSError as e:
        budget.finish(row,status='launch_error',wall_seconds=time.monotonic()-start,evidence=str(directory.relative_to(ROOT)),error=str(e))
        raise RuntimeError('SPICE could not launch') from e

def scalar(log,name):
    m=re.search(r'(?m)^'+re.escape(name)+r'\s*=\s*([-+0-9.eE]+)\s*$',log)
    if not m:raise ValueError('Missing measured '+name)
    v=float(m.group(1))
    if not np.isfinite(v):raise ValueError('Nonfinite measured '+name)
    return v

def table(directory,name,columns):
    a=np.loadtxt(Path(directory)/name,skiprows=1)
    if a.ndim!=2 or a.shape[1]!=columns or not np.all(np.isfinite(a)) or np.any(np.diff(a[:,0])<=0):raise ValueError('Invalid raw waveform '+name)
    return a

def first_crossing(f,values,level):
    inds=np.flatnonzero((values[:-1]>=level)&(values[1:]<level))
    if len(inds)!=1:raise ValueError('Missing or ambiguous descending crossover')
    i=int(inds[0]);u=(level-values[i])/(values[i+1]-values[i])
    return float(np.exp(np.log(f[i])+u*np.log(f[i+1]/f[i])))

def loop_metrics(f,loop):
    """Full output-loop voltage injection, with explicit sampled Nyquist check.

    Requires the reviewed open amplifier/passive return path to have no RHP poles;
    this is a bounded small-signal engineering check, not a formal global proof.
    """
    gain=20*np.log10(abs(loop));phase=np.unwrap(np.angle(loop))
    ugf=first_crossing(f,gain,0.)
    pm=180+float(np.interp(np.log(ugf),np.log(f),phase))*180/np.pi
    winding=float((np.unwrap(np.angle(1+loop))[-1]-np.unwrap(np.angle(1+loop))[0])*2/(2*np.pi))
    return {'loop_ugf_hz':ugf,'phase_margin_deg':pm,'sampled_nyquist_turns':winding,'nyquist_min_distance':float(np.min(abs(1+loop)))}

def characterize(d,directory,budget):
    deck=header(d)+'''
Vdd vdd 0 1.8
Vin vp 0 DC 1.2 AC 1
XB vdd 0 vbias bias
XU vp vn out vdd 0 vbias ota
Lfb out vn 1e12
Cfb vn 0 1
Vddz vddz 0 1.8
Vinz vpz 0 DC 1.2 AC 0
XBz vddz 0 vbiasz bias
XUz vpz vnz outz vddz 0 vbiasz ota
Lfbz outz vnz 1e12
Cfbz vnz 0 1
Itest 0 outz DC 0 AC 1
.control
set wr_singlescale
set wr_vecnames
set numdgt=15
'''+op_commands()+'''
ac dec 40 1 1e9
let ar=real(v(out))
let ai=imag(v(out))
let zr=real(v(outz))
let zi=imag(v(outz))
wrdata descriptor_ac.dat ar ai zr zi
quit
.endc
.end
'''
    log=run_deck(deck,directory,budget,'ota_descriptor');a=table(directory,'descriptor_ac.dat',5)
    f=a[:,0];gain=a[:,1]+1j*a[:,2];z=a[:,3]+1j*a[:,4]
    a0=float(abs(gain[0]));fp=first_crossing(f,20*np.log10(abs(gain)),20*np.log10(a0)-3.01029995664)
    eta=OTA(a0,2*np.pi*fp,float(z[0].real))
    result={'devices':asdict(d),'eta':asdict(eta),'power_w':scalar(log,'power_w'),'dc_out':scalar(log,'dc_out'),'minimum_saturation_margin_v':min(scalar(log,'margin'+str(i)) for i in range(1,8)),'unloaded':True,'evidence':str(Path(directory).relative_to(ROOT))}
    (Path(directory)/'measurement.json').write_text(json.dumps(result,indent=2))
    return eta,result

def fit_lowpass(f,h,target_hz):
    keep=(f>=target_hz*.05)&(f<=target_hz*5)
    if np.count_nonzero(keep)<20:raise ValueError('Insufficient measured fitting band')
    ff=f[keep];hh=h[keep]
    def error(x):
        gain,f0,q=np.exp(x);s=1j*ff/f0
        r=(gain/(1+s/q+s*s)-hh)/np.maximum(abs(hh),.05)
        return np.r_[r.real,r.imag]
    fit=least_squares(error,np.log([max(abs(h[0]),1e-6),target_hz,.707]),bounds=(np.log([.01,target_hz/10,.1]),np.log([10,target_hz*10,10])),max_nfev=80)
    gain,f0,q=np.exp(fit.x)
    if not fit.success:raise ValueError('Lowpass fit did not converge')
    return {'fitted_gain':float(gain),'f0_hz':float(f0),'q':float(q),'fit_max_component_error':float(np.max(abs(error(fit.x))))}

def calibration_from_evidence(eta,evidence):
    a=table(ROOT/evidence['evidence'],'descriptor_ac.dat',5)
    return OTACalibration(eta,tuple(a[:,0]),tuple(a[:,1]+1j*a[:,2]),tuple(a[:,3]+1j*a[:,4]))

def measure_filter(p,d,target_hz,directory,budget,load_f=5e-12):
    if not math.isfinite(load_f) or load_f<=0:raise ValueError('Positive explicit load required')
    stop=12/target_hz;delay=1/target_hz
    deck=header(d)+f'''
Vdd vdd 0 1.8
Vin in 0 DC 1.2 AC 1 PULSE(1.2 1.201 {delay} {delay/100} {delay/100} {stop*2} {stop*4})
XB vdd 0 vbias bias
XU vp out ampout vdd 0 vbias ota
Vloop ampout out DC 0 AC 0
Iport 0 out DC 0 AC 0
R1 in a {p.r1:.16g}
R2 a vp {p.r2:.16g}
Cg vp 0 {p.cg:.16g}
Cf a out {p.cf:.16g}
CL out 0 {load_f:.16g}
.control
set wr_singlescale
set wr_vecnames
set numdgt=15
'''+op_commands()+f'''
ac dec 40 1 1e8
let hr=real(v(out))
let hi=imag(v(out))
wrdata filter_ac.dat hr hi
alter Vin acmag=0
alter Vloop acmag=1
ac dec 40 1 1e10
let lr=real(-v(ampout)/v(out))
let li=imag(-v(ampout)/v(out))
wrdata loop_ac.dat lr li
'''+port_commands()+f'''
alter Vloop acmag=0
alter Iport acmag=0
tran {stop/3000} {stop}
wrdata step.dat v(out)
quit
.endc
.end
'''
    log=run_deck(deck,directory,budget,'full_filter_op_ac_loop_tran')
    a=table(directory,'filter_ac.dat',3);f=a[:,0];h=a[:,1]+1j*a[:,2]
    step=table(directory,'step.dat',2)
    metrics=fit_lowpass(f,h,target_hz)
    loop=table(directory,'loop_ac.dat',3)
    legacy=loop_metrics(loop[:,0],loop[:,1]+1j*loop[:,2])
    va=table(directory,'port_voltage.dat',7);ia=table(directory,'port_current.dat',7)
    if not np.array_equal(va[:,0],ia[:,0]):raise ValueError('Port excitation grids mismatch')
    ret,condition=two_port_return(va[:,0],va[:,1:],ia[:,1:])
    metrics.update(loop_metrics(va[:,0],ret))
    metrics.update(dc_gain=float(abs(h[0])),dc_gain_method='measured 1 Hz AC proxy; fitted gain is separate',dc_gain_frequency_hz=float(f[0]),legacy_voltage_ratio_pm_deg=legacy['phase_margin_deg'],port_excitation_condition=condition)
    # Settling is measured relative to the actual DC bias and low-frequency gain;
    # finite DC offset is separately retained and separately checked at acceptance.
    expected_final=scalar(log,'dc_out')+.001*float(h[0].real)
    metrics.update(power_w=scalar(log,'power_w'),power_method='quiescent 1.8 V VDD supply including the ideal reference-current branch; no signal-source power',dc_out=scalar(log,'dc_out'),minimum_saturation_margin_v=min(scalar(log,'margin'+str(i)) for i in range(1,8)),step_final_error_v=float(abs(np.mean(step[-100:,1])-expected_final)),step_tail_peak_to_peak_v=float(np.ptp(step[-100:,1])),stability_method='Tian two-port output return ratio with full filter present + full-circuit small step; internal OTA/parasitic loops are not all cut')
    metrics['port_and_step_checks_passed']=metrics['phase_margin_deg']>0 and abs(metrics['sampled_nyquist_turns'])<.1 and metrics['step_final_error_v']<1e-5 and metrics['step_tail_peak_to_peak_v']<1e-5
    try:
        matrix_folder=Path(directory)/'closed_loop_matrix'
        matrix_log=run_deck(matrix_deck(deck),matrix_folder,budget,'closed_loop_full_matrix')
        stability_check=analyze_dut(matrix_folder,p,PDK,matrix_log)
        if stability_check['status']=='pass' and abs(stability_check['op_dc_out_v']-metrics['dc_out'])>1e-7:
            stability_check.update(status='unknown',reason='Measurement and matrix DC points disagree')
    except (ValueError,RuntimeError) as exc:
        stability_check={'status':'unknown','reason':str(exc),'formal_global_certificate':'not_claimed'}
    metrics['stability_check']=stability_check
    step_ok=metrics['step_final_error_v']<1e-5 and metrics['step_tail_peak_to_peak_v']<1e-5
    metrics['stable']=False if stability_check['status']=='fail' or not step_ok else (True if stability_check['status']=='pass' else None)
    metrics['full_stability_status']='pass_numerical_local' if metrics['stable'] is True else ('fail' if metrics['stable'] is False else 'unknown')
    metrics['stability_method']='Complete quasistatic G+sC modes from all MNA states, exact ideal-source constraint elimination, two Jacobian coefficients, scaled eigenpairs, full AC reconstruction and finite small-step recovery'
    metrics['measurement_version']='4_complete_mna_gc_numerical'
    result={'passives':asdict(p),'devices':asdict(d),'load_f':load_f,'metrics':metrics,'source':'full_transistor_spice','evidence':str(Path(directory).relative_to(ROOT))}
    (Path(directory)/'measurement.json').write_text(json.dumps(result,indent=2))
    return f,h,result
