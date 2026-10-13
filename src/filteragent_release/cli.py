"""Explicit single-case and 45-case Full runs with fresh outputs."""
import argparse,json,os,random,shutil
from pathlib import Path

def ordered_cases(rows):
    groups=[];rng=random.Random(0)
    for typ in ('T1','T2','T3'):
        for level in ('L1','L2','L3'):
            group=[r for r in rows if r['type_id']==typ and r['level']==level]
            rng.shuffle(group);groups.append(group)
    ordered=[]
    while any(groups):
        for group in groups:
            if group:ordered.append(group.pop(0))
    return ordered

def load_cases():
    from .paths import BENCHMARK_DIR
    from .integrity import verify
    from filteragent_data.runtime_adapter import load_case,binding_report
    verify()
    rows=json.loads((BENCHMARK_DIR/'runtime_index.json').read_text())['cases']
    if len(rows)!=45 or len({r['case_id'] for r in rows})!=45:raise ValueError('Expected exactly 45 unique cases')
    cases=[]
    for row in rows:
        case=load_case(row['case_id']);binding_report(case);cases.append(case)
    return cases

def construct(case,out,ngspice,executor=None):
    from .paths import CODE_ROOT
    from filteragent.backend import T1Backend
    from filteragent_type_v3.backend import TypeBackend
    from filteragent_r7a.core import install
    cls=TypeBackend if case['type_id']!='T1' else T1Backend
    protocol=json.loads((CODE_ROOT/'src/filteragent_bootstrap_optimization_v1/protocols'/('T1.json' if case['type_id']=='T1' else 'Type.json')).read_text())
    kwargs={'launch_authorized':True}
    if cls is TypeBackend:kwargs['executor']=executor
    backend=cls(case,out,protocol,ngspice,**kwargs)
    if cls is T1Backend and executor is not None:backend.engine.executor=executor
    if backend.engine.cache or backend.engine.candidates or backend.engine.api_calls or backend.engine.endpoint is not None:
        raise ValueError('Fresh cold instance required')
    install(backend)
    return backend

def execute(case,out,ngspice):
    from .integrity import verify
    from filteragent_r7a.loop import run_loop
    from filteragent_bootstrap_optimization_v1.telemetry import Trace,attach
    from filteragent_bootstrap_optimization_v1.client import Client
    from rc_agent.qwen_adapter import QwenConnectivityAdapter
    from filteragent_fallback_v2.evidence import save
    source=verify();trace=Trace(out,enabled=True)
    try:
        with trace.span('case.worker'):
            with trace.span('backend.construction'):backend=construct(case,out,ngspice)
            attach(backend,trace)
            client=Client(backend.engine,trace=trace,adapter_factory=QwenConnectivityAdapter.from_existing_config,
                runtime_authority={'verified_release':True,'execution_authorized':True,'case_id':case['case_id'],'source_sha256':source})
            with trace.span('method.loop'):result=run_loop(backend,client,identity={'run_id':out.name,'case_id':case['case_id']})
            if len(backend.engine.api_calls)>32 or client.actual_retries>1:raise ValueError('API/retry budget exceeded')
            result.update(release='filteragent-r7a-full',source_sha256=source,numerical_local_seed=0,
                historical_answer_injected=False,automatic_case_restart=False,
                new_API_calls=len(backend.engine.api_calls),actual_transport_retries=client.actual_retries,
                adaptive_target_feedback=backend.adaptive_target_feedback,refinement_audit=backend.refinement_audit,
                scheduler_action_memory=backend.scheduler_memory,new_instance_endpoint_count=int(backend.engine.endpoint is not None))
            save(out/'result.json',result)
        trace.summarize({'queue_wall_seconds':0,'execution':'sequential local fresh case'})
        return result
    except BaseException as exc:
        out.mkdir(parents=True,exist_ok=True)
        save(out/'entry_failure.json',{'method_success':False,'error_class':type(exc).__name__,'automatic_restart':False})
        raise

def main():
    parser=argparse.ArgumentParser(description='Full FilterAgent r7a: validate or run the independent 45-case benchmark')
    group=parser.add_mutually_exclusive_group(required=True);group.add_argument('--case');group.add_argument('--all',action='store_true')
    parser.add_argument('--benchmark',type=Path,help='Benchmark package root; default: sibling FilterAgent-benchmark')
    mode=parser.add_mutually_exclusive_group();mode.add_argument('--check-only',action='store_true');mode.add_argument('--execute',action='store_true',help='Explicitly enable ngspice and paid model requests when needed')
    parser.add_argument('--out',type=Path,default=Path('output/full45'),help='Fresh output directory within the code package output/')
    parser.add_argument('--ngspice',default='ngspice',help='ngspice executable with KLU and mdump support')
    args=parser.parse_args()
    if args.benchmark:os.environ['FILTERAGENT_BENCHMARK']=str(args.benchmark.expanduser().resolve())
    from .paths import CODE_ROOT,PDK
    rows=load_cases();selected=ordered_cases(rows) if args.all else [c for c in rows if c['case_id']==args.case]
    if not selected:parser.error('Unknown case ID')
    if not args.execute:
        print(json.dumps({'status':'PASS_readonly','cases':len(selected),'unique_case_ids':[c['case_id'] for c in selected],
            'API_started':0,'SPICE_started':0,'PDK_installed':PDK.is_file(),'ngspice_found':shutil.which(args.ngspice) is not None}));return 0
    if not PDK.is_file():parser.error('Set FILTERAGENT_PDK_LIB to an installed open_pdks SKY130 sky130.lib.spice')
    if shutil.which(args.ngspice) is None:parser.error('ngspice executable not found')
    out=args.out.resolve() if args.out.is_absolute() else (CODE_ROOT/args.out).resolve()
    if not out.is_relative_to(CODE_ROOT/'output'):parser.error('--out must be within code package output/')
    if out.exists():parser.error('Output already exists; use a new directory (no automatic resume/overwrite)')
    out.mkdir(parents=True,exist_ok=False);records=[]
    for case in selected:
        folder=out/case['case_id']
        try:
            result=execute(case,folder,args.ngspice)
            record={'case_id':case['case_id'],'method_success':result.get('method_success',False),'cost':result.get('cost',{}),'token_accounting':result.get('token_accounting',{})}
        except Exception as exc:record={'case_id':case['case_id'],'method_success':False,'error_class':type(exc).__name__}
        records.append(record);(out/'summary.json').write_text(json.dumps(records,indent=2,allow_nan=False)+'\n')
        print(json.dumps(record),flush=True)
    return 0 if all(r['method_success'] for r in records) else 1
