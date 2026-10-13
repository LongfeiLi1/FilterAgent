"""Independent A/Z at the registered circuit's DC point, without AC loading.

T2's follower DC matches the shared unity bench. T3's capacitors open at DC,
the OTA input gate draws no current under the frozen PDK switches, and Rlink
therefore has no DC drop. Rin/Rfb determine the exact DC feedback and output
load. Large inductors retain that DC circuit while isolating its AC loading.
No complete-filter response is used to identify this model.
"""
from orchestration.filteragent_v3.m1_type_v3 import characterization as common

def deck(devices, case, passives):
    text=common.deck(devices)
    if case["type_id"]=="T2":
        return text
    if case["type_id"]!="T3":
        raise ValueError("Only registered T2/T3 supported")
    rin,rfb=passives["rin"],passives["rfb"]
    for suffix in ("","z"):
        out,vn="out"+suffix,"vn"+suffix
        old=f"Lfb{suffix} {out} {vn} 1e12"
        # Lout carries the real DC output current; Cdc prevents feedback-network
        # AC feedthrough. Lfb/Cfb remove AC negative feedback independently.
        new=f"""Vdc{suffix} vidc{suffix} 0 DC 1.2 AC 0
Loutdc{suffix} {out} outdc{suffix} 1e12
Coutdc{suffix} outdc{suffix} 0 1
Rindc{suffix} vidc{suffix} vfbdc{suffix} {rin:.17g}
Rfbdc{suffix} outdc{suffix} vfbdc{suffix} {rfb:.17g}
Lfb{suffix} vfbdc{suffix} {vn} 1e12"""
        if text.count(old)!=1:raise ValueError("Shared characterization deck changed")
        text=text.replace(old,new)
    return text

def analyze(folder,devices,type_id):
    eta,calibration,evidence=common.analyze(folder,devices,type_id)
    evidence.update(dc_condition="registered Rin/Rfb feedback and DC output load; AC isolated"
                    if type_id=="T3" else "registered follower DC; identical shared T2 bench",
                    source="independent DC-matched unloaded AC two-OTA A/Z; no full-filter fit",
                    m1_difference=type_id=="T3")
    return eta,calibration,evidence

