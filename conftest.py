"""Offline suite containment; dummy PDK is for text rendering only."""
import socket,subprocess
import pytest

@pytest.fixture(autouse=True)
def offline_only(monkeypatch,tmp_path):
    from rc_agent import physical
    dummy=tmp_path/'offline_model_placeholder.spice'
    dummy.write_text('* Offline rendering placeholder, never simulated.\n')
    monkeypatch.setattr(physical,'PDK',dummy)
    def deny(*args,**kwargs):raise AssertionError('Offline tests forbid network and physical subprocesses')
    monkeypatch.setattr(socket,'socket',deny)
    monkeypatch.setattr(socket,'create_connection',deny)
    monkeypatch.setattr(subprocess,'Popen',deny)
