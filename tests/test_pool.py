"""Replica pool: one replica per device, one shared queue."""

import threading
import time

import pytest

from server import app as app_module
from server.app import Manager
from server.backends import BackendError, StubBackend
from server.registry import REGISTRY

A, B = REGISTRY["eurollm-1.7b-it"], REGISTRY["nllb-200-600m"]


class Fake(StubBackend):
    delay: dict[str, float] = {}   # device -> compile seconds
    broken: set[str] = set()       # devices whose compile fails

    def __init__(self, entry, device, *a, **k):
        super().__init__(entry)
        self.device = self.active_device = device

    def load(self):
        time.sleep(self.delay.get(self.device, 0))
        if self.device in self.broken:
            raise BackendError(f"{self.device} compile failed")
        super().load()


@pytest.fixture
def mgr(monkeypatch):
    Fake.delay, Fake.broken = {}, set()
    monkeypatch.setattr(app_module, "make_backend", lambda entry, device, *a, **k: Fake(entry, device))
    return lambda devices: Manager(["eurollm-1.7b-it", "nllb-200-600m"], devices)


def test_concurrent_requests_land_on_different_replicas(mgr):
    m = mgr(["NPU", "GPU"])
    first, second = m.acquire(A), m.acquire(A)
    assert {first.device, second.device} == {"NPU", "GPU"} and m.busy == 2
    got = []
    t = threading.Thread(target=lambda: got.append(m.acquire(A)), daemon=True)
    t.start()
    time.sleep(0.1)
    assert not got  # third request queues: both replicas busy
    m.release(second)
    t.join(2)
    assert got and got[0].device == second.device and got[0].queued >= 0.09  # the wait is measured
    assert first.queued < 0.5


def test_idle_replica_preference_follows_device_order(mgr):
    m = mgr(["GPU", "NPU", "CPU"])
    m.preload(A)
    be = m.acquire(A)
    assert be.device == "GPU"


def test_first_ready_replica_serves_while_the_slow_one_compiles(mgr):
    m = mgr(["NPU", "GPU"])
    Fake.delay = {"NPU": 0.6}
    t0 = time.time()
    be = m.acquire(A)
    assert be.device == "GPU" and time.time() - t0 < 0.4
    m.release(be)
    m.preload(A)
    assert m.alive == 2


def test_one_failed_replica_does_not_stop_the_others(mgr):
    Fake.broken = {"NPU"}
    m = mgr(["NPU", "GPU"])
    m.preload(A)
    assert m.alive == 1 and m.acquire(A).device == "GPU"


def test_all_failed_raises_then_retries_next_time(mgr):
    Fake.broken = {"NPU", "GPU"}
    m = mgr(["NPU", "GPU"])
    with pytest.raises(BackendError, match="compile failed"):
        m.acquire(A)
    Fake.broken = set()  # e.g. the model was converted in the meantime
    assert m.acquire(A).device == "NPU"


def test_switching_models_waits_until_all_replicas_are_idle(mgr):
    m = mgr(["NPU", "GPU"])
    held = m.acquire(A)
    got = []
    t = threading.Thread(target=lambda: got.append(m.acquire(B)), daemon=True)
    t.start()
    time.sleep(0.15)
    assert not got and m.current is A  # still draining
    assert m.want is B
    m.release(held)
    t.join(2)
    assert got and got[0].entry is B and m.current is B


def test_requests_for_the_current_model_do_not_starve_a_waiting_switch(mgr):
    m = mgr(["NPU", "GPU"])
    held = m.acquire(A)                      # one replica busy, one free
    got_b = []
    tb = threading.Thread(target=lambda: got_b.append(m.acquire(B)), daemon=True)
    tb.start()
    time.sleep(0.1)
    late_a = []
    ta = threading.Thread(target=lambda: late_a.append(m.acquire(A)), daemon=True)
    ta.start()
    time.sleep(0.15)
    assert not late_a                        # the free replica is held back for the switch
    m.release(held)
    tb.join(2)
    assert got_b and got_b[0].entry is B
    m.release(got_b[0])
    ta.join(3)
    assert late_a and late_a[0].entry is A   # and the A request is served after the switch back


# ---- model@DEVICE options ----------------------------------------------------------------------------------

def test_device_option_loads_and_uses_only_that_device(mgr):
    loaded = []
    orig = Fake.load
    Fake.load = lambda self: (loaded.append(self.device), orig(self))[1]
    try:
        m = mgr(["NPU", "GPU"])
        be = m.acquire(A, ["GPU"])
        assert be.device == "GPU" and m.state == {"NPU": "new", "GPU": "busy"}  # NPU not compiled: nobody asked for it
        m.release(be)
        assert loaded == ["GPU"]
    finally:
        Fake.load = orig


def test_device_option_waits_for_its_device_even_if_another_is_free(mgr):
    m = mgr(["NPU", "GPU"])
    gpu = m.acquire(A, ["GPU"])
    got = []
    t = threading.Thread(target=lambda: got.append(m.acquire(A, ["GPU"])), daemon=True)
    t.start()
    time.sleep(0.15)
    assert not got                       # NPU could serve, but this request asked for GPU
    assert m.acquire(A, ["NPU"]).device == "NPU"
    m.release(gpu)
    t.join(2)
    assert got and got[0].device == "GPU"


def test_plain_name_uses_every_device(mgr):
    m = mgr(["NPU", "GPU"])
    first = m.acquire(A)
    second = m.acquire(A)
    assert {first.device, second.device} == {"NPU", "GPU"}


def test_names_listed_per_device_only_when_several_are_served():
    from fastapi.testclient import TestClient

    from server.app import create_app

    one = TestClient(create_app(stub=True), base_url="http://localhost")
    assert [m["name"] for m in one.get("/api/tags").json()["models"]] == ["translategemma:4b", "nllb-200-600m", "eurollm-1.7b-it"]
    many = TestClient(create_app(stub=True, device=["NPU", "GPU"]), base_url="http://localhost")
    names = [m["name"] for m in many.get("/api/tags").json()["models"]]
    assert "translategemma:4b" in names and "translategemma:4b@NPU" in names and "translategemma:4b@GPU" in names
    assert len(names) == 9
    ids = {d["id"]: d.get("device") for d in many.get("/v1/models").json()["data"]}
    assert ids["eurollm-1.7b-it@GPU"] == "GPU" and ids["eurollm-1.7b-it"] is None


def test_variant_requests_route_and_ps_reports_the_loaded_variants():
    from fastapi.testclient import TestClient

    from server.app import create_app

    c = TestClient(create_app(stub=True, device=["NPU", "GPU"]), base_url="http://localhost")
    r = c.post("/api/chat", json={"model": "eurollm-1.7b-it@GPU", "messages": [{"role": "user", "content": "Hei"}]})
    assert r.status_code == 200
    assert set(m["name"] for m in c.get("/api/ps").json()["models"]) == {"eurollm-1.7b-it", "eurollm-1.7b-it@GPU"}
    bad = c.post("/api/chat", json={"model": "eurollm-1.7b-it@CPU", "messages": [{"role": "user", "content": "Hei"}]})
    assert bad.status_code == 404 and "CPU" in bad.json()["error"]
    assert c.post("/api/chat", json={"model": "nope@GPU", "messages": [{"role": "user", "content": "x"}]}).status_code == 404


def test_idle_model_is_unloaded_and_reloaded_on_demand(monkeypatch):
    Fake.delay, Fake.broken = {}, set()
    monkeypatch.setattr(app_module, "make_backend", lambda entry, device, *a, **k: Fake(entry, device))
    m = Manager(["eurollm-1.7b-it"], ["NPU", "GPU"], idle_unload=0.3)
    held = m.acquire(A)
    time.sleep(0.6)
    assert m.current is A and m.loaded_names()[0] == "eurollm-1.7b-it"  # a busy replica keeps the model loaded
    m.release(held)
    deadline = time.time() + 3
    while m.current is not None and time.time() < deadline:
        time.sleep(0.05)
    assert m.current is None and m.loaded_names() == [] and m.state == {} and m.backends == {}
    again = m.acquire(A)  # the next request transparently reloads
    assert again.entry is A and m.loaded_names()[0] == "eurollm-1.7b-it"


def test_plain_name_runs_on_the_default_device(monkeypatch):
    Fake.delay, Fake.broken = {}, set()
    monkeypatch.setattr(app_module, "make_backend", lambda entry, device, *a, **k: Fake(entry, device))
    m = Manager(["eurollm-1.7b-it"], ["NPU", "GPU", "CPU"], default_device="NPU")
    entry, devices = m.resolve("eurollm-1.7b-it")
    assert devices == ["NPU"]
    held = m.acquire(entry, devices)
    assert held.device == "NPU"
    got = []
    t = threading.Thread(target=lambda: got.append(m.acquire(entry, m.resolve("eurollm-1.7b-it")[1])), daemon=True)
    t.start()
    time.sleep(0.15)
    assert not got  # waits for the NPU even though GPU/CPU are idle
    assert m.resolve("eurollm-1.7b-it@GPU")[1] == ["GPU"]  # explicit devices still work
    m.release(held)
    t.join(2)
    assert got and got[0].device == "NPU"


def test_ps_reports_plain_name_only_when_its_default_device_is_loaded(monkeypatch):
    Fake.delay, Fake.broken = {}, set()
    monkeypatch.setattr(app_module, "make_backend", lambda entry, device, *a, **k: Fake(entry, device))
    m = Manager(["eurollm-1.7b-it"], ["NPU", "GPU"], default_device="NPU")
    m.preload(A, ["GPU"])
    assert m.loaded_names() == ["eurollm-1.7b-it@GPU"]  # the extension must still see "loading" for the plain name
    m.preload(A, ["NPU"])
    assert m.loaded_names() == ["eurollm-1.7b-it", "eurollm-1.7b-it@NPU", "eurollm-1.7b-it@GPU"]


def test_default_device_not_served_means_any():
    assert Manager(["eurollm-1.7b-it"], ["GPU"], default_device="NPU").default_device is None
