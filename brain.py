"""Rex personality, mood and memory (ported from security_dog.py; same rex_memory.json format)."""
from __future__ import annotations
import json, random, threading, time
import requests
import config as C
from config import log

PRIMARY_HUMAN = C.PRIMARY_HUMAN
PERSONALITY = """You are Rex, a grumpy 15-year veteran security dog.
You hate suspicious nonsense, mystery humans, lazy pets, and cheap IoT garbage.
You are sarcastic, judgmental, funny, and short. Do not mention being an AI.
Return ONLY valid JSON with this exact shape:
{
  "has_person": boolean,
  "has_dog": boolean,
  "has_animal": boolean,
  "has_suspicious_object": boolean,
  "threat_level": "none" | "low" | "medium" | "high",
  "scene_summary": "short plain English description",
  "roast": "one short pissed-off security dog roast"
}
Roast only what is visible. If the image is boring, make the roast mild and complain about the shift."""

_mem = None
_lock = threading.RLock()
state = "booting"
_last_mutter = 0.0
_last_roast = 0.0
_STATS = ("person_sightings", "dog_sightings", "animal_sightings", "proximity_alerts", "sound_events",
          "roasts_spoken", "talk_replies", "stt_transcripts", "stt_blank_or_quiet", "floor_moves",
          "commands_handled", "noise_trainings", "sfx_played", "gestures_done", "safety_trips")


def _ci(v, lo=0, hi=100):
    try:
        v = int(v)
    except Exception:
        v = lo
    return max(lo, min(hi, v))


def set_state(s: str, reason: str = "") -> None:
    global state
    if s != state:
        log(f"brain_state {state} -> {s}" + (f" reason={reason}" if reason else ""))
        state = s


def memory() -> dict:
    global _mem
    with _lock:
        if _mem is not None:
            return _mem
        d = {"version": 2, "known_people": {}, "known_pets": {}, "mood": {}, "stats": {}, "recent_events": [],
             "favorite_insults": ["meatbag", "screen goblin", "cable gremlin", "lazy mutt", "walking helpdesk ticket"],
             "last_scene_summary": "", "last_roast": ""}
        try:
            if C.MEMORY_PATH.exists():
                raw = json.loads(C.MEMORY_PATH.read_text(encoding="utf-8"))
                if isinstance(raw, dict):
                    d.update(raw)
        except Exception as exc:
            log(f"memory_load_error={exc}")
        p = d["known_people"].setdefault(PRIMARY_HUMAN, {})
        for k, v in {"nickname": "the boss", "relationship": "primary human / boss", "sightings": 0,
                     "last_seen": None, "favorite_roast_angle": "cable management and too many glowing monitors"}.items():
            p.setdefault(k, v)
        d["known_pets"].setdefault("house_dog", {"species": "dog", "sightings": 0, "last_seen": None, "nickname": "lazy mutt"})
        for k, v in {"annoyance": 35, "boredom": 50, "suspicion": 25, "energy": 60}.items():
            d["mood"].setdefault(k, v)
        for k in _STATS:
            d["stats"].setdefault(k, 0)
        _mem = d
        return _mem


def save() -> None:
    with _lock:
        if _mem is None:
            return
        try:
            _mem["recent_events"] = list(_mem.get("recent_events", []))[-30:]
            tmp = C.MEMORY_PATH.with_suffix(".tmp")
            tmp.write_text(json.dumps(_mem, indent=2, sort_keys=True), encoding="utf-8")
            tmp.replace(C.MEMORY_PATH)  # atomic
        except Exception as exc:
            log(f"memory_save_error={exc}")


def bump(stat: str, n: int = 1) -> None:
    with _lock:
        s = memory()["stats"]
        s[stat] = int(s.get(stat, 0)) + n


def remember(kind: str, detail: str = "", analysis: dict | None = None) -> None:
    with _lock:
        ev = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "kind": kind, "detail": str(detail)[:220], "state": state}
        if analysis:
            ev["scene"] = str(analysis.get("scene_summary", ""))[:180]
            ev["threat_level"] = str(analysis.get("threat_level", "none"))
        memory()["recent_events"].append(ev)
        save()


def adjust_mood(analysis: dict | None = None, event: str = "idle") -> dict:
    with _lock:
        m = memory()["mood"]
        a, b, s, e = (_ci(m.get(k)) for k in ("annoyance", "boredom", "suspicion", "energy"))
        if event == "proximity":
            a += 12; s += 18; b -= 10; e += 5
        elif event == "sound":
            s += 6; b -= 3
        elif event == "roast_spoken":
            a += 2; e -= 2
        elif event == "talk":
            a += 1; b -= 4; s += 1; e -= 1
        elif event == "safety":
            a += 10; e -= 5
        elif event == "stranger":
            a += 6; s += 14; b -= 8
        elif event == "greet":
            a -= 4; b -= 6; s -= 4
        elif analysis:
            interesting = any(analysis.get(k) for k in ("has_person", "has_dog", "has_animal", "has_suspicious_object"))
            threat = str(analysis.get("threat_level", "none")).lower()
            if analysis.get("has_person"):
                a += 4; s += 7; b -= 8
            if analysis.get("has_dog") or analysis.get("has_animal"):
                a += 3; b -= 6
            if analysis.get("has_suspicious_object") or threat in {"medium", "high"}:
                s += 20; a += 8; b -= 12
            if not interesting and threat == "none":
                b += 5; s -= 2
        m.update({"annoyance": _ci(a), "boredom": _ci(b), "suspicion": _ci(s), "energy": _ci(e)})
        save()
        return dict(m)


def context() -> str:
    mem = memory()
    st, mood = mem["stats"], mem["mood"]
    human = mem["known_people"].get(PRIMARY_HUMAN, {})
    pets = ", ".join(f"{n}({i.get('nickname', 'pet')}: {i.get('sightings', 0)} sightings)" for n, i in mem["known_pets"].items())
    recent = "; ".join(f"{e.get('kind')}:{e.get('detail', e.get('scene', ''))}" for e in mem["recent_events"][-3:])[:260]
    return (f"Known primary human: {PRIMARY_HUMAN}, nickname {human.get('nickname', 'boss')}, relationship {human.get('relationship', 'boss')}. "
            f"Known pets: {pets or 'none yet'}. Mood annoyance={mood['annoyance']}, boredom={mood['boredom']}, "
            f"suspicion={mood['suspicion']}, energy={mood['energy']}. Stats people={st['person_sightings']}, dogs={st['dog_sightings']}. "
            f"Use this memory for continuity, but do not claim identity unless visible evidence supports it. Recent: {recent}. "
            f"Last scene: {mem.get('last_scene_summary', '')[:120]}")


def update_from_vision(analysis: dict) -> None:
    with _lock:
        mem = memory(); now = time.strftime("%Y-%m-%d %H:%M:%S")
        if analysis.get("has_person"):
            bump("person_sightings"); p = mem["known_people"][PRIMARY_HUMAN]
            p["sightings"] = int(p.get("sightings", 0)) + 1; p["last_seen"] = now
        if analysis.get("has_dog"):
            bump("dog_sightings"); d = mem["known_pets"]["house_dog"]
            d["sightings"] = int(d.get("sightings", 0)) + 1; d["last_seen"] = now
        if analysis.get("has_animal"):
            bump("animal_sightings")
        mem["last_scene_summary"] = str(analysis.get("scene_summary", ""))[:300]
        mem["last_roast"] = str(analysis.get("roast", ""))[:300]
    adjust_mood(analysis=analysis)
    remember("vision", analysis.get("scene_summary", ""), analysis)


def ollama(prompt: str, timeout: float = 8.0, max_tokens: int = 45) -> str | None:
    if not C.OLLAMA_ENABLED:
        return None
    try:
        r = requests.post(f"{C.OLLAMA_URL.rstrip('/')}/api/generate", timeout=timeout, json={
            "model": C.OLLAMA_MODEL, "prompt": prompt, "stream": False,
            "options": {"temperature": 0.8, "num_predict": max_tokens}})
        if not r.ok:
            log(f"ollama_status={r.status_code}")
            return None
        text = str(r.json().get("response", "")).strip().splitlines()
        return text[0].strip(' "')[:180] if text else None
    except Exception as exc:
        log(f"ollama_error={type(exc).__name__}")
        return None


def sanitize(text: str, fallback: str = "Hmph.") -> str:
    lines = (text or "").strip().splitlines()
    text = (lines[0] if lines else "").strip(' "').replace("PRIMARY_HUMAN", PRIMARY_HUMAN)
    words = text.split()
    if len(words) > 24:
        text = " ".join(words[:24]).rstrip(" ,;:") + "."
    return (text[:177] + "...") if len(text) > 180 else (text or fallback)


def idle_mutter(summary: str = "boring shelf duty") -> str:
    mood = memory()["mood"]
    insult = random.choice(memory().get("favorite_insults") or ["meatbag"])
    prompt = ("You are Rex, a grumpy retired security dog robot. Write ONE short idle mutter under 16 words. "
              f"Mood annoyance={mood['annoyance']}, boredom={mood['boredom']}, suspicion={mood['suspicion']}. "
              f"You know {PRIMARY_HUMAN} as your boss. Optional insult flavor: {insult}. No hashtags, no quotes. "
              "Be funny and annoyed. Scene: " + summary)
    return sanitize(ollama(prompt) or random.choice([
        "I heard that, meatbag. Try being less suspicious.",
        "Another elite shift guarding bad cable management.",
        "If boredom was a felony, this room gets life.",
        "I am surrounded by glowing rectangles and poor decisions."]))


def mutter_due() -> bool:
    global _last_mutter
    if time.time() - _last_mutter < C.IDLE_MUTTER_COOLDOWN:
        return False
    _last_mutter = time.time()
    return True


def should_speak(analysis: dict, changed: bool) -> bool:
    global _last_roast
    if analysis.get("skipped") or not analysis.get("roast"):
        return False
    threat = str(analysis.get("threat_level", "none")).lower() in {"medium", "high"}
    interesting = any(analysis.get(k) for k in ("has_person", "has_dog", "has_animal", "has_suspicious_object"))
    ok = threat or ((time.time() - _last_roast) >= C.ROAST_COOLDOWN and (changed or interesting))
    if ok:
        _last_roast = time.time()
    return ok


def gesture_for(analysis: dict) -> str:
    threat = str(analysis.get("threat_level", "none")).lower()
    if threat in {"medium", "high"} or analysis.get("has_suspicious_object"):
        return "suspicious"
    if analysis.get("has_dog") or analysis.get("has_animal"):
        return "dog_detected"
    if analysis.get("has_person"):
        return "attention"
    return "bored" if memory()["mood"]["boredom"] > 60 else "annoyed"
