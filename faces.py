"""Local face enrollment and recognition.

Off unless ``REX_FACE=1``. Embeddings are written to ``~/.config/pidog/faces``
(or ``REX_FACE_DIR``), which is outside the git repo. Backends, in order when
``REX_FACE_BACKEND=auto``:

* ``face_recognition`` (dlib) when that package imports
* OpenCV YuNet + SFace when ``REX_YUNET_MODEL`` and ``REX_SFACE_MODEL`` point at files
* OpenCV Haar cascade, embedding the crop with the Pillow embedder
* Pillow 8x8 luminance embedder (always available; what CI uses)

The Pillow path only *recognizes* enrolled people. It does not invent a stranger
from an empty shelf. A real detector, or a simulated ``dog.people`` sighting,
marks a non-matching face as a stranger. Cory (``PIDOG_PRIMARY_HUMAN``) is greeted.
Unknowns are remembered as strangers and are not enrolled.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

import config as C
from config import log

# None means "never". Zero is not safe: time.monotonic() is small on a fresh boot,
# so a 0 baseline looks like the cooldown is still running.
_last_greet = None
_last_stranger = None


def reset_state() -> None:
    global _last_greet, _last_stranger
    _last_greet = None
    _last_stranger = None


class FaceHit:
    def __init__(self, name: str, stranger: bool, distance: float, bbox, source: str):
        self.name = name or ""
        self.stranger = bool(stranger)
        self.distance = float(distance)
        self.bbox = tuple(float(v) for v in (bbox or (0.2, 0.15, 0.6, 0.7)))
        self.source = source

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "stranger": self.stranger,
            "distance": round(self.distance, 4),
            "bbox": list(self.bbox),
            "source": self.source,
        }


def cosine_distance(a, b) -> float:
    if not a or not b or len(a) != len(b):
        return 1.0
    dot = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(a, b):
        x = float(x)
        y = float(y)
        dot += x * y
        na += x * x
        nb += y * y
    if na <= 0.0 or nb <= 0.0:
        return 1.0
    sim = max(-1.0, min(1.0, dot / math.sqrt(na * nb)))
    return 1.0 - sim


def embed_pillow(image) -> list[float]:
    """8x8 grayscale levels in 0..1. Absolute brightness is kept so a dark frame is not a bright one."""
    im = image.convert("L").resize((8, 8))
    pixels = im.get_flattened_data() if hasattr(im, "get_flattened_data") else im.getdata()
    return [round(p / 255.0, 5) for p in pixels]


def _valid_name(name: str) -> bool:
    name = (name or "").strip()
    if not name or len(name) > 40:
        return False
    return all(ch.isalpha() or ch in " -'" for ch in name)


class PillowBackend:
    name = "pillow"
    detects_strangers = False

    def detect(self, image):
        return [((0.2, 0.15, 0.6, 0.7), embed_pillow(image))]


class FaceRecognitionBackend:
    """dlib embeddings. One hit per face box. An unmatched box is a stranger."""

    name = "face_recognition"
    detects_strangers = True

    def detect(self, image):
        import face_recognition
        arr = _rgb_array(image)
        locs = face_recognition.face_locations(arr) or []
        if not locs:
            return []
        encs = face_recognition.face_encodings(arr, locs) or []
        height, width = arr.shape[:2]
        out = []
        for (top, right, bottom, left), enc in zip(locs, encs):
            bbox = (left / width, top / height, max(1, right - left) / width, max(1, bottom - top) / height)
            out.append((bbox, [float(x) for x in enc]))
        return out


class YuNetSFaceBackend:
    name = "opencv-sface"
    detects_strangers = True

    def __init__(self, cv2, yunet: str, sface: str):
        self.cv2 = cv2
        self.yunet = yunet
        self.sface = sface

    def detect(self, image):
        import numpy as np
        cv2 = self.cv2
        bgr = cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2BGR)
        height, width = bgr.shape[:2]
        detector = cv2.FaceDetectorYN.create(self.yunet, "", (width, height), score_threshold=0.6)
        found = detector.detect(bgr)
        faces = found[1] if isinstance(found, tuple) and len(found) > 1 else found
        if faces is None:
            return []
        recognizer = cv2.FaceRecognizerSF.create(self.sface, "")
        out = []
        for row in faces:
            aligned = recognizer.alignCrop(bgr, row)
            feat = recognizer.feature(aligned)
            vec = [float(x) for x in feat.flatten()]
            x, y, bw, bh = (float(v) for v in row[:4])
            bbox = (x / width, y / height, bw / width, bh / height)
            out.append((bbox, vec))
        return out


class HaarBackend:
    """Stock OpenCV cascade. The crop is embedded with Pillow so no extra model file is required."""

    name = "opencv-haar"
    detects_strangers = True

    def __init__(self, cv2, cascade: str):
        self.cv2 = cv2
        self.cascade = cascade

    def detect(self, image):
        import numpy as np
        cv2 = self.cv2
        rgb = np.array(image.convert("RGB"))
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        clf = cv2.CascadeClassifier(self.cascade)
        rects = clf.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)
        if rects is None or len(rects) == 0:
            return []
        height, width = gray.shape[:2]
        out = []
        for (x, y, bw, bh) in rects:
            crop = image.crop((int(x), int(y), int(x + bw), int(y + bh)))
            bbox = (float(x) / width, float(y) / height, float(bw) / width, float(bh) / height)
            out.append((bbox, embed_pillow(crop)))
        return out


def _rgb_array(image):
    import numpy as np
    return np.array(image.convert("RGB"))


def _face_stack_available() -> bool:
    try:
        import face_recognition  # noqa: F401
        import numpy  # noqa: F401
        return True
    except Exception:
        return False


def get_backend():
    prefer = (C.FACE_BACKEND or "auto").lower()
    if prefer in {"auto", "face_recognition"}:
        if _face_stack_available():
            return FaceRecognitionBackend()
        if prefer == "face_recognition":
            log("face_backend_missing face_recognition; using pillow")
    if prefer in {"auto", "opencv"}:
        try:
            import cv2
            yunet, sface = C.YUNET_MODEL, C.SFACE_MODEL
            if yunet and sface and Path(yunet).exists() and Path(sface).exists():
                return YuNetSFaceBackend(cv2, yunet, sface)
            cascade = Path(getattr(cv2.data, "haarcascades", "")) / "haarcascade_frontalface_default.xml"
            if cascade.exists():
                return HaarBackend(cv2, str(cascade))
        except Exception as exc:
            if prefer == "opencv":
                log(f"face_backend_missing opencv ({type(exc).__name__}); using pillow")
    return PillowBackend()


class FaceStore:
    def __init__(self, root: Path | None = None):
        self.root = Path(root or C.FACE_DIR)
        self.path = self.root / "index.json"

    def load(self) -> dict:
        if not self.path.exists():
            return {"version": 1, "faces": []}
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except Exception as exc:
            log(f"face_index_error={type(exc).__name__}")
            return {"version": 1, "faces": []}
        data.setdefault("version", 1)
        data.setdefault("faces", [])
        return data

    def save(self, data: dict) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(data), encoding="utf-8")
        os.replace(tmp, self.path)
        try:
            os.chmod(self.path, 0o600)
        except Exception:
            pass

    def names(self) -> list[str]:
        seen = []
        for face in self.load()["faces"]:
            name = face.get("name") or ""
            if name and name not in seen:
                seen.append(name)
        return seen

    def has(self, name: str) -> bool:
        want = (name or "").strip().lower()
        return any((face.get("name") or "").lower() == want for face in self.load()["faces"])

    def enroll(self, name: str, embedding, backend: str) -> None:
        data = self.load()
        kept = [f for f in data["faces"]
                if not ((f.get("name") or "").lower() == name.lower() and f.get("backend") == backend)]
        kept.append({
            "name": name,
            "backend": backend,
            "embedding": [float(x) for x in embedding],
            "enrolled": time.strftime("%Y-%m-%d %H:%M:%S"),
        })
        data["faces"] = kept
        self.save(data)
        log(f"face_enrolled name={name} backend={backend} count={len(kept)}")

    def match(self, embedding, backend: str) -> tuple[str, float]:
        best_name = ""
        best = 1.0
        for face in self.load()["faces"]:
            if face.get("backend") != backend:
                continue
            dist = cosine_distance(face.get("embedding") or [], embedding)
            if dist < best:
                best = dist
                best_name = face.get("name") or ""
        if best_name and best <= C.FACE_THRESHOLD:
            return best_name, best
        return "", best


def recognize_image(image, store: FaceStore | None = None, backend=None) -> list[FaceHit]:
    store = store or FaceStore()
    backend = backend or get_backend()
    try:
        found = backend.detect(image) or []
    except Exception as exc:
        log(f"face_detect_error={type(exc).__name__}")
        return []
    strangers = bool(getattr(backend, "detects_strangers", False))
    hits = []
    for bbox, emb in found:
        name, dist = store.match(emb, backend.name)
        if name:
            hits.append(FaceHit(name, False, dist, bbox, backend.name))
        elif strangers:
            hits.append(FaceHit("", True, dist, bbox, backend.name))
    return hits


def _sim_people(hw):
    if not getattr(hw, "simulated", False):
        return []
    sensor = getattr(hw.dog, "people", None)
    if sensor is None:
        return []
    try:
        people = sensor.peek() or []
    except Exception:
        return []
    return list(people)


def hits_from_sim(people, store: FaceStore) -> list[FaceHit]:
    hits = []
    for person in people:
        bbox = person.get("bbox") or (0.2, 0.15, 0.6, 0.7)
        name = str(person.get("name") or "").strip()
        if "stranger" in person:
            stranger = bool(person["stranger"])
        elif person.get("known") is True and name:
            stranger = False
        elif name and store.has(name):
            stranger = False
        else:
            stranger = True
            name = ""
        hits.append(FaceHit("" if stranger else name, stranger, 0.0 if not stranger else 1.0, bbox, "sim"))
    return hits


def _open_photo(photo):
    from PIL import Image
    if photo is None:
        return None
    if hasattr(photo, "convert"):
        return photo
    path = Path(photo)
    if not path.exists():
        return None
    with Image.open(path) as im:
        return im.convert("RGB")


def recognize_hw(hw, store: FaceStore | None = None, photo=None, backend=None) -> list[FaceHit]:
    store = store or FaceStore()
    people = _sim_people(hw)
    if people:
        return hits_from_sim(people, store)
    image = _open_photo(photo)
    if image is None:
        path = hw.photo()
        image = _open_photo(path)
    if image is None:
        return []
    return recognize_image(image, store=store, backend=backend)


def enroll_image(image, name: str, store: FaceStore | None = None, backend=None) -> bool:
    if not _valid_name(name):
        log("face_enroll_rejected bad_name")
        return False
    store = store or FaceStore()
    backend = backend or get_backend()
    try:
        found = backend.detect(image) or []
    except Exception as exc:
        log(f"face_enroll_error={type(exc).__name__}")
        return False
    if not found:
        log(f"face_enroll_failed name={name} reason=no_face")
        return False
    _bbox, emb = max(found, key=lambda item: item[0][2] * item[0][3])
    store.enroll(name.strip(), emb, backend.name)
    return True


def enroll_camera(hw, name: str, store: FaceStore | None = None) -> bool:
    path = hw.photo()
    image = _open_photo(path)
    if image is None:
        log("face_enroll_failed reason=no_photo")
        return False
    return enroll_image(image, name, store=store)


def _primary() -> str:
    return (C.PRIMARY_HUMAN or "Cory").strip() or "Cory"


def observe(session, photo=None, store: FaceStore | None = None) -> list[FaceHit]:
    """Recognize whoever is in frame. Greet the primary human. Strangers stay strangers."""
    if not C.FACE_ENABLED:
        session.face_hits = []
        session.person_seen = "none"
        return []
    store = store or FaceStore()
    hits = recognize_hw(session.hw, store=store, photo=photo)
    session.face_hits = hits
    if not hits:
        session.person_seen = "none"
        return hits
    primary = _primary().lower()
    known = [h for h in hits if not h.stranger and h.name]
    strangers = [h for h in hits if h.stranger]
    primary_hit = next((h for h in known if h.name.lower() == primary), None)
    if primary_hit is not None:
        session.person_seen = primary_hit.name
        _greet(session, primary_hit.name)
    elif known:
        session.person_seen = known[0].name
        _greet(session, known[0].name)
    elif strangers:
        session.person_seen = "stranger"
        _note_stranger(session)
    else:
        session.person_seen = "none"
    return hits


def _greet(session, name: str) -> None:
    global _last_greet
    import audio
    import brain
    now = time.monotonic()
    brain.remember("seen", name)
    if _last_greet is not None and now - _last_greet < C.FACE_GREET_COOLDOWN:
        return
    _last_greet = now
    audio.speak(f"Hello {name}.", bool(session.tts))
    audio.sfx("happy", bool(session.tts))
    if not getattr(getattr(session, "loco", None), "enabled", False):
        try:
            session.hw.wag(2, 40)
        except Exception:
            pass
    mem = brain.memory()
    person = mem["known_people"].setdefault(name, {})
    person["sightings"] = int(person.get("sightings", 0)) + 1
    person["last_seen"] = time.strftime("%Y-%m-%d %H:%M:%S")
    brain.adjust_mood(event="greet")
    brain.remember("greet", name)
    log(f"face_greet name={name}")


def _note_stranger(session) -> None:
    global _last_stranger
    import audio
    import brain
    now = time.monotonic()
    brain.remember("stranger", "unknown face")
    brain.adjust_mood(event="stranger")
    if _last_stranger is not None and now - _last_stranger < C.FACE_STRANGER_COOLDOWN:
        return
    _last_stranger = now
    audio.speak("I don't know you. That's suspicious.", bool(session.tts))
    log("face_stranger")


def _ext(dash) -> None:
    from flask import jsonify, request

    def do_enroll(payload):
        name = str(payload.get("name") or "").strip()
        ok = enroll_camera(dash.hw, name)
        return {"enrolled": bool(ok), "name": name}

    dash.on("enroll", do_enroll)

    def status(body):
        body["face_enabled"] = bool(C.FACE_ENABLED)
        try:
            body["known_faces"] = FaceStore().names()
        except Exception:
            body["known_faces"] = []

    dash.status_hooks.append(status)

    @dash.app.post("/api/enroll")
    def api_enroll():
        if not C.FACE_ENABLED:
            return jsonify({"ok": False, "error": "face_disabled"}), 403
        name = str((request.get_json(silent=True) or {}).get("name") or "").strip()
        if not _valid_name(name):
            return jsonify({"ok": False, "error": "bad_name"}), 400
        dash.submit("enroll", name=name)
        return jsonify({"ok": True, "queued": "enroll"}), 202


def ensure_registered() -> None:
    import dashboard
    if _ext not in dashboard.extensions:
        dashboard.extensions.append(_ext)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="faces", description="Enroll and list Rex's local faces")
    sub = parser.add_subparsers(dest="cmd", required=True)
    enroll = sub.add_parser("enroll", help="enroll one person from a still or the camera")
    enroll.add_argument("name")
    enroll.add_argument("--image", type=Path, default=None)
    sub.add_parser("list", help="print enrolled names")
    recognize = sub.add_parser("recognize", help="recognize faces in a still or the camera")
    recognize.add_argument("--image", type=Path, default=None)
    args = parser.parse_args(argv)
    store = FaceStore()
    if args.cmd == "list":
        for face in store.load()["faces"]:
            print(f"{face.get('name')}\t{face.get('backend')}")
        return 0
    if args.cmd == "enroll" and args.image:
        from PIL import Image
        with Image.open(args.image) as im:
            return 0 if enroll_image(im.convert("RGB"), args.name, store=store) else 1
    if args.cmd == "recognize" and args.image:
        from PIL import Image
        with Image.open(args.image) as im:
            hits = recognize_image(im.convert("RGB"), store=store)
        if not hits:
            print("none")
            return 0
        for hit in hits:
            label = "stranger" if hit.stranger else hit.name
            print(f"{label}\t{hit.distance:.3f}\t{hit.source}")
        return 0
    # Camera path: one Pidog (or SimDog), and only if Rex is not already running.
    import main as rex
    rex.single_instance()
    from hardware import Hardware
    hw = Hardware()
    try:
        if args.cmd == "enroll":
            return 0 if enroll_camera(hw, args.name, store=store) else 1
        hits = recognize_hw(hw, store=store)
    finally:
        hw.shutdown()
    if args.cmd == "recognize":
        if not hits:
            print("none")
        for hit in hits:
            label = "stranger" if hit.stranger else hit.name
            print(f"{label}\t{hit.distance:.3f}\t{hit.source}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
