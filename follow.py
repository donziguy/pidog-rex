"""Supervised follow. Starts only with the --follow flag, never from the environment or rex.service.

Rex steps toward the bounding box of a recognized person (the primary human when
several are known). Strangers are not followed. The same safety thread watches
each step. A lost person stops the stepping without a search pattern. The session
ends at REX_FOLLOW_MAX_STEPS or REX_FOLLOW_MAX_SECONDS, then Rex sits. Guard mode
still refuses to walk.
"""
from __future__ import annotations

import time

import config as C
from config import log


def _known(hits):
    out = []
    for hit in hits or []:
        if getattr(hit, "stranger", True):
            continue
        name = getattr(hit, "name", "") or ""
        if name:
            out.append(hit)
    return out


def choose_target(hits):
    """Prefer the primary human. Otherwise the first known person. Never a stranger."""
    known = _known(hits)
    primary = (C.PRIMARY_HUMAN or "").strip().lower()
    for hit in known:
        if hit.name.lower() == primary:
            return hit
    return known[0] if known else None


def action_for_bbox(bbox) -> str:
    x, _y, w, _h = bbox
    cx = float(x) + float(w) / 2.0
    if cx < 0.40:
        return "turn_left"
    if cx > 0.60:
        return "turn_right"
    return "forward"


class Follow:
    def __init__(self, session):
        self.session = session
        self.active = False
        self.steps = 0
        self.misses = 0
        self.started = 0.0

    def start(self) -> bool:
        guard = getattr(self.session, "guard", None)
        if guard is not None and not guard.allow_walk():
            log("follow_refused guard_armed")
            return False
        allow = bool(getattr(self.session.args, "allow_no_imu", False))
        stale = bool(getattr(self.session.hw, "imu_stale", False))
        if stale and not allow:
            log("follow_refused imu_stale")
            return False
        if not self.session.hw.imu_ok and not allow:
            log("follow_refused imu_unavailable")
            return False
        self.active = True
        self.steps = 0
        self.misses = 0
        self.started = time.time()
        log(f"follow_enabled steps={C.FOLLOW_MAX_STEPS} seconds={C.FOLLOW_MAX_SECONDS} speed={C.FOLLOW_SPEED}")
        return True

    def stop(self, why: str) -> None:
        if self.active:
            log(f"follow_disabled why={why} steps={self.steps}")
        self.active = False

    def tick(self, hits=None) -> str:
        if not self.active:
            return "off"
        guard = getattr(self.session, "guard", None)
        if guard is not None and not guard.allow_walk():
            self.stop("guard_armed")
            return "guard"
        if not self.session.safety.allow_move():
            self.stop("locked")
            return "locked"
        if self.steps >= C.FOLLOW_MAX_STEPS or (time.time() - self.started) > C.FOLLOW_MAX_SECONDS:
            log("follow_cap")
            self.session.loco._do("sit", 1, 40, walking=False)
            self.stop("cap")
            return "cap"
        if self.session.safety.distance_hold:
            outcome = self.session.loco.poll_distance_hold()
            if outcome != "clear":
                if outcome in ("halt", "locked", "aborted"):
                    self.stop(outcome)
                return outcome
        target = choose_target(hits if hits is not None else self.session.face_hits)
        if target is None:
            self.misses += 1
            log(f"follow_lost misses={self.misses}")
            if self.misses >= C.FOLLOW_LOST_TICKS:
                self.stop("lost")
            return "lost"
        self.misses = 0
        action = action_for_bbox(target.bbox)
        ok = self.session.loco._do(action, 1, C.FOLLOW_SPEED, walking=True)
        if self.session.safety.distance_hold and self.session.safety.allow_move():
            # The step was cut short because the ranger went quiet. Stay in
            # follow and wait; do not take another blind step. A real safety
            # lock still falls through to the aborted return below.
            return "hold"
        if not ok:
            self.stop("aborted")
            return "aborted"
        self.steps += 1
        log(f"follow_step action={action} steps={self.steps} target={target.name}")
        return action
