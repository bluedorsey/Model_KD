"""Response tiers with hysteresis (roadmap step 14, README §9) and the README §10 failure rules.

  0 Normal  1 Monitor (p<0.01)  2 Shape (p<0.001)  3 Verify (p<1e-4, or tier 2 sustained)
  4 Suspend (account p AND cluster p below 1e-4: two independent families, or failed verification)

Up: immediately, unless the window is incomplete or scoring is frozen (never escalate on missing data).
Down: one tier after `clean_windows` consecutive clean windows; tier 3 only by verification; tier 4 by
manual review. Cluster-inherited tiers 1-2 expire when cluster state is stale; 3-4 persist.

`block_only` (stage 3 D2/D3, README §12.3 rule 6): no graded tiers; p < p_tier3 suspends (tier 4), nothing else.
"""
from gateway.schema import DAY


def p_to_tier(p, P):
    return 3 if p < P["p_tier3"] else 2 if p < P["p_tier2"] else 1 if p < P["p_tier1"] else 0


class Policy:
    def __init__(self, cfg):
        self.P = cfg["policy"]
        self.own = {}        # account -> own tier
        self.clean = {}      # account -> consecutive clean windows
        self.t2 = {}         # account -> windows spent at tier 2
        self.cluster = {}    # account -> (inherited tier, ts set)
        self.verified_until = {}
        self.p_cluster = {}  # account -> last inherited cluster p

    def update_account(self, a, p, ts, incomplete=False, frozen=False):
        if frozen:
            return
        if self.P.get("block_only"):
            if p < self.P["p_tier3"] and not incomplete:
                self.own[a] = 4
            return
        P, cur = self.P, self.own.get(a, 0)
        if cur >= 3:
            pass  # only verification or review moves tiers 3-4 down
        else:
            target = p_to_tier(p, P)
            if target > cur and not incomplete:
                cur, self.clean[a] = target, 0
            elif target < cur:
                self.clean[a] = self.clean.get(a, 0) + 1
                if self.clean[a] >= P["clean_windows"]:
                    cur, self.clean[a] = cur - 1, 0
            else:
                self.clean[a] = 0
            self.t2[a] = self.t2.get(a, 0) + 1 if cur == 2 else 0
            if self.t2[a] >= P["sustained_tier2_windows"]:
                cur = 3
        if p < P["p_tier3"] and self.p_cluster.get(a, 1.0) < P["p_tier3"] and not incomplete:
            cur = 4
        self.own[a] = cur

    def update_cluster(self, a, p_cluster, ts):
        self.p_cluster[a] = p_cluster
        if self.P.get("block_only"):
            if p_cluster < self.P["p_tier3"]:
                self.cluster[a] = (4, ts)
            return
        self.cluster[a] = (p_to_tier(p_cluster, self.P), ts)

    def tier(self, a, now):
        own = self.own.get(a, 0)
        ct, cts = self.cluster.get(a, (0, now))
        if ct <= 2 and now - cts > self.P["stale_cluster_days"] * DAY:
            ct = 0
        t = max(own, ct)
        if t < 4 and self.verified_until.get(a, -1) > now:
            t = min(t, 1)
        return t

    def verified(self, a, now, passed):
        if passed:
            self.own[a] = min(self.own.get(a, 0), 1)
            self.t2[a] = 0
            self.verified_until[a] = now + self.P["verify_grace_days"] * DAY
        else:
            self.own[a] = 4
