"""Six DISTINCT, paper-faithful biologically-plausible motor learners.

Why this file exists (capstone review): the earlier zoo collapsed e-prop / RTRRL / BTSP /
R-STDP / predictive-coding / 3-factor-Hebb into ONE `Reservoir` class with a single `elig`
flag -- so R-STDP and predictive coding had literally the same code and the same diagram.
That is not "13 different algorithms". Here each rule is a genuinely different architecture
+ local learning rule, faithful to its source paper, with a DISTINCT diagram.

Design contract (so every rule actually solves the reach and clears the completion bar):
every method shares ONE explicitly-labelled memory crutch -- a fixed high-dimensional
recurrent RESERVOIR (echo-state / liquid substrate; cortical recurrent dynamics, NOT trained
by BPTT or the plant gradient) -- and the environment's morphological / muscle HEAD for
actuation. What differs, and carries each method's identity, is:
  * the plastic component and its local update rule,
  * the eligibility / trace form,
  * the third factor (learning signal / dopamine / prediction error / perturbation reward),
  * and a characteristic architectural element that shows up in the diagram
    (ALIF adaptation units, a random-feedback pathway, a dendritic plateau gate, spiking
     units + STDP window, explicit error units, or exploratory node perturbation).
None uses BPTT. All six minimise MotorNet's OWN task loss (L1 fingertip-to-goal position
error) -- the identical objective the gradient models get -- and convert that task-space error
into a local weight change through a FIXED feedback projection (feedback alignment), never
through the plant's Jacobian. There is no demonstrator: an earlier imitation setup was removed
because it could not be posed identically across heads. See motor_core.py.

Interface (zoo harness): nn.Module with .name/.cite/.kind/.wins, .init_state(B),
.act(obs,state)->(action,state), .fit(env,budget,probe,batch), .ops(env)->OpCounter. The
host zoo calls `configure(head, obs_norm, OpCounter)` once after import.
"""
import math
import torch as th
import torch.nn as nn

HEAD = None; OBS_NORM = None; OPCOUNTER = None
KIND = "local-plausible"
RES_NR = 4096
REFLEX_KP, REFLEX_KD = 1.0, 0.15   # spinal PD reflex: sweep optimum (5.2cm/59% alone)
REFLEX_AVOID = 8.0   # maze: gain on the outward obstacle-avoidance force added to the reflex target
                     # (a flexor-withdrawal reflex -- pushes the endpoint out of a barrier it entered)
SPINDLE_KD = 0.15   # Ia velocity sensitivity (s). See project_error.
REFLEX_GAIN = 4.0   # sensory error (m) -> endpoint-force command. The monosynaptic stretch
                    # reflex's loop gain: the one constant a local rule must assume in place
                    # of the plant Jacobian it is not allowed to compute.                       # rich fixed reservoir -> strong linear imitation ceiling

def configure(head, obs_norm, OpCounter):
    global HEAD, OBS_NORM, OPCOUNTER
    HEAD, OBS_NORM, OPCOUNTER = head, obs_norm, OpCounter


def _raw_dim(teacher):
    """Width of the plastic readout = width the action head consumes. With no demonstrator in
    the picture this is simply the morphological head's 3 (force_x, force_y, co-contraction)."""
    if teacher is None: return 3
    for attr in ("fc", "head"):
        m = getattr(teacher, attr, None)
        if isinstance(m, nn.Linear): return m.out_features
    return 3


RES_P = 0.10        # recurrent connectivity. Cortex is SPARSE (~0.1 local connection probability) and
                    # reservoir computing has used sparse Wr since Jaeger 2001. A fully dense 4096x4096
                    # recurrent matrix is neither biologically plausible NOR honest to charge as energy:
                    # it is 16.8M MACs/step (~77 nJ) purely because of the crutch's size.


def _reservoir(O, Nr, dev, rho=1.1, sin=1.0, seed=0, p=RES_P):
    g = th.Generator(device="cpu").manual_seed(seed)
    Wr = th.randn(Nr, Nr, generator=g)
    if p < 1.0:
        Wr *= (th.rand(Nr, Nr, generator=g) < p).float()     # sparse mask: ~p*Nr presynaptic contacts/unit
    ev = th.linalg.eigvals(Wr).abs().max().item()
    Wr *= rho / max(ev, 1e-8)                                # re-normalise spectral radius AFTER masking
    Win = sin * th.randn(Nr, O, generator=g); bres = 0.1 * th.randn(Nr, generator=g)
    return Wr.to(dev), Win.to(dev), bres.to(dev)


class _ResBase(nn.Module):
    """Fixed reservoir (memory crutch) + a plastic readout. Subclasses override the DISTINCT
    plasticity (fit) and, where it is part of the method's identity, the substrate (_recur)."""
    kind = KIND
    def __init__(self, env, teacher=None, Nr=RES_NR, a=0.5, rho=1.1, lam=1e-4, seed=0):
        super().__init__()
        self.dev = env.device; self.teacher = teacher; self.O = env.observation_space.shape[0]
        self.Nr = Nr; self.R = _raw_dim(teacher); self.dt = env.dt; self.a = a; self.lam = lam
        Wr, Win, bres = _reservoir(self.O, Nr, self.dev, rho=rho, seed=seed)
        self.register_buffer("Wr", Wr); self.register_buffer("Win", Win); self.register_buffer("bres", bres)
        self.register_buffer("W", th.zeros(self.R, Nr + self.O, device=self.dev))
        b0 = th.zeros(self.R, device=self.dev)
        if self.R == 3: b0[2] = -3.0                     # start compliant (low co-contraction) for the 2-D morph head
        self.register_buffer("b", b0)
        mu, sig = OBS_NORM(env); self.register_buffer("mu", mu); self.register_buffer("sig", sig)
        # FIXED feedback from TASK space (2-D fingertip error) into this readout's R raw units.
        # The shared objective is MotorNet's L1 position error; a local rule cannot backprop the
        # Jacobian of body+head to convert that into a readout update, so it uses a fixed
        # projection instead (feedback alignment, Lillicrap+16). Fixed => not a learned
        # parameter, and built identically for every rule, so no rule gets a better signal.
        #
        # The first two raw units of the morphological head ARE an endpoint force, and the
        # direction that reduces a position error IS a force in that direction, so those two
        # channels get an identity map -- a spinal-reflex-like sensory-to-force projection,
        # not a lucky hyperparameter. Remaining channels (co-contraction) get small random
        # feedback. For a non-morphological readout this degrades to pure random feedback.
        Bfb = th.zeros(2, self.R); Bfb[0, 0] = Bfb[1, 1] = REFLEX_GAIN
        self.register_buffer("Bfb", Bfb.to(self.dev)); self._pe = None
        self.to(self.dev)

    def project_error(self, c):
        """Shared task error -> a DELTA on this readout's raw command, via the SPINAL REFLEX.

        A local rule needs a TARGET, not just an error: a rule of the form dW ~ err (x) z
        converges to a correlation solution whose command merely tracks position error --
        proportional control -- and a P-controller on a point mass orbits its goal forever
        (measured: 25 cm floor, 0% completion, unchanged from 8k to 80k episodes).

        The reflex supplies that target from quantities the organism actually has: the task
        error itself (goal - fingertip, THE shared objective's signal) and proprioceptive
        velocity. It is a monosynaptic stretch reflex written as PD, and on its own -- with
        no learning whatsoever -- it already reaches 5.2 cm / 59% completion. So it is a
        sound thing to regress onto, and the readout can then beat it by becoming
        state- and mass-dependent, which a fixed reflex cannot be.

        NOTE this introduces NO demonstrator: no trained network supplies the target, and
        the constants are the same two numbers for all six rules.
        """
        reach = REFLEX_KP * c.err - REFLEX_KD * c.vel
        # maze: a spinal obstacle-avoidance (flexor-withdrawal) reflex adds an outward push whenever
        # the endpoint has entered a barrier -- so the SAME reflex target now says "reach the goal AND
        # get out of walls", from task-space signals only (no plant Jacobian, no privileged state).
        if getattr(c, "collision_force", None) is not None:
            reach = reach + REFLEX_AVOID * c.collision_force
            # (A stall-triggered LATERAL "wall-following" term and a proximity repulsion were both tried
            # to lift the no-vision maze ceiling; both REVERTED -- any lateral/avoidance force beyond the
            # inside push-out disrupts the reach (e-prop 91%->41%->0% as the gain rises). Kept reach + the
            # inside withdrawal only, which is the honest best.)
        tgt = th.cat([reach, c.cmd[:, 2:]], -1)
        return tgt - c.cmd

    # the shared core asks every learner for its action head; ours is the module-level head
    # installed by configure() (morphological for the plausible family).
    @property
    def HEAD(self): return HEAD
    def init_state(self, B): return th.zeros(B, self.Nr, device=self.dev)
    def _recur(self, x, h):
        return (1 - self.a) * h + self.a * th.tanh(h @ self.Wr.t() + x @ self.Win.t() + self.bres)
    def _feat(self, obs, h):
        x = (obs - self.mu) / self.sig; h = self._recur(x, h); return h, th.cat([h, x], -1)
    def act(self, obs, h, explore=False):
        h, z = self._feat(obs, h); return HEAD(obs, z @ self.W.t() + self.b), h
    def ops(self, env):
        """Inference ops for ONE control step. The recurrent term costs one MAC per synapse that
        ACTUALLY EXISTS (the reservoir is sparse); charging a dense Nr x Nr matrix measured the size
        of the memory crutch, not the algorithm, and inflated every local rule by ~1/RES_P."""
        c = OPCOUNTER()
        c.mac += float((self.Wr != 0).sum().item())          # sparse recurrent synapses
        c.dense(self.O, self.Nr); c.dense(self.Nr + self.O, self.R); return c

    def update_ops(self, env):
        """Cost of ONE learning update per control step. A local rule touches only the plastic
        readout -- a rank-1 outer product dW ~ err (x) z. NO backward pass, NO BPTT, no stored
        trajectory. This is the axis on which plausible learning genuinely wins; it is invisible
        if you only count inference (which tracks the size of the fixed reservoir crutch)."""
        c = OPCOUNTER(); c.dense(self.Nr + self.O, self.R); return c


# =============================================================================
# 1. e-prop  (Bellec et al. 2020, Nat. Commun.)  -- ADAPTIVE units + eligibility trace
# -----------------------------------------------------------------------------
# The reservoir units are ADAPTIVE (ALIF): each carries a slow adaptation current a_t that
# raises its effective threshold -- the long-timescale state e-prop is named for. Credit is
# assigned by the e-prop factorisation: an ELIGIBILITY TRACE e = low-pass(psi * reservoir
# activity) gated by the postsynaptic pseudo-derivative psi, combined with a top-down
# LEARNING SIGNAL L = err. dW ~ L (x) e, computed forward in time -- no BPTT. The adaptation
# current + the temporal eligibility trace are what set e-prop apart from the instantaneous
# RFLO rule below.
# =============================================================================
class EProp(_ResBase):
    name = "e-prop (ALIF reservoir · eligibility trace + learning signal)"
    cite = "Bellec+20 e-prop (Nat.Commun.); adaptive units, forward eligibility; morphological head"
    wins = "sample efficiency (eligibility credit assignment, no BPTT)"
    def __init__(self, env, teacher=None, lr=0.02, tau_e=0.05, beta=0.4, rho_a=0.9, **kw):
        super().__init__(env, teacher, **kw); self.lr, self.tau_e, self.beta, self.rho_a = lr, tau_e, beta, rho_a
    def init_state(self, B):
        z = th.zeros(B, self.Nr, device=self.dev); return (z, z.clone())
    def _recur(self, x, st):
        h, adap = st
        h = (1 - self.a) * h + self.a * th.tanh(h @ self.Wr.t() + x @ self.Win.t() + self.bres - self.beta * adap)
        return (h, self.rho_a * adap + h)
    def _feat(self, obs, st):
        x = (obs - self.mu) / self.sig; st = self._recur(x, st); return st, th.cat([st[0], x], -1)
    # ---- migrated to the shared core: the loop + the objective now live in motor_core.train ----
    def forward(self, obs, st):
        st, z = self._feat(obs, st)
        return z @ self.W.t() + self.b, st, z                  # aux = reservoir feature vector z

    def on_episode_start(self, B):
        self._elig = th.zeros(B, self.Nr + self.O, device=self.dev)
        self._decay = 1 - self.dt / self.tau_e

    @th.no_grad()
    def on_step(self, c):
        """e-prop credit assignment: eligibility trace gated by the pseudo-derivative, combined
        with the SHARED learning signal c.err_local. This method is the ONLY thing that differs from
        every other rule -- the rollout, the target and the budget are the core's."""
        z, st = c.aux, c.state
        psi = 1 - st[0] ** 2                                    # ALIF pseudo-derivative
        zg = th.cat([self.a * psi, th.ones_like(c.obs)], -1) * z
        self._elig = self._decay * self._elig + (1 - self._decay) * zg
        self.W += self.lr / c.n * (c.err_local.t() @ self._elig / c.batch - self.lam * self.W)
        self.b += self.lr / c.n * c.err_local.mean(0)

    def fit(self, env, budget, probe, batch=256):
        import motor_core as _core                              # lazy: avoids a circular import
        return _core.train(self, env, budget, probe, batch=batch)


# =============================================================================
# 2. RTRRL / RFLO  (Murray 2019, eLife)  -- real-time recurrent learning, random feedback
# -----------------------------------------------------------------------------
# Random-Feedback Local Online learning. RFLO's two signatures, both implemented here on the
# plastic readout (the reservoir is the fixed shared crutch, so the readout is the plastic layer):
#   (1) a LEAKY eligibility trace of presynaptic activity  p <- (1-dt/tau) p + (dt/tau) aux
#       -- Murray's forward, real-time credit trace (NOT instantaneous; eq. 6 is a leaky integrator),
#   (2) the readout error routed through a FIXED RANDOM FEEDBACK matrix B (feedback alignment,
#       no weight transport) -- the RFLO signature.
# Real-time, forward-in-time, no BPTT, no stored trajectory. Distinct from e-prop (no ALIF
# pseudo-derivative gate) and from a plain delta rule (which has neither the trace nor B).
# =============================================================================
class RTRRL(_ResBase):
    name = "RTRRL / RFLO (real-time recurrent · leaky eligibility + random feedback)"
    cite = "Murray 19 RFLO (eLife); leaky eligibility trace + feedback alignment; morphological head"
    wins = "online adaptation (real-time forward eligibility, no BPTT)"
    def __init__(self, env, teacher=None, lr=0.01, tau_e=0.1, seed=1, **kw):
        super().__init__(env, teacher, seed=seed, **kw); self.lr, self.tau_e = lr, tau_e
        g = th.Generator(device="cpu").manual_seed(seed + 5)
        # RFLO's fixed RANDOM FEEDBACK: the readout error is projected through B (no weight transport)
        # to assign credit -- feedback alignment (Murray 2019). Identity + a small rotation keeps the
        # misalignment mild enough that the linear readout still converges.
        self.register_buffer("B", (th.eye(self.R) + 0.1 * th.randn(self.R, self.R, generator=g)).to(self.dev))
    def on_episode_start(self, B):
        self._elig = th.zeros(B, self.Nr + self.O, device=self.dev)
        self._decay = 1 - self.dt / self.tau_e
    def forward(self, obs, h):
        h, z = self._feat(obs, h)
        return z @ self.W.t() + self.b, h, z

    @th.no_grad()
    def on_step(self, c):
        """RFLO (Murray 2019): a LEAKY eligibility trace of presynaptic activity (real-time,
        forward-in-time), with the readout error routed through the FIXED random-feedback matrix B
        (feedback alignment, no weight transport). Distinct from e-prop (no ALIF pseudo-derivative
        gate) and from a plain delta rule (which has neither the trace nor B)."""
        if getattr(self, "_elig", None) is None or self._elig.shape[0] != c.aux.shape[0]:
            self.on_episode_start(c.aux.shape[0])
        self._elig = self._decay * self._elig + (1 - self._decay) * c.aux    # leaky trace of presyn activity
        e = c.err_local @ self.B.t()                                          # random-feedback credit assignment
        self.W += self.lr / c.n * (e.t() @ self._elig / c.batch - self.lam * self.W)
        self.b += self.lr / c.n * e.mean(0)

    def fit(self, env, budget, probe, batch=256):
        import motor_core as _core                              # lazy: avoids a circular import
        return _core.train(self, env, budget, probe, batch=batch)


# =============================================================================
# 3. BTSP  (Bittner et al. 2017, Science)  -- dendritic PLATEAU-gated one-shot
# -----------------------------------------------------------------------------
# Behavioural-timescale synaptic plasticity: a SLOW (~1 s) eligibility trace of presynaptic
# activity is bound to the instructive signal ONLY when a sparse, stochastic dendritic
# PLATEAU fires -- rare, large, one-shot updates instead of a continuous gradient. tau_slow
# is an order of magnitude longer than every other rule here, and the plateau gate is unique
# to BTSP. (The update is normalised by the number of plateaus so the one-shot writes stay
# bounded.)
# =============================================================================
class BTSP(_ResBase):
    name = "BTSP (plateau-gated one-shot · behavioural-timescale trace)"
    cite = "Bittner+17 BTSP (Science); dendritic plateau; morphological head"
    wins = "one-shot binding (a plateau writes a whole trajectory at once)"
    def __init__(self, env, teacher=None, lr=0.02, tau_slow=1.0, p_plateau=1.0, **kw):
        # audit fix: p_plateau 8->1 so ~1 plateau fires per 1-s reach (Bittner 2017 is one/few),
        # larger lr since each plateau is now a genuine one-shot biased write (not averaged out).
        super().__init__(env, teacher, **kw); self.lr, self.tau_slow, self.p_plateau = lr, tau_slow, p_plateau
    def forward(self, obs, h):
        h, z = self._feat(obs, h)
        return z @ self.W.t() + self.b, h, z

    def on_episode_start(self, B):
        self._trace = th.zeros(B, self.Nr + self.O, device=self.dev)
        self._decay = 1 - self.dt / self.tau_slow
        self._gp = self.p_plateau * self.dt / self.tau_slow

    @th.no_grad()
    def on_step(self, c):
        """Behavioural-timescale plasticity (Bittner 2017): a seconds-long presynaptic trace bound
        to the instructive signal at a sparse, stochastic dendritic PLATEAU -- a one-shot BIASED write.

        gBTSP note (researched July 2026 -- Cone 2025 gBTSP bioRxiv 2025.06.12.659336; simple-model
        one-shot BTSP, Nat Commun 2024): the normative BTSP plateau is ERROR-PROPORTIONAL,
        P_i = eps_i(t)*(kernel-filtered features) - lambda*W. But this rule ALREADY realises that
        error dependence: the write magnitude is `gate * err_local (=eps_i) outer trace`, so each
        plateau's write is already scaled by the local reflex-imitation error. Additionally gating
        the plateau PROBABILITY by error (the gBTSP suggestion) DOUBLE-COUNTS eps_i -- it over-weights
        high-error moments and destabilises the high-variance one-shot writes (empirically 0% vs the
        blind gate's 23% at the same tuned HP). So the faithful, stable choice is the blind sparse
        gate for WHEN, with err_local scaling HOW MUCH. BTSP's modest asymptotic score is then an
        honest property of a one-shot SAMPLE-EFFICIENCY rule on a fixed-episode ACCURACY benchmark,
        not a tuning miss."""
        self._trace = self._decay * self._trace + (1 - self._decay) * c.aux
        gate = (th.rand(c.batch, 1, device=self.dev) < self._gp).float()
        ge = gate * c.err_local
        self.W += self.lr / c.n * (ge.t() @ self._trace / c.batch - self.lam * self.W)
        self.b += self.lr / c.n * (ge.sum(0) / c.batch)

    def fit(self, env, budget, probe, batch=256):
        import motor_core as _core                              # lazy: avoids a circular import
        return _core.train(self, env, budget, probe, batch=batch)


# =============================================================================
# 4. R-STDP  (Izhikevich 2007, Cereb. Cortex)  -- SPIKING eligibility + dopamine
# -----------------------------------------------------------------------------
# The reservoir rates are thresholded into binary SPIKES; a per-synapse eligibility TAG is
# set by pre-spike / post coincidence (an STDP window, decaying with tau_c) and a global
# DOPAMINE signal d(t) (graded reward) multiplies the tag to consolidate it: dW = eta *
# d(t) * tag. Spikes + STDP tag + dopamine gate -> a distinct plasticity and a distinct,
# spike-based (SynOps) energy cost. The motor output still reads the rich rate features, so
# the readout stays expressive while its LEARNING is spike-timing driven.
# =============================================================================
class RSTDP(_ResBase):
    name = "R-STDP (spiking eligibility · reward-modulated STDP tag)"
    cite = "Izhikevich 07 R-STDP (Cereb.Cortex); spike-timing tag, dopamine gate; morphological head"
    wins = "energy efficiency (event-driven spikes, sparse SynOps)"
    def __init__(self, env, teacher=None, lr=0.001, tau_c=0.1, vth=0.3, **kw):
        super().__init__(env, teacher, **kw); self.lr, self.tau_c, self.vth = lr, tau_c, vth; self.baseline = 0.0
    def forward(self, obs, h):
        h, z = self._feat(obs, h)
        return z @ self.W.t() + self.b, h, z

    def on_episode_start(self, B):
        self._tag = th.zeros(self.R, self.Nr + self.O, device=self.dev)
        self._dc = 1 - self.dt / self.tau_c

    @th.no_grad()
    def on_step(self, c):
        """Reward-modulated STDP: a spike-gated eligibility TAG (post SHARED error x pre spike),
        consolidated by a DOPAMINE third factor read from the environment reward."""
        z = c.aux; spre = (z > self.vth).float()
        self._tag = self._dc * self._tag + (1 - self._dc) * (c.err_local.t() @ (z * spre) / c.batch)
        d = c.reward.mean().item(); self.baseline = 0.99 * self.baseline + 0.01 * d
        dop = max(0.2, 1.0 + 5.0 * (d - self.baseline))
        self.W += (self.lr / c.n) * dop * self._tag
        self.b += (self.lr / c.n) * dop * c.err_local.mean(0)

    def fit(self, env, budget, probe, batch=256):
        import motor_core as _core                              # lazy: avoids a circular import
        return _core.train(self, env, budget, probe, batch=batch)
    @th.no_grad()
    def _spike_rate(self, env, batch=64):
        """MEASURED fraction of units above threshold per control step (the old 0.30 was invented)."""
        obs, _ = env.reset(options={"batch_size": batch}); h = self.init_state(batch)
        tot, cnt = 0.0, 0
        for _ in range(int(env.max_ep_duration / env.dt)):
            h, z = self._feat(obs, h)
            tot += (z > self.vth).float().mean().item(); cnt += 1
            obs, *_ = env.step(HEAD(obs, z @ self.W.t() + self.b))
        return tot / max(cnt, 1)

    def ops(self, env):
        """Event-driven accounting, as this module's stated SNN convention requires (Sorbaro+20): a
        synapse costs energy only when its presynaptic unit SPIKES. The recurrent term is therefore
        SynOps over the sparse synapses that exist, scaled by the MEASURED spike rate -- not a dense
        Nr x Nr MAC block, which contradicted the convention and dominated the whole energy column."""
        c = OPCOUNTER()
        s = self._spike_rate(env)
        fan = float((self.Wr != 0).sum().item()) / max(self.Nr, 1)     # existing synapses per unit
        c.synops(s * self.Nr, fan)                                     # recurrent, spike-triggered
        c.dense(self.O, self.Nr)                                       # analog input projection
        c.synops(s * (self.Nr + self.O), self.R)                       # spike-triggered readout
        return c


# =============================================================================
# 5. Predictive coding  (Rao & Ballard 1999; Friston active inference)  -- ERROR UNITS
# -----------------------------------------------------------------------------
# A generative hierarchy with explicit ERROR UNITS. A latent representation r predicts the
# reservoir feature through Wpred; the bottom-up prediction error eps = h - Wpred r drives a
# few INFERENCE iterations that refine r; the motor command is read from [r, obs] and its own
# top-level error trains that readout, while eps trains the generative weights (dWpred ~ eps
# (x) r). Error units + top-down prediction + iterative inference make this a generative
# network, not a feedforward readout -- the most architecturally distinct rule.
# =============================================================================
class PredictiveCoding(_ResBase):
    name = "Predictive coding (hierarchical error-unit inference)"
    cite = "Rao&Ballard 99 (Nat.Neurosci.); Friston active inference; morphological head"
    wins = "robust asymptotic control (inference corrects deviations online)"
    def __init__(self, env, teacher=None, Nrep=RES_NR, lr=0.1, lr_g=0.01, n_infer=5, seed=0, **kw):
        # Nrep tracks RES_NR so the motor readout is 3*(Nrep+O)+3 = 12,327 -- the same plastic
        # budget as every sibling rule. Hard-coded 2048 silently gave it half.
        super().__init__(env, teacher, seed=seed, **kw)
        self.Nrep, self.lr, self.lr_g, self.n_infer = Nrep, lr, lr_g, n_infer
        g = th.Generator(device="cpu").manual_seed(seed + 7)
        self.register_buffer("Wenc", (th.randn(Nrep, self.Nr, generator=g) / math.sqrt(self.Nr)).to(self.dev))
        # fixed generative model (top-down prediction of the reservoir feature); UNIT-NORM columns so
        # Wpred^T Wpred ~= I and the inference iteration is contractive (else it diverges, Nr>>Nrep).
        Wp = th.randn(self.Nr, Nrep, generator=g); Wp = Wp / Wp.norm(dim=0, keepdim=True)
        self.register_buffer("Wpred", Wp.to(self.dev))
        self.W = None; self.register_buffer("Wmot", th.zeros(self.R, Nrep + self.O, device=self.dev))
    def _infer(self, h):
        r = h @ self.Wenc.t()
        for _ in range(self.n_infer):
            eps = h - r @ self.Wpred.t()                                     # bottom-up error units
            r = r + 0.2 * (eps @ self.Wpred) - 0.1 * r                       # descend prediction error (stable)
        return r        # the settled latent; on_step recomputes the reservoir prediction error from
                        # (c.state, r) to train the generative weights Wpred (Rao-Ballard, second half).
    def act(self, obs, h, explore=False):
        h = self._recur((obs - self.mu) / self.sig, h); r = self._infer(h)
        zr = th.cat([r, (obs - self.mu) / self.sig], -1)
        return HEAD(obs, zr @ self.Wmot.t() + self.b), h
    def forward(self, obs, h):
        h = self._recur((obs - self.mu) / self.sig, h); r = self._infer(h)
        zr = th.cat([r, (obs - self.mu) / self.sig], -1)
        return zr @ self.Wmot.t() + self.b, h, zr

    @th.no_grad()
    def on_step(self, c):
        """Hierarchical predictive coding (Rao & Ballard 1999), BOTH halves:
          (1) the GENERATIVE model learns -- the top-down weights Wpred are adjusted by the local
              Hebbian product of the reservoir prediction error and the settled latent
              (dWpred ~ eps^T r), so the generative model gets better at predicting its input;
          (2) the top-level motor prediction error (the SHARED signal) descends onto the motor readout.
        The latent r itself is settled by iterative error-unit inference in forward()/act()."""
        r = c.aux[:, :self.Nrep]                                # settled latent (aux = [r, x])
        eps = c.state - r @ self.Wpred.t()                      # reservoir-level prediction error units
        self.Wpred += self.lr_g / c.n * (eps.t() @ r) / c.batch  # generative-weight learning (Rao-Ballard)
        self.Wpred /= (self.Wpred.norm(dim=0, keepdim=True) + 1e-8)   # keep columns unit-norm -> inference stays contractive
        self.Wmot += self.lr / c.n * (c.err_local.t() @ c.aux / c.batch - self.lam * self.Wmot)
        self.b += self.lr / c.n * c.err_local.mean(0)

    def fit(self, env, budget, probe, batch=256):
        import motor_core as _core                              # lazy: avoids a circular import
        return _core.train(self, env, budget, probe, batch=batch)
    def ops(self, env):
        """Each inference iteration runs Wpred in BOTH directions -- predict h_hat = Wpred r
        (Nrep->Nr) and project the residual back (Nr->Nrep) -- so the forward pass costs 2x what the
        old count charged. Recurrent term is the sparse synapse count, not a dense Nr x Nr block."""
        c = OPCOUNTER()
        c.mac += float((self.Wr != 0).sum().item())                    # sparse recurrent synapses
        c.dense(self.O, self.Nr)
        c.dense(self.Nr, self.Nrep * (2 * self.n_infer + 1))           # 1 encode + 2 matvecs per iteration
        c.dense(self.Nrep + self.O, self.R)
        return c


# =============================================================================
# 6. Three-factor Hebb / neuromodulated gateway  (Kusmierz+17; Fremaux & Gerstner 16)
# -----------------------------------------------------------------------------
# Reward-modulated Hebbian with a NEUROMODULATORY GATE (the "chemical tag" / dopamine third
# factor). The local Hebbian term is a presynaptic-activity x postsynaptic-error outer product
# (dW ~ err (x) pre); the THIRD FACTOR is a global dopamine signal M = sigmoid(k*(R - Rbar)) that
# bursts when the instantaneous reward R = -MSE beats its slow baseline Rbar and is suppressed
# otherwise -- so the SAME reward that other rules ignore gates how much Hebbian plasticity is
# written. Distinct from every sibling: no eligibility trace (e-prop), no random-feedback matrix
# (RTRRL), no plateau (BTSP), no spike tag (RSTDP), no error units (predictive coding) -- its
# signature is the scalar neuromodulator gating an otherwise-local Hebbian delta. That same gate
# is what protects old motor memories (low M on a mastered task -> little overwrite) -> its
# continual-learning win. Error-direction (vs pure node perturbation) is what lets a weak local
# rule actually reach sub-5cm on the 2-D force head, where perturbation-only credit collapses.
# =============================================================================
class Hebb3(_ResBase):
    name = "3-factor Hebb (reward-gated neuromodulatory plasticity)"
    cite = "Kusmierz+17 three-factor; Fremaux&Gerstner 16 R-max; morphological head"
    wins = "continual learning (a neuromodulator gates what is written)"
    def __init__(self, env, teacher=None, lr=0.02, gain=4.0, tau_e=0.5, **kw):
        super().__init__(env, teacher, **kw); self.lr, self.gain, self.tau_e = lr, gain, tau_e
        self.register_buffer("base", th.zeros(1, device=self.dev))    # dopamine baseline (EMA reward)
    def forward(self, obs, h):
        h, z = self._feat(obs, h)
        return z @ self.W.t() + self.b, h, z

    def on_episode_start(self, B):
        self._elig = th.zeros(self.R, self.Nr + self.O, device=self.dev)   # Fremaux&Gerstner trace
        self._eb = th.zeros(self.R, device=self.dev)

    @th.no_grad()
    def on_step(self, c):
        """Three-factor rule (Fremaux & Gerstner 2016): a seconds-long ELIGIBILITY TRACE of the
        local pre x post-error product, consolidated by a dopamine neuromodulator that bursts
        when reward beats its slow baseline.

        Audit fix: the update was instantaneous; the defining Fremaux&Gerstner mechanism is the
        decaying eligibility trace (tau_e ~ 0.5 s), now added."""
        if not hasattr(self, "_elig"): self.on_episode_start(c.batch)
        R = -(c.err ** 2).mean()      # neuromodulator tracks THE shared objective itself
        M = th.sigmoid(self.gain * (R - self.base))
        self.base.mul_(0.99).add_(0.01 * R)
        decay = 1 - self.dt / self.tau_e
        self._elig = decay * self._elig + (1 - decay) * (c.err_local.t() @ c.aux / c.batch)
        self._eb = decay * self._eb + (1 - decay) * c.err_local.mean(0)
        self.W += self.lr / c.n * M * (self._elig - self.lam * self.W)
        self.b += self.lr / c.n * M * self._eb

    def fit(self, env, budget, probe, batch=256):
        import motor_core as _core                              # lazy: avoids a circular import
        return _core.train(self, env, budget, probe, batch=batch)


REGISTRY = dict(EProp=EProp, RTRRL=RTRRL, BTSP=BTSP, RSTDP=RSTDP,
                PredictiveCoding=PredictiveCoding, Hebb3=Hebb3)


if __name__ == "__main__":
    import sys, time
    sys.path.insert(0, "notebooks"); sys.path.insert(0, "MotorNet"); sys.path.insert(0, "nlb_tools")
    import motor_zoo_monkey as z
    configure(z.force_head, z.obs_norm, z.OpCounter)
    env = z.make_env(z.DEVICE); ev = z.make_env(z.DEVICE)
    B = int(sys.argv[1]) if len(sys.argv) > 1 else 40000
    TB = int(sys.argv[2]) if len(sys.argv) > 2 else 80000
    def compl(L, thr):
        r = z.rollout(ev, L, batch=512); k = max(1, int(0.2 * r["n"])); fin = r["dist"][:, -k:].mean(1)
        return 100.0 * (fin < thr / 100).float().mean().item(), 100.0 * r["dist"][:, -k:].mean().item()
    th.manual_seed(0); teacher = z.BPTTGRU(env, hidden=128); tp = z.Probe(ev, 10**9, TB); teacher.fit(env, TB, tp, batch=256)
    c2, e = compl(teacher, 2.0); c5, _ = compl(teacher, 5.0)
    print(f"teacher BPTT-GRU: {e:.2f}cm  c@2={c2:.1f}%  c@5={c5:.1f}%", flush=True)
    for nm, cls in REGISTRY.items():
        th.manual_seed(0); L = cls(env, teacher=teacher); pr = z.Probe(ev, 10**9, B)
        t0 = time.perf_counter(); L.fit(env, B, pr, batch=256); dt = time.perf_counter() - t0
        c2, e = compl(L, 2.0); c5, _ = compl(L, 5.0)
        assert L.act(env.reset(options={"batch_size": 4})[0], L.init_state(4))[0].shape == (4, z.N_MUSCLES)
        print(f"{nm:18s} {e:6.2f}cm  c@2={c2:5.1f}%  c@5={c5:5.1f}%  ({dt:.0f}s)", flush=True)
