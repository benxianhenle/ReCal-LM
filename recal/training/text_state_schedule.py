"""Token-based ramps and validation-gated fixed-point regularization."""
from dataclasses import dataclass, field

@dataclass
class TextStateSchedule:
    start_tokens: int
    alpha_scale: float = 1.
    follow_scale: float = 1.
    r_start: int | None = None
    previous_ce: float | None = None
    worsening: int = 0
    clean_validations: int = 0
    baseline: dict = field(default_factory=dict)
    alerts: list = field(default_factory=list)
    stop: bool = False

    def coefficients(self, tokens):
        elapsed=max(0,tokens-self.start_tokens)
        alpha=.1*min(elapsed/50_000_000,1.)*self.alpha_scale
        a=.05*min(max(elapsed-50_000_000,0)/100_000_000,1.)*self.follow_scale
        r=0. if self.r_start is None else .01*min(max(tokens-self.r_start,0)/100_000_000,1.)
        r=min(r,a*.2)
        phase=1 if elapsed<50_000_000 else 2 if elapsed<150_000_000 else 3 if self.r_start is not None else '3_waiting_validation'
        return dict(alpha=alpha,lambda_a=a,lambda_r=r),phase

    def observe(self, m, tokens):
        ce=m['ce_depth_3']
        if not self.baseline:
            self.baseline=dict(m);self.previous_ce=ce
            return
        self.worsening=self.worsening+1 if ce>self.previous_ce*1.002 else 0
        alerts=[]
        for k in range(1,4):
            for metric,ratio in [('state_norm',.7),('token_variance',.5),('delta_norm',.3)]:
                key=f'{metric}_depth_{k}'
                if m[key]<self.baseline[key]*ratio: alerts.append(f'{key}_fell')
        if m['ce_depth_3']>m['ce_depth_1']*1.02: alerts.append('deeper_ce_worse')
        if m['cos_depth_3']>.995 and ce>=self.previous_ce: alerts.append('alignment_without_text_gain')
        if self.worsening>=3: alerts.append('validation_worsening')
        self.alerts=alerts
        self.clean_validations=0 if alerts else self.clean_validations+1
        if alerts:
            self.r_start=None
            if tokens>self.start_tokens+50_000_000:
                self.follow_scale=max(self.follow_scale*.5,.125)
            if 'validation_worsening' in alerts:self.alpha_scale=max(self.alpha_scale*.5,.125)
        if self.worsening>=5:self.stop=True
        if (tokens>=max(200_000_000,self.start_tokens+150_000_000)
            and self.clean_validations>=3 and self.r_start is None
            and ce<=self.baseline['ce_depth_3'] and m['cos_depth_3']>self.baseline['cos_depth_3']):
            self.r_start=tokens
        self.previous_ce=ce
