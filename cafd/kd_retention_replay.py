"""Keep diagnostic MPC calls/RNG; actual block coordinate comes only from old logs."""
import copy
from .mpc_controller import MPCController
U_BY_BLOCK = [0.5, 1.0, 1.5, 1.5, 1.5, 2.0, 2.5, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0]

class ReplayController(MPCController):
    def choose_action(self):
        if self.pending is not None:
            return copy.deepcopy(self.pending)
        recommendation = super().choose_action()
        actual = float(U_BY_BLOCK[self.completed_full_blocks])
        old = recommendation["u_before"]
        increment = (actual-old)/self.config.delta
        if increment not in (0.,1.):
            raise RuntimeError("replay transition is not a valid original hold/advance")
        action = int(increment)
        self.u = actual
        self.pending.update(mpc_recommendation=recommendation, kind="old_schedule_replay",
            u=actual, action=action, effective_increment=increment,
            features=None if self.state is None or self.previous_state is None else
            self._features(self.state,self.previous_state,old,self.previous_increment,
                           action,self.completed_rounds).tolist())
        return copy.deepcopy(self.pending)
