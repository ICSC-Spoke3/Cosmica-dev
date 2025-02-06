from jax import Array, lax, numpy as jnp
from jax.typing import ArrayLike

from PyCosmica.utils.generic_math import smooth_transition

from PyCosmica.structures import delta_m, rhelio, omega, omega, rhelio, \
    high_rigi_suppression_smoothness, high_rigi_suppression_trans_point, \
    PropagationState, PropagationConstantsItem, Position3D
from PyCosmica.utils import solar_wind_speeed


def eval_p0_drift_suppression_factor(regular: bool, solar_phase: int, tilt_angle_deg: float, ssn: float) -> float:
    if regular:  # Regular drift
        initial_val = 0.5
        final_val = 4.0
        if solar_phase == 0:
            center_of_transition = 73.0
            smoothness = 1.0
        else:
            center_of_transition = 65.0
            smoothness = 5.0
    else:  # NS drift
        initial_val = 0.5
        final_val = ssn / 50.
        if solar_phase == 0:
            center_of_transition = 68.0
            smoothness = 1.0
        else:
            center_of_transition = 57.0
            smoothness = 5.0

    return smooth_transition(initial_val, final_val, center_of_transition, smoothness, tilt_angle_deg)


def eval_high_rigidity_drift_suppression_plateau(solar_phase: int, tilt_angle_deg: float) -> float:
    # Plateau time dependence
    if solar_phase == 0:
        center_of_transition = 35.0
        smoothness = 5.0
    else:
        center_of_transition = 40.0
        smoothness = 5.0

    return 1.0 - smooth_transition(1.0, 0.0, center_of_transition, smoothness, tilt_angle_deg)



# ----------------------------------------------------------------
#  Drift model functions
# ----------------------------------------------------------------
def eval_high_rigi_supp(state: PropagationState, const: PropagationConstantsItem) -> Array:
    return const.LIM.plateau + (1.0 - const.LIM.plateau) / (
        1.0 + jnp.exp(high_rigi_suppression_smoothness * (state.R - high_rigi_suppression_trans_point))
    )


def eval_E_drift(state: PropagationState, const: PropagationConstantsItem, 
                    is_polar_region: ArrayLike, V_sw: ArrayLike) -> Array:
    def is_polar():
        return (delta_m ** 2 * state.r ** 2 * V_sw ** 2 +
                omega ** 2 * rhelio ** 2 * (state.r - rhelio) ** 2 * jnp.sin(state.th) ** 4 +
                rhelio ** 2 * V_sw ** 2 * jnp.sin(state.th) ** 2)
    
    def is_not_polar():
        return omega ** 2 * (state.r - rhelio) ** 2 * jnp.sin(state.th) ** 2 + V_sw ** 2
    
    return lax.cond(is_polar_region, is_polar, is_not_polar)

def eval_C_drift_reg(state: PropagationState, const: PropagationConstantsItem, 
                        is_polar_region: ArrayLike, A_sun: ArrayLike, Ka: ArrayLike, fth: ArrayLike, E: ArrayLike) -> Array:
    
    def compute_C():
        def is_polar():
            return fth * jnp.sin(state.th) * Ka * state.r * rhelio / (A_sun * E**2)
        
        def is_not_polar():
            return omega * fth * Ka * state.r / (A_sun * E**2)
        
        return lax.cond(is_polar_region, is_polar, is_not_polar)
    
    def compute_reduction():
        return state.R**2 / (state.R**2 + const.LIM.P0_d**2)
        
    return compute_C() * compute_reduction()


def eval_C_drift_ns(state: PropagationState, const: PropagationConstantsItem, 
                        is_polar_region: ArrayLike, A_sun: ArrayLike, Ka: ArrayLike, 
                        fth: ArrayLike, E: ArrayLike, Dftheta_dtheta:ArrayLike, V_sw: ArrayLike) -> Array:
    def compute_C():
        def is_polar():
            return V_sw * Dftheta_dtheta * jnp.sin(state.th)**2 * Ka * state.r * rhelio**2 / (A_sun * E)
        
        def is_not_polar():
            return V_sw * Dftheta_dtheta * Ka * state.r / (A_sun * E)
        
        return lax.cond(is_polar_region, is_polar, is_not_polar)

    def compute_reduction():
        return state.R**2 / (state.R**2 + const.LIM.P0_dNS**2)

    return compute_C() * compute_reduction()


def drift_pm89(state: PropagationState, const: PropagationConstantsItem,
                is_polar_region: ArrayLike, A_sun: ArrayLike, Ka: ArrayLike, fth: ArrayLike, Dftheta_dtheta: ArrayLike,
                V_sw: ArrayLike, dV_dth: ArrayLike, high_rigi_supp: ArrayLike) -> Array:
    
    
    def apply_high_rigi_supp(r_, th_, phi_):
        return r_ * high_rigi_supp, th_ * high_rigi_supp, phi_ * high_rigi_supp

    def is_polar():
        # Polar region
        E = eval_E_drift(state, const, 1., V_sw)
        C = eval_C_drift_reg(state, const, 1., A_sun, Ka, fth, E)
        # Regular drift contribution
        v_r = - C * omega * rhelio * 2.0 * (state.r - rhelio) * jnp.sin(state.th) * (
                (2.0 * (delta_m**2) * state.r**2 + rhelio**2 * jnp.sin(state.th)**2) * V_sw**3 * jnp.cos(state.th)
                - 0.5 * (delta_m**2 * state.r**2 * V_sw**2
                         - omega**2 * rhelio**2 * (state.r - rhelio)**2 * jnp.sin(state.th)**4
                         + rhelio**2 * V_sw**2 * jnp.sin(state.th)**2)
                  * jnp.sin(state.th) * dV_dth )
        v_th = - C * omega * rhelio * V_sw * jnp.sin(state.th)**2 * (
                2.0 * state.r * (state.r - rhelio) * (delta_m**2 * state.r * V_sw**2 + omega**2 * rhelio**2 * (state.r - rhelio) * jnp.sin(state.th)**4)
                - (4.0 * state.r - 3.0 * rhelio) * E )
        v_phi = 2.0 * C * V_sw * (
                - (delta_m**2) * state.r**2 * (delta_m * state.r + rhelio * jnp.cos(state.th)) * V_sw**3
                + 2.0 * delta_m * state.r * E * V_sw
                - omega**2 * rhelio**2 * (state.r - rhelio) * jnp.sin(state.th)**4 * (
                    delta_m * state.r**2 * V_sw - rhelio * (state.r - rhelio) * V_sw * jnp.cos(state.th)
                    + rhelio * (state.r - rhelio) * jnp.sin(state.th) * dV_dth ) )

        # ns contribution
        C = eval_C_drift_ns(state, const, 1., A_sun, Ka, fth, E, Dftheta_dtheta, V_sw)
        v_r += - C * omega * jnp.sin(state.th) * (state.r - rhelio)
        v_th = - C * V_sw * jnp.sin(state.th) * (
                  2.0 * omega**2 * state.r * (state.r - rhelio)**2 * jnp.sin(state.th)**2
                  - (4.0 * state.r - 3.0 * rhelio) * E )
        v_phi += - C * V_sw

        return Position3D(*apply_high_rigi_supp(v_r, v_th, v_phi))
    
    def is_not_polar():
        # Not Polar region (assume B_th = 0)
        E = eval_E_drift(state, const, 0., V_sw)
        # Regular drift contribution
        C = eval_C_drift_reg(state, const, 1., A_sun, Ka, fth, E)
        v_r = - 2.0 * C * (state.r - rhelio) * (
                0.5 * (omega**2 * (state.r - rhelio)**2 * jnp.sin(state.th)**2 - V_sw**2) * jnp.sin(state.th) * dV_dth
                + V_sw**3 * jnp.cos(state.th) )
        v_phi = 2.0 * C * V_sw * omega * (state.r - rhelio)**2 * jnp.sin(state.th) * (
                    V_sw * jnp.cos(state.th) - jnp.sin(state.th) * dV_dth )
        
        # ns contribution
        C = eval_C_drift_ns(state, const, 1., A_sun, Ka, fth, E, Dftheta_dtheta, V_sw)
        v_r += - C * omega * (state.r - rhelio) * jnp.sin(state.th)
        v_phi += - C * V_sw
        v_th = 0.0 

        return Position3D(*apply_high_rigi_supp(v_r, v_th, v_phi)) 

    return lax.cond(is_polar_region, is_polar, is_not_polar)


def energy_loss(state: PropagationState, const: PropagationConstantsItem):
    def in_heliosphere():
        return 2. / 3. * solar_wind_speeed(state, const) / state.r * state.R
    
    def out_heliosphere():
        return 0.0

    return lax.cond(state.rad_zone < const.N_regions, in_heliosphere, out_heliosphere)
    
