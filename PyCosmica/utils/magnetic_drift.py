from PyCosmica.utils.generic_math import smooth_transition


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
