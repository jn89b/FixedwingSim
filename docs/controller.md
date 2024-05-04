# Updating Controller Architecture for PX4 
- https://docs.px4.io/main/en/advanced_config/parameter_reference.html
- https://docs.px4.io/main/en/flight_stack/controller_diagrams.html

## Important Notes on parameters
- Time constants
  - roll time constant default value : 0.4s
  - pitch time constant default value : 0.4s
- Max Rates
  - roll max setpoint: 70 deg/s 
  - pitch up max setpoint 60 deg/s
  - pitch down max setpoint 60 deg/s
  - yaw rate max setpoint 50 deg/s

## How does the controller run
- In PX4 check out the FixedWingAttitudeControl.cpp Run() method 
- This mechanism computes the set attitudes and spits out/publishes the rates of the controller
```cpp
/* Run attitude controllers */
if (_vcontrol_mode.flag_control_attitude_enabled && _in_fw_or_transition_wo_tailsitter_transition) {
    if (PX4_ISFINITE(_att_sp.roll_body) && PX4_ISFINITE(_att_sp.pitch_body)) {
        _roll_ctrl.control_roll(_att_sp.roll_body, _yaw_ctrl.get_euler_rate_setpoint(), euler_angles.phi(),
                    euler_angles.theta());
        _pitch_ctrl.control_pitch(_att_sp.pitch_body, _yaw_ctrl.get_euler_rate_setpoint(), euler_angles.phi(),
                        euler_angles.theta());
        _yaw_ctrl.control_yaw(_att_sp.roll_body, _pitch_ctrl.get_euler_rate_setpoint(), euler_angles.phi(),
                        euler_angles.theta(), get_airspeed_constrained());

        if (wheel_control) {
            _wheel_ctrl.control_attitude(_att_sp.yaw_body, euler_angles.psi());

        } else {
            _wheel_ctrl.reset_integrator();
        }

        /* Update input data for rate controllers */
        Vector3f body_rates_setpoint = Vector3f(_roll_ctrl.get_body_rate_setpoint(), _pitch_ctrl.get_body_rate_setpoint(),
                            _yaw_ctrl.get_body_rate_setpoint());

        autotune_attitude_control_status_s pid_autotune;
        matrix::Vector3f bodyrate_autotune_ff;

        if (_autotune_attitude_control_status_sub.copy(&pid_autotune)) {
            if ((pid_autotune.state == autotune_attitude_control_status_s::STATE_ROLL
                    || pid_autotune.state == autotune_attitude_control_status_s::STATE_PITCH
                    || pid_autotune.state == autotune_attitude_control_status_s::STATE_YAW
                    || pid_autotune.state == autotune_attitude_control_status_s::STATE_TEST)
                && ((hrt_absolute_time() - pid_autotune.timestamp) < 1_s)) {

                bodyrate_autotune_ff = matrix::Vector3f(pid_autotune.rate_sp);
                body_rates_setpoint += bodyrate_autotune_ff;
            }
        }

        /* add yaw rate setpoint from sticks in all attitude-controlled modes */
        if (_vcontrol_mode.flag_control_manual_enabled) {
            body_rates_setpoint(2) += math::constrain(_manual_control_setpoint.yaw * radians(_param_man_yr_max.get()),
                            -radians(_param_fw_y_rmax.get()), radians(_param_fw_y_rmax.get()));
        }

        // Tailsitter: transform from FW to hover frame (all interfaces are in hover (body) frame)
        if (_vehicle_status.is_vtol_tailsitter) {
            body_rates_setpoint = Vector3f(body_rates_setpoint(2), body_rates_setpoint(1), -body_rates_setpoint(0));
        }

        /* Publish the rate setpoint for analysis once available */
        _rates_sp.roll = body_rates_setpoint(0);
        _rates_sp.pitch = body_rates_setpoint(1);
        _rates_sp.yaw = body_rates_setpoint(2);

        _rates_sp.timestamp = hrt_absolute_time();

        _rate_sp_pub.publish(_rates_sp);
    }
}
```

- After that is done check out the FixedwingRateControl.hpp and the vehicle manual poll
```cpp

``` 
