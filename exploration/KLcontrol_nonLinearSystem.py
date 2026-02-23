

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
from concurrent.futures import ProcessPoolExecutor
import os
from tqdm import tqdm

TIMESTEP = 0.05  # τ in paper: Euler discretization time step (seconds)
HORIZON = 40     # N in paper: finite horizon length (number of time steps)
TOTAL_SIMULATION_TIME = 150

PENDULUM_MASS = 0.1  # m in paper: mass at end of pole (kg)
ROD_LENGTH = 1.0     # L in paper: length of massless rod (m)
CART_MASS = 1.0      # M in paper: mass of cart (kg)
GRAVITY = 9.8        # g in paper: gravitational acceleration (m/s²)

NOISE_STDEV = 5.0    # σ in paper: standard deviation of noise distribution (N)

NUM_WORKERS = os.cpu_count()  # parallel workers for policy computation


# =============================================================================
# SECTION 3: COST FUNCTION WEIGHTS
# Paper Section 6.2, cost function ℓ_k(x_k)
# =============================================================================

# State cost: ℓ(x) = q1|x̄| + q2|x̄̇| + q3|θ| + q4|θ̇|
# Penalizes deviations from upright equilibrium at origin
# Control cost: r(u) = R·|u|, penalizes large control inputs

Q_CART_POSITION = 7.0      # q1 in paper: penalty on cart position (m⁻¹)
Q_CART_VELOCITY = 2.5      # q2 in paper: penalty on cart velocity (s/m)
Q_POLE_ANGLE = 7.0         # q3 in paper: penalty on pole angle from vertical (rad⁻¹)
Q_POLE_ANGULAR_VEL = 2.5   # q4 in paper: penalty on pole angular velocity (s/rad)
R_CONTROL = 0.1            # r in paper extension: penalty on control input magnitude (N⁻¹)


# =============================================================================
# SECTION 4: DISCRETE INPUT SPACE
# Paper Section 6.2: U = {2i N}_{i=-10}^{10}
# =============================================================================

# Discrete input space U (both control inputs and noise samples come from U)
# This is Assumption 3.2-(i): W ⊆ U (noise support contained in control space)
INPUT_SPACE = np.arange(-20, 22, 2)  # U = {-20, -18, ..., 18, 20} Newtons

# =============================================================================
# SECTION 5: INITIAL CONDITION
# Paper Section 6.2
# =============================================================================

# State vector: x = [x̄, x̄̇, θ, θ̇]ᵀ where:
#   x̄  : cart position (m)
#   x̄̇  : cart velocity (m/s)
#   θ  : pole angle from vertical, θ=0 is upright (rad)
#   θ̇  : pole angular velocity (rad/s)

INITIAL_STATE = np.array([
    2.0,  # x̄₀  : cart starts 2m from origin
    0.0,  # x̄̇₀  : cart initially at rest
    0.5,  # θ₀  : pole starts at 0.5 rad (~29°) from vertical
    0.0   # θ̇₀  : pole initially not rotating
])



# =============================================================================
# SECTION 6: SYSTEM DYNAMICS - CONTINUOUS-TIME MODEL
# Paper Section 6.2, Equations (48)-(49)
# =============================================================================

def compute_cart_acceleration(pole_angle, pole_angular_velocity, force):
    """
    Compute cart acceleration: ẍ̄ = h₁(θ, θ̇, u)
    
    From paper Equation (48):
    ẍ̄ = [-mL(θ̇)² sin θ + mg sin θ cos θ + u] / [M + m sin²θ]
    
    This comes from the Euler-Lagrange equations for the cart-pole system.
    
    Args:
        pole_angle: θ, angle of pole from vertical (rad)
        pole_angular_velocity: θ̇, angular velocity of pole (rad/s)
        force: u, horizontal force applied to cart (N)
    
    Returns:
        cart_acceleration: ẍ̄ (m/s²)
    
    Physical interpretation:
    - Centrifugal force: -mL(θ̇)² sin θ (pushes cart when pole swings)
    - Gravity component: mg sin θ cos θ (gravitational torque effect on cart)
    - Applied force: u (control input)
    - Effective mass: M + m sin²θ (varies with pole angle)
    """
    numerator = (
        -PENDULUM_MASS * ROD_LENGTH * pole_angular_velocity**2 * np.sin(pole_angle) +
        PENDULUM_MASS * GRAVITY * np.sin(pole_angle) * np.cos(pole_angle) +
        force
    )
    denominator = CART_MASS + PENDULUM_MASS * np.sin(pole_angle)**2
    
    return numerator / denominator


def compute_pole_angular_acceleration(pole_angle, pole_angular_velocity, force):
    """
    Compute pole angular acceleration: θ̈ = h₂(θ, θ̇, u)
    
    From paper Equation (49):
    θ̈ = (1/L)[h₁(θ, θ̇, u) cos θ + g sin θ]
    
    This is the rotational equation of motion for the pole.
    
    Args:
        pole_angle: θ, angle of pole from vertical (rad)
        pole_angular_velocity: θ̇, angular velocity of pole (rad/s)
        force: u, horizontal force applied to cart (N)
    
    Returns:
        pole_angular_acceleration: θ̈ (rad/s²)
    
    Physical interpretation:
    - Cart acceleration couples to pole: h₁ cos θ (cart motion affects pole)
    - Gravitational torque: g sin θ (gravity tries to pull pole down)
    - Rod length: 1/L (longer rod → smaller angular acceleration)
    """
    cart_accel = compute_cart_acceleration(pole_angle, pole_angular_velocity, force)
    
    angular_acceleration = (
        (1.0 / ROD_LENGTH) * (cart_accel * np.cos(pole_angle) + 
                              GRAVITY * np.sin(pole_angle))
    )
    
    return angular_acceleration


# =============================================================================
# SECTION 7: DISCRETE-TIME DYNAMICS
# Paper Section 6.2, Equation (50) - Euler discretization
# This implements f(x_k, u_k) from the paper (Assumption 3.2)
# =============================================================================

def dynamics_step(state, control_or_noise):
    """
    Discrete-time dynamics: x_{k+1} = f(x_k, u_k)
    
    From paper Equation (50), using forward Euler method with timestep τ:
    
    x̄_{k+1}  = x̄_k  + τ·x̄̇_k
    x̄̇_{k+1}  = x̄̇_k  + τ·h₁(θ_k, θ̇_k, u_k)
    θ_{k+1}  = θ_k  + τ·θ̇_k
    θ̇_{k+1}  = θ̇_k  + τ·h₂(θ_k, θ̇_k, u_k)
    
    This is the function f(x,u) that appears throughout the paper.
    
    Args:
        state: x_k = [x̄, x̄̇, θ, θ̇]ᵀ, current state (4D vector)
        control_or_noise: u_k, can be either:
                         - Control input u (for controlled dynamics)
                         - Noise sample w (for noise-driven dynamics x̄)
    
    Returns:
        next_state: x_{k+1}, state at next timestep (4D vector)
    
    Paper connection:
    - Equation (5): x_{k+1} = f(x_k, u_k) for controlled system
    - Equation (7): x̄_{k+1} = f(x̄_k, w_k) for noise-driven system
    """
    # Unpack current state
    cart_pos, cart_vel, pole_angle, pole_angular_vel = state
    
    # Compute accelerations using continuous-time dynamics
    cart_accel = compute_cart_acceleration(
        pole_angle, pole_angular_vel, control_or_noise
    )
    pole_angular_accel = compute_pole_angular_acceleration(
        pole_angle, pole_angular_vel, control_or_noise
    )
    
    # Forward Euler integration
    next_state = np.zeros(4)
    next_state[0] = cart_pos + TIMESTEP * cart_vel
    next_state[1] = cart_vel + TIMESTEP * cart_accel
    next_state[2] = pole_angle + TIMESTEP * pole_angular_vel
    next_state[3] = pole_angular_vel + TIMESTEP * pole_angular_accel
    
    return next_state


# =============================================================================
# SECTION 8: COST FUNCTION
# Paper Section 6.2: ℓ_k(x_k) = q₁|x̄| + q₂|x̄̇| + q₃|θ| + q₄|θ̇|
# This is ℓ_k(x) from Problem 3.1
# =============================================================================

def stage_cost(state):
    """
    Stage cost function: ℓ(x) = q₁|x̄| + q₂|x̄̇| + q₃|θ| + q₄|θ̇|
    
    From paper Section 6.2. This is the running cost that penalizes:
    - Being away from origin (cart position)
    - Moving fast (cart velocity)
    - Being tilted (pole angle from vertical)
    - Rotating fast (pole angular velocity)
    
    The absolute values create a Manhattan distance / L1 norm cost.
    Goal: Regulate system to upright equilibrium at origin.
    
    Args:
        state: x = [x̄, x̄̇, θ, θ̇]ᵀ, current state
    
    Returns:
        cost: ℓ(x), scalar cost value (dimensionless)
    
    Paper connection:
    - This is ℓ_k(x_k) in the objective function (Equation 9)
    - Terminal cost ℓ_N is the same function evaluated at final time
    """
    cart_pos, cart_vel, pole_angle, pole_angular_vel = state
    
    cost = (
        Q_CART_POSITION * np.abs(cart_pos) +
        Q_CART_VELOCITY * np.abs(cart_vel) +
        Q_POLE_ANGLE * np.abs(pole_angle) +
        Q_POLE_ANGULAR_VEL * np.abs(pole_angular_vel)
    )
    
    return cost


# =============================================================================
# SECTION 8b: VECTORIZED (BATCH) DYNAMICS AND COST
# Same math as above, but operates on (batch, 4) state arrays for speed.
# Eliminates Python loops over Monte Carlo samples.
# =============================================================================

def dynamics_step_batch(states, forces):
    """
    Vectorized dynamics: advance a batch of states in one numpy call.

    Args:
        states: (batch, 4) array — each row is [x̄, x̄̇, θ, θ̇]
        forces: (batch,) array — force applied to each state

    Returns:
        next_states: (batch, 4) array
    """
    cart_pos = states[:, 0]
    cart_vel = states[:, 1]
    pole_angle = states[:, 2]
    pole_ang_vel = states[:, 3]

    # Cart acceleration (Eq. 48)
    num = (
        -PENDULUM_MASS * ROD_LENGTH * pole_ang_vel**2 * np.sin(pole_angle)
        + PENDULUM_MASS * GRAVITY * np.sin(pole_angle) * np.cos(pole_angle)
        + forces
    )
    den = CART_MASS + PENDULUM_MASS * np.sin(pole_angle)**2
    cart_accel = num / den

    # Pole angular acceleration (Eq. 49)
    pole_ang_accel = (1.0 / ROD_LENGTH) * (
        cart_accel * np.cos(pole_angle) + GRAVITY * np.sin(pole_angle)
    )

    # Forward Euler
    next_states = np.empty_like(states)
    next_states[:, 0] = cart_pos + TIMESTEP * cart_vel
    next_states[:, 1] = cart_vel + TIMESTEP * cart_accel
    next_states[:, 2] = pole_angle + TIMESTEP * pole_ang_vel
    next_states[:, 3] = pole_ang_vel + TIMESTEP * pole_ang_accel

    return next_states


def stage_cost_batch(states):
    """Vectorized stage cost over (batch, 4) states. Returns (batch,) costs."""
    return (
        Q_CART_POSITION * np.abs(states[:, 0])
        + Q_CART_VELOCITY * np.abs(states[:, 1])
        + Q_POLE_ANGLE * np.abs(states[:, 2])
        + Q_POLE_ANGULAR_VEL * np.abs(states[:, 3])
    )


# Precompute noise probability vector (used by vectorized sampling)
_UNNORM_NOISE_PROBS = np.exp(-INPUT_SPACE**2 / (2 * NOISE_STDEV**2))
_NOISE_PROBS = _UNNORM_NOISE_PROBS / np.sum(_UNNORM_NOISE_PROBS)


# =============================================================================
# SECTION 9: NOISE SAMPLING
# Paper Section 6.2, Equation (51): P(w_k = w) ∝ exp(-w²/2σ²)
# This implements ρ_w_k in the paper
# =============================================================================

def sample_noise_from_discretized_gaussian():
    """
    Sample noise w_k from discretized Gaussian distribution over U.
    
    From paper Equation (51):
    P(w_k = w) ∝ exp(-1/(2σ²) w²), for w ∈ W = U
    
    This is a discrete approximation to a continuous Gaussian distribution.
    The noise w_k drives the reference dynamics x̄_{k+1} = f(x̄_k, w_k).
    
    Returns:
        noise_sample: w_k ∈ U, sampled from discretized Gaussian
    
    Paper connection:
    - This implements ρ_w_k from Equation (7)
    - Assumption 3.2-(i) requires W ⊆ U (noise space ⊆ control space)
    - The noise distribution ρ_w_k appears in the optimal policy (Equation 13)
    
    Implementation notes:
    - Unnormalized probabilities: p(w) ∝ exp(-w²/2σ²)
    - Normalize: p(w) = exp(-w²/2σ²) / Σ_w' exp(-w'²/2σ²)
    - Sample from discrete distribution using np.random.choice
    """
    # Compute unnormalized probabilities for each element in U
    # Gaussian: exp(-w²/2σ²) centered at 0 with variance σ²
    unnormalized_probs = np.exp(-INPUT_SPACE**2 / (2 * NOISE_STDEV**2))
    
    # Normalize to create valid probability distribution (sum = 1)
    noise_probabilities = unnormalized_probs / np.sum(unnormalized_probs)
    
    # Sample from discrete distribution
    noise_sample = np.random.choice(INPUT_SPACE, p=noise_probabilities)
    
    return noise_sample


# =============================================================================
# SECTION 10: TRAJECTORY SIMULATION
# Paper Equation (7): x̄_{k+1} = f(x̄_k, w_k)
# Used in Monte Carlo estimation of desirability function
# =============================================================================

def simulate_noise_driven_trajectory(initial_state, start_time, horizon):
    """
    Simulate one noise-driven trajectory from start_time to horizon.
    
    From paper Equation (7)-(8):
    x̄_{k+1} = f(x̄_k, w_k), where w_k ~ ρ_w_k
    x̄_0 ~ ρ_x0
    
    This generates a sample path of the system under ONLY noise (no control).
    The trajectory is used to estimate the desirability function Z via
    Monte Carlo sampling (Corollary 4.2).
    
    Args:
        initial_state: x̄_{start_time}, starting state for trajectory
        start_time: k, time index to start simulation
        horizon: N, final time index (exclusive endpoint)
    
    Returns:
        total_accumulated_cost: Σ_{s=start_time}^{horizon} ℓ_s(x̄_s)
    
    Paper connection:
    - Used in Corollary 4.2 path integral representation
    - The expectation over these trajectories gives Z(k,x)
    - No control is applied; only noise w_k drives the system
    
    Implementation details:
    - Accumulates cost from start_time to horizon (inclusive of both endpoints)
    - Uses noise samples w_k ~ ρ_w_k at each timestep
    - Returns scalar total cost (not the trajectory itself)
    """
    # Initialize trajectory at given state
    current_state = initial_state.copy()
    
    # Start accumulating cost from initial state
    total_accumulated_cost = stage_cost(current_state)
    
    # Simulate forward from start_time to horizon-1
    # Range is [start_time, horizon) which gives us horizon - start_time steps
    for time_index in range(start_time, horizon):
        # Sample noise from discretized Gaussian
        noise_sample = sample_noise_from_discretized_gaussian()
        
        # Apply noise-driven dynamics: x̄_{k+1} = f(x̄_k, w_k)
        current_state = dynamics_step(current_state, noise_sample)
        
        # Accumulate cost: add ℓ(x̄_{k+1})
        total_accumulated_cost += stage_cost(current_state)
    
    return total_accumulated_cost


# =============================================================================
# SECTION 11: DESIRABILITY FUNCTION (MONTE CARLO ESTIMATION)
# Paper Theorem 4.1 and Corollary 4.2, Equation (22)
# Core of the KL control algorithm
# =============================================================================

def compute_desirability_function(time_index, state, horizon, num_samples=5000):
    """
    Compute log-desirability log Z(k,x) via VECTORIZED Monte Carlo.

    All num_samples trajectories are simulated simultaneously using
    batched numpy operations — no Python loop over samples.

    From paper Corollary 4.2, Equation (22):
    Z(k,x) = E[exp(-Σ_{s=k}^N ℓ_s(x̄_s)) | x̄_k = x]

    Returns log Z(k,x) for numerical stability (log-sum-exp trick).
    """
    # Initialize all sample trajectories at the same state: (num_samples, 4)
    states = np.tile(state, (num_samples, 1))

    # Cost from initial state
    total_costs = stage_cost_batch(states)

    # Simulate all trajectories forward in parallel
    for _ in range(time_index, horizon):
        # Sample noise for ALL trajectories at once
        noise_samples = np.random.choice(
            INPUT_SPACE, size=num_samples, p=_NOISE_PROBS
        )
        states = dynamics_step_batch(states, noise_samples)
        total_costs += stage_cost_batch(states)

    # Log-sum-exp trick: log Z = log(mean(exp(-C)))
    neg_costs = -total_costs
    max_neg = np.max(neg_costs)
    log_desirability = (
        max_neg
        + np.log(np.sum(np.exp(neg_costs - max_neg)))
        - np.log(num_samples)
    )

    return log_desirability


# =============================================================================
# SECTION 12: OPTIMAL POLICY COMPUTATION
# Paper Theorem 4.1, Equation (13)
# Uses ProcessPoolExecutor to evaluate all 21 controls in parallel.
# =============================================================================

def _evaluate_control(args):
    """
    Worker function for parallel policy computation.
    Must be top-level (not nested) so it can be pickled by multiprocessing.
    """
    control_value, time_index, state, horizon, num_samples = args
    next_state = dynamics_step(state, control_value)
    log_Z = compute_desirability_function(
        time_index=time_index + 1,
        state=next_state,
        horizon=horizon,
        num_samples=num_samples,
    )
    return log_Z


def compute_optimal_policy(time_index, state, horizon, num_samples_per_control=1000):
    """
    Compute optimal stochastic policy π*_k(u|x) for KL control.

    From paper Theorem 4.1, Equation (13):
    π*_k(u|x) = [ρ_w_k(u) · Z(k+1, f(x,u))] / [A_ρ̄_{k+1}[Z](k,x)]

    The 21 control evaluations are dispatched to a process pool so they
    run across all CPU cores simultaneously.
    """
    # Log noise probabilities (unnormalized — normalization cancels)
    log_noise_probs = -INPUT_SPACE**2 / (2 * NOISE_STDEV**2)

    # Build argument tuples for each control u ∈ U
    args_list = [
        (u, time_index, state, horizon, num_samples_per_control)
        for u in INPUT_SPACE
    ]

    # Evaluate all controls in parallel across CPU cores
    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as pool:
        log_Z_values = np.array(list(pool.map(_evaluate_control, args_list)))

    # log[ρ(u) · exp(-r(u)) · Z] = log ρ(u) - r(u) + log Z
    control_costs = R_CONTROL * np.abs(INPUT_SPACE)
    log_weights = log_noise_probs - control_costs + log_Z_values

    # Log-desirability of current state: log A_ρ̄[Z](k,x)
    # This is log Σ_u [ρ(u) · exp(-r(u)) · Z(k+1, f(x,u))]
    max_lw = np.max(log_weights)
    log_desirability = max_lw + np.log(np.sum(np.exp(log_weights - max_lw)))

    # Softmax → probability distribution π*_k(u|x)
    log_weights -= max_lw
    policy_distribution = np.exp(log_weights) / np.sum(np.exp(log_weights))

    return policy_distribution, log_desirability


def sample_control_from_policy(policy_distribution):
    """
    Sample a control action from the stochastic policy distribution.
    
    Args:
        policy_distribution: π*_k(u|x), probability for each u ∈ U
    
    Returns:
        control_sample: u_k ∈ U, sampled according to π*_k(·|x)
    
    Paper connection:
    - The optimal control is STOCHASTIC (not deterministic)
    - This reflects the KL divergence penalty in the cost function
    - Stochasticity provides exploration and robustness
    """
    control_sample = np.random.choice(INPUT_SPACE, p=policy_distribution)
    return control_sample







def simulation():
    """
    Run the full KL control simulation and record state/control/policy history.

    Returns:
        state_history: (TOTAL_SIMULATION_TIME+1, 4) array of states
        control_history: (TOTAL_SIMULATION_TIME,) array of applied controls
        policy_history: (TOTAL_SIMULATION_TIME, |U|) array of policy distributions
        log_desirability_history: (TOTAL_SIMULATION_TIME,) array of log Z values
    """
    state = INITIAL_STATE.copy()
    state_history = [state.copy()]
    control_history = []
    policy_history = []
    log_desirability_history = []

    pbar = tqdm(range(TOTAL_SIMULATION_TIME), desc="KL control", unit="step")
    for i in pbar:
        policy, log_desirability = compute_optimal_policy(
            time_index=0, state=state, horizon=HORIZON
        )
        control_input = sample_control_from_policy(policy_distribution=policy)

        state = dynamics_step(state, control_or_noise=control_input)

        state_history.append(state.copy())
        control_history.append(control_input)
        policy_history.append(policy.copy())
        log_desirability_history.append(log_desirability)

        pbar.set_postfix_str(
            f"x={state[0]:+.2f} θ={state[2]:+.2f} u={control_input:+.0f}"
        )

    return (
        np.array(state_history),
        np.array(control_history),
        np.array(policy_history),
        np.array(log_desirability_history),
    )


# =============================================================================
# SECTION 13: DIAGNOSTIC METRICS
# Paper connections noted below for each metric.
# =============================================================================

def plot_metrics(state_history, control_history, policy_history,
                 log_desirability_history):
    """
    Plot five diagnostic metrics that demonstrate KL control is working.

    Panels:
    1. Cumulative stage cost — should flatten as the controller stabilizes.
    2. Policy entropy H(π*_k) — starts high (uncertain), decreases near
       equilibrium as the policy becomes more confident.
    3. KL(π*_k || ρ_w) — how much the optimal policy deviates from the
       uncontrolled noise prior.  High when active control is needed,
       decreases near equilibrium.
    4. State norm ||x||₂ — should converge toward 0.
    5. Log-desirability log Z(k,x) — increases as the state improves
       (more "desirable" futures from better states).

    Args:
        state_history:             (T+1, 4) state trajectory
        control_history:           (T,)     applied forces
        policy_history:            (T, |U|) optimal policy distributions
        log_desirability_history:  (T,)     log Z at each step
    """
    num_steps = len(control_history)
    time_vec = np.arange(num_steps) * TIMESTEP

    fig, axes = plt.subplots(3, 2, figsize=(14, 10))
    fig.suptitle("KL Control — Diagnostic Metrics", fontsize=14, y=0.98)

    # ── 1. Cumulative stage cost ──
    # ℓ_k(x_k) summed over time; should flatten as the system stabilizes
    ax = axes[0, 0]
    costs = np.array([stage_cost(s) for s in state_history[:num_steps]])
    cumulative_cost = np.cumsum(costs)
    ax.plot(time_vec, cumulative_cost, color="tab:blue")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("Σ ℓ(x)")
    ax.set_title("Cumulative Stage Cost")
    ax.grid(True, alpha=0.3)

    # ── 2. Policy entropy H(π*_k) ──
    # Shannon entropy: H(π) = −Σ π(u) ln π(u)
    # High early on (exploration / uncertainty), decreases as the system
    # approaches equilibrium and the policy concentrates on fewer actions.
    ax = axes[0, 1]
    entropies = np.zeros(num_steps)
    for i, pi in enumerate(policy_history):
        mask = pi > 0
        entropies[i] = -np.sum(pi[mask] * np.log(pi[mask]))
    ax.plot(time_vec, entropies, color="tab:orange")
    # Reference: maximum entropy for uniform distribution over |U|
    max_entropy = np.log(len(INPUT_SPACE))
    ax.axhline(max_entropy, color="grey", ls="--", lw=0.8,
               label=f"H_max = ln|U| = {max_entropy:.2f}")
    # Reference: noise prior entropy H(ρ_w) — theoretical floor for π*
    # Optimal policy cannot have lower entropy (Problem 3.1:
    # deterministic policies incur infinite KL cost)
    noise_entropy = -np.sum(_NOISE_PROBS * np.log(_NOISE_PROBS))
    ax.axhline(noise_entropy, color="tab:red", ls="--", lw=1.0,
               label=f"H(ρ_w) = {noise_entropy:.2f} (floor)")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("H(π*)")
    ax.set_title("Policy Entropy (should floor at H(ρ_w), never → 0)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # ── 3. KL divergence KL(π*_k || ρ_w) ──
    # Measures how much the optimal policy deviates from the uncontrolled
    # noise prior ρ_w.  From Equation (13), the policy re-weights ρ_w by
    # desirability; this KL quantifies the "effort" of control.
    ax = axes[1, 0]
    kl_divs = np.zeros(num_steps)
    for i, pi in enumerate(policy_history):
        mask = pi > 0
        kl_divs[i] = np.sum(
            pi[mask] * (np.log(pi[mask]) - np.log(_NOISE_PROBS[mask]))
        )
    ax.plot(time_vec, kl_divs, color="tab:red")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("KL(π* || ρ_w)")
    ax.set_title("KL Divergence from Noise Prior")
    ax.grid(True, alpha=0.3)

    # ── 4. State norm ||x||₂ ──
    # Should converge toward 0 (upright equilibrium at origin)
    ax = axes[1, 1]
    state_norms = np.linalg.norm(state_history[:num_steps], axis=1)
    ax.plot(time_vec, state_norms, color="tab:green")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("||x||₂")
    ax.set_title("State Norm")
    ax.grid(True, alpha=0.3)

    # ── 5. Log-desirability log Z(k,x) ──
    # log A_ρ̄[Z](k,x) from Theorem 4.1.  Should increase as the state
    # improves, reflecting more "desirable" reachable futures.
    ax = axes[2, 0]
    ax.plot(time_vec, log_desirability_history, color="tab:purple")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("log Z")
    ax.set_title("Log-Desirability")
    ax.grid(True, alpha=0.3)

    # Hide the unused subplot
    axes[2, 1].set_visible(False)

    fig.tight_layout()
    fig.savefig("exploration/metrics.png", dpi=150, bbox_inches="tight")
    print(f"\nMetrics figure saved to exploration/metrics.png")
    plt.show()


# =============================================================================
# SECTION 13b: PAPER VALIDATION METRICS
# Reproduces key figures from Ito & Kashima (2022):
#   - MC convergence of V(0, x₀) (cf. Fig. 1)
#   - Multi-trajectory overlay in x̄-θ plane (cf. Fig. 4)
#   - Mean ± 1σ bands for state components (cf. Fig. 5)
# =============================================================================

def plot_paper_validation_metrics(num_seeds=10, multi_steps=80,
                                  multi_samples=500):
    """
    Generate paper validation metrics and save to
    exploration/paper_validation_metrics.png.

    1. MC convergence: V(0, x₀) = -log Z(0, x₀) vs sample count with
       error bars (10 reps each). Validates MC estimation converges.
    2. Multi-trajectory overlay: all seeds plotted in x̄-θ phase plane.
       All should converge toward origin.
    3-4. Mean ± 1σ bands of state components over time. Means → 0,
         stds plateau (confirming stochastic policy doesn't collapse).

    Args:
        num_seeds: number of independent simulation runs
        multi_steps: simulation length per seed (reduced for speed)
        multi_samples: MC samples per control (reduced for speed)
    """
    print("\n" + "=" * 60)
    print("Computing paper validation metrics...")
    print(f"  MC convergence: 6 sample counts × 10 reps")
    print(f"  Multi-seed: {num_seeds} seeds × {multi_steps} steps, "
          f"{multi_samples} samples/ctrl")
    print("=" * 60)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Paper Validation Metrics (Ito & Kashima 2022)",
                 fontsize=14, y=0.98)

    # ── Panel 1: MC Convergence of V(0, x₀) (cf. Fig. 1) ──
    # V(k, x) = -log Z(k, x) is the value function.
    # As num_samples → ∞, the MC estimate should converge; variance
    # should shrink as O(1/√N).
    ax = axes[0, 0]
    sample_counts = [100, 500, 1000, 2000, 5000, 10000]
    num_reps = 10
    v_means = []
    v_stds = []

    print("\n[1/3] MC convergence...")
    for n in tqdm(sample_counts, desc="Sample counts"):
        values = []
        for _ in range(num_reps):
            log_z = compute_desirability_function(
                0, INITIAL_STATE, HORIZON, num_samples=n
            )
            values.append(-log_z)  # V = -log Z
        v_means.append(np.mean(values))
        v_stds.append(np.std(values))

    ax.errorbar(sample_counts, v_means, yerr=v_stds,
                marker='o', capsize=5, color='tab:blue', lw=1.5)
    ax.set_xlabel("Number of MC samples")
    ax.set_ylabel("V(0, x₀) = −log Z(0, x₀)")
    ax.set_title("MC Convergence of Value Function (cf. Fig. 1)")
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3)

    # ── Multi-seed simulation (shared data for panels 2-4) ──
    print(f"\n[2/3] Running {num_seeds} simulations "
          f"({multi_steps} steps each)...")
    all_histories = []
    for seed in tqdm(range(num_seeds), desc="Multi-seed sims"):
        np.random.seed(seed + 42)
        state = INITIAL_STATE.copy()
        history = [state.copy()]
        for _ in range(multi_steps):
            policy, _ = compute_optimal_policy(
                time_index=0, state=state, horizon=HORIZON,
                num_samples_per_control=multi_samples,
            )
            control = sample_control_from_policy(policy)
            state = dynamics_step(state, control)
            history.append(state.copy())
        all_histories.append(np.array(history))

    # (num_seeds, multi_steps+1, 4)
    all_states = np.array(all_histories)
    time_vec = np.arange(multi_steps + 1) * TIMESTEP

    # ── Panel 2: Trajectory overlay in x̄-θ plane (cf. Fig. 4) ──
    ax = axes[0, 1]
    for i, hist in enumerate(all_histories):
        label = (f"seed {i}" if i < 3
                 else ("_nolegend_" if i < num_seeds - 1
                       else f"… {num_seeds} total"))
        ax.plot(hist[:, 0], hist[:, 2], alpha=0.6, lw=1.0, label=label)
    ax.plot(INITIAL_STATE[0], INITIAL_STATE[2], 'rs',
            markersize=10, zorder=5, label="start")
    ax.plot(0, 0, 'k*', markersize=15, zorder=5, label="target (origin)")
    ax.set_xlabel("Cart position x̄ (m)")
    ax.set_ylabel("Pole angle θ (rad)")
    ax.set_title(f"Trajectory Overlay — {num_seeds} seeds, "
                 f"{multi_steps} steps (cf. Fig. 4)")
    ax.legend(fontsize=7, loc="upper right")
    ax.grid(True, alpha=0.3)

    # ── Panel 3: Mean ± 1σ — position states (cf. Fig. 5) ──
    print("\n[3/3] Plotting statistics...")
    state_labels = ["x̄ (m)", "x̄̇ (m/s)", "θ (rad)", "θ̇ (rad/s)"]
    colors = ["tab:blue", "tab:cyan", "tab:red", "tab:orange"]

    ax = axes[1, 0]
    for j in [0, 1]:
        mean = all_states[:, :, j].mean(axis=0)
        std = all_states[:, :, j].std(axis=0)
        ax.plot(time_vec, mean, color=colors[j],
                label=state_labels[j], lw=1.5)
        ax.fill_between(time_vec, mean - std, mean + std,
                        color=colors[j], alpha=0.2)
    ax.axhline(0, color="grey", ls="--", lw=0.8)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("value")
    ax.set_title("Mean ± 1σ — Position States (cf. Fig. 5)\n"
                 "σ plateau confirms stochastic policy")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # ── Panel 4: Mean ± 1σ — angle states (cf. Fig. 5) ──
    ax = axes[1, 1]
    for j in [2, 3]:
        mean = all_states[:, :, j].mean(axis=0)
        std = all_states[:, :, j].std(axis=0)
        ax.plot(time_vec, mean, color=colors[j],
                label=state_labels[j], lw=1.5)
        ax.fill_between(time_vec, mean - std, mean + std,
                        color=colors[j], alpha=0.2)
    ax.axhline(0, color="grey", ls="--", lw=0.8)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("value")
    ax.set_title("Mean ± 1σ — Angle States (cf. Fig. 5)\n"
                 "σ plateau confirms stochastic policy")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig("exploration/paper_validation_metrics.png",
                dpi=150, bbox_inches="tight")
    print(f"\nPaper validation metrics saved to "
          f"exploration/paper_validation_metrics.png")
    plt.show()


# =============================================================================
# SECTION 14: ANIMATION
# =============================================================================

def animate_cart_pole(state_history, control_history):
    """
    Create a matplotlib animation of the cart-pole simulation.

    Args:
        state_history: (T+1, 4) array — [cart_pos, cart_vel, pole_angle, pole_ang_vel]
        control_history: (T,) array — applied force at each step
    """
    num_frames = len(state_history)

    # ── figure layout ──
    fig = plt.figure(figsize=(12, 7))
    gs = fig.add_gridspec(2, 2, height_ratios=[3, 1], hspace=0.45, wspace=0.30)
    ax_cart = fig.add_subplot(gs[0, :])   # cart-pole visualisation (top)
    ax_state = fig.add_subplot(gs[1, 0])  # state traces (bottom-left)
    ax_ctrl = fig.add_subplot(gs[1, 1])   # control input (bottom-right)

    # ── cart-pole axes setup ──
    cart_positions = state_history[:, 0]
    x_margin = 1.5
    x_lo = min(cart_positions.min(), -0.5) - x_margin
    x_hi = max(cart_positions.max(),  0.5) + x_margin
    ax_cart.set_xlim(x_lo, x_hi)
    ax_cart.set_ylim(-0.5, ROD_LENGTH + 0.6)
    ax_cart.set_aspect("equal")
    ax_cart.set_xlabel("x (m)")
    ax_cart.set_title("KL Control — Cart-Pole")

    # ground line
    ax_cart.axhline(0, color="grey", lw=0.8, ls="--")
    # target position
    ax_cart.axvline(0, color="green", lw=0.8, ls=":", alpha=0.5, label="target x=0")

    # cart rectangle (Rectangle works on all matplotlib versions)
    cart_w, cart_h = 0.4, 0.2
    cart_patch = patches.Rectangle(
        (0, 0), cart_w, cart_h,
        fc="royalblue", ec="black", lw=1.5, zorder=3,
    )
    ax_cart.add_patch(cart_patch)

    # pole line & bob
    (pole_line,) = ax_cart.plot([], [], "o-", color="firebrick", lw=3,
                                 markersize=10, markerfacecolor="orange",
                                 markeredgecolor="firebrick", zorder=4)

    # force arrow
    force_arrow = ax_cart.annotate(
        "", xy=(0, 0), xytext=(0, 0),
        arrowprops=dict(arrowstyle="->", color="purple", lw=2),
        zorder=5,
    )

    # info text
    info_text = ax_cart.text(
        0.02, 0.95, "", transform=ax_cart.transAxes,
        fontsize=9, verticalalignment="top", fontfamily="monospace",
    )

    # ── state trace axes ──
    time_vec = np.arange(num_frames) * TIMESTEP
    ax_state.set_xlim(0, time_vec[-1])
    ax_state.set_xlabel("time (s)")
    ax_state.set_ylabel("state")
    ax_state.set_title("State trajectories")
    line_x, = ax_state.plot([], [], label="x (m)", color="tab:blue")
    line_theta, = ax_state.plot([], [], label="θ (rad)", color="tab:red")
    ax_state.legend(loc="upper right", fontsize=8)
    ax_state.axhline(0, color="grey", lw=0.5, ls="--")
    y_abs = max(np.abs(state_history[:, 0]).max(), np.abs(state_history[:, 2]).max()) * 1.2
    ax_state.set_ylim(-y_abs, y_abs)

    # ── control trace axes ──
    ctrl_time = np.arange(len(control_history)) * TIMESTEP
    ax_ctrl.set_xlim(0, time_vec[-1])
    ax_ctrl.set_ylim(INPUT_SPACE[0] - 2, INPUT_SPACE[-1] + 2)
    ax_ctrl.set_xlabel("time (s)")
    ax_ctrl.set_ylabel("u (N)")
    ax_ctrl.set_title("Control input")
    ax_ctrl.axhline(0, color="grey", lw=0.5, ls="--")
    line_u, = ax_ctrl.plot([], [], color="purple", lw=0.8)

    # ── animation update ──
    def update(frame):
        cart_x = state_history[frame, 0]
        theta = state_history[frame, 2]

        # cart
        cart_patch.set_xy((cart_x - cart_w / 2, -cart_h / 2))

        # pole: θ=0 is upright, positive θ tilts right
        pole_tip_x = cart_x + ROD_LENGTH * np.sin(theta)
        pole_tip_y = ROD_LENGTH * np.cos(theta)
        pole_line.set_data([cart_x, pole_tip_x], [0, pole_tip_y])

        # force arrow
        if frame < len(control_history):
            u = control_history[frame]
            arrow_scale = 0.02  # scale force to visual length
            force_arrow.xy = (cart_x + np.sign(u) * cart_w / 2 + u * arrow_scale, 0)
            force_arrow.xyann = (cart_x + np.sign(u) * cart_w / 2, 0)
            force_arrow.set_visible(abs(u) > 0.1)
        else:
            force_arrow.set_visible(False)

        # info
        t = frame * TIMESTEP
        u_val = control_history[frame] if frame < len(control_history) else 0.0
        info_text.set_text(
            f"t = {t:.2f} s   step {frame}/{num_frames-1}\n"
            f"x = {cart_x:+.3f} m    θ = {theta:+.3f} rad\n"
            f"u = {u_val:+.1f} N"
        )

        # state traces up to current frame
        line_x.set_data(time_vec[:frame + 1], state_history[:frame + 1, 0])
        line_theta.set_data(time_vec[:frame + 1], state_history[:frame + 1, 2])

        if frame < len(control_history):
            line_u.set_data(ctrl_time[:frame + 1], control_history[:frame + 1])

        return cart_patch, pole_line, force_arrow, info_text, line_x, line_theta, line_u

    anim = FuncAnimation(
        fig, update, frames=num_frames,
        interval=TIMESTEP * 1000,   # real-time playback
        blit=False, repeat=True,
    )

    plt.show()
    return anim


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("Running KL control cart-pole simulation...")
    print(f"  Steps: {TOTAL_SIMULATION_TIME}, Horizon: {HORIZON}, "
          f"Samples/control: 1000, |U|={len(INPUT_SPACE)}")
    print(f"  Workers: {NUM_WORKERS}\n")

    state_hist, ctrl_hist, policy_hist, log_z_hist = simulation()
    plot_metrics(state_hist, ctrl_hist, policy_hist, log_z_hist)
    plot_paper_validation_metrics()
    animate_cart_pole(state_hist, ctrl_hist)