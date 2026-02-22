import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
from concurrent.futures import ProcessPoolExecutor
import os
from tqdm import tqdm

TIMESTEP = 0.05  # τ in paper: Euler discretization time step (seconds)
HORIZON = 40     # N in paper: finite horizon length (number of time steps)
TOTAL_SIMULATION_TIME = 100
EPS = 1e-12

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

    # Keep raw log_weights (unnormalized) for diagnostics (rank, regret)
    log_weights_raw = log_weights.copy()

    # Softmax → probability distribution π*_k(u|x)
    log_weights -= np.max(log_weights)
    weights = np.exp(log_weights)
    policy_distribution = weights / np.sum(weights)

    return policy_distribution, log_weights_raw


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


def plot_diagnostics(time_vec, metrics, savefig=False):
    """
    Plot diagnostics collected during simulation.

    metrics: dict with keys 'entropy', 'regret', 'rank', 'expected_stage_cost', 'cumulative_cost'
    """
    fig, axes = plt.subplots(4, 1, figsize=(8, 10), sharex=True)

    axes[0].plot(time_vec, metrics['entropy'], color='tab:blue')
    axes[0].set_ylabel('Policy entropy')
    axes[0].grid(True)

    axes[1].plot(time_vec, metrics['regret'], color='tab:red')
    axes[1].set_ylabel('Regret (log-weight gap)')
    axes[1].grid(True)

    axes[2].plot(time_vec, metrics['rank'], color='tab:green')
    axes[2].set_ylabel('Chosen action rank')
    axes[2].grid(True)

    axes[3].plot(time_vec, metrics['cumulative_cost'], color='tab:purple', label='actual')
    axes[3].plot(time_vec, metrics['expected_stage_cost'], color='tab:orange', label='expected stage cost')
    axes[3].set_ylabel('Cost')
    axes[3].set_xlabel('time (s)')
    axes[3].legend()
    axes[3].grid(True)

    fig.tight_layout()
    if savefig:
        plt.savefig('klcontrol_diagnostics.png', dpi=200)
    plt.show()

# NOTE: static diagnostics plot kept for backward compatibility but
# the primary visualisation is now the animated diagnostics inside
# `animate_cart_pole` which shows policy bars, log-weights and live metrics.







def simulation():
    """
    Run the full KL control simulation and record state/control history.

    Returns:
        state_history: (TOTAL_SIMULATION_TIME+1, 4) array of states
        control_history: (TOTAL_SIMULATION_TIME,) array of applied controls
    """
    state = INITIAL_STATE.copy()
    state_history = [state.copy()]
    control_history = []

    # Containers for diagnostics
    entropies = []
    regrets = []
    ranks = []
    expected_stage_costs = []
    cumulative_cost = 0.0
    cumulative_costs = []

    pbar = tqdm(range(TOTAL_SIMULATION_TIME), desc="KL control", unit="step")
    for i in pbar:
        policy, log_weights = compute_optimal_policy(
            time_index=0, state=state, horizon=HORIZON
        )
        # store policy / log-weights for animation diagnostics
        try:
            policy_history.append(policy.copy())
            log_weights_history.append(log_weights.copy())
        except NameError:
            policy_history = [policy.copy()]
            log_weights_history = [log_weights.copy()]
        # sample action
        control_input = sample_control_from_policy(policy_distribution=policy)

        # diagnostics: entropy
        entropy = -np.sum(policy * np.log(policy + EPS))

        # find index of chosen control and compute regret / rank
        chosen_idx = int(np.where(INPUT_SPACE == control_input)[0][0])
        max_log = np.max(log_weights)
        regret = max_log - log_weights[chosen_idx]
        # rank: 1 is best (highest log-weight)
        rank = int(np.argsort(-log_weights).tolist().index(chosen_idx) + 1)

        # expected immediate stage cost under policy
        next_states = np.array([dynamics_step(state, u) for u in INPUT_SPACE])
        stage_costs = np.array([stage_cost(s) for s in next_states])
        expected_stage_cost = np.sum(policy * stage_costs)

        # apply chosen action
        state = dynamics_step(state, control_or_noise=control_input)

        # update cumulative cost (actual realized cost)
        cumulative_cost += stage_cost(state)

        # record
        state_history.append(state.copy())
        control_history.append(control_input)
        entropies.append(entropy)
        regrets.append(regret)
        ranks.append(rank)
        expected_stage_costs.append(expected_stage_cost)
        cumulative_costs.append(cumulative_cost)

        pbar.set_postfix_str(
            f"x={state[0]:+.2f} θ={state[2]:+.2f} u={control_input:+.0f}"
        )

    metrics = {
        'entropy': np.array(entropies),
        'regret': np.array(regrets),
        'rank': np.array(ranks),
        'expected_stage_cost': np.array(expected_stage_costs),
        'cumulative_cost': np.array(cumulative_costs),
        'policy': np.vstack(policy_history),
        'log_weights': np.vstack(log_weights_history),
    }

    return np.array(state_history), np.array(control_history), metrics


# =============================================================================
# SECTION 13: ANIMATION
# =============================================================================

def animate_cart_pole(state_history, control_history, metrics=None):
    """
    Create a matplotlib animation of the cart-pole simulation.

    Args:
        state_history: (T+1, 4) array — [cart_pos, cart_vel, pole_angle, pole_ang_vel]
        control_history: (T,) array — applied force at each step
    """
    num_frames = len(state_history)

    # ── figure layout ──
    fig = plt.figure(figsize=(14, 8))
    # layout: 2 rows x 3 cols
    gs = fig.add_gridspec(2, 3, width_ratios=[2, 2, 1], height_ratios=[3, 1], hspace=0.35, wspace=0.35)
    ax_cart = fig.add_subplot(gs[0, 0:2])   # cart-pole visualisation (top-left, spans 2 cols)
    ax_policy = fig.add_subplot(gs[0, 2])   # policy bar chart (top-right)
    ax_state = fig.add_subplot(gs[1, 0:2])  # state traces (bottom-left, spans 2 cols)
    ax_diag = fig.add_subplot(gs[1, 2])     # diagnostics (entropy / regret) (bottom-right)

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

    # ── policy bar chart axes ──
    ax_policy.set_title('Policy distribution (π_k over actions)')
    ax_policy.set_xlabel('u (N)')
    ax_policy.set_ylabel('probability')
    bars = ax_policy.bar(INPUT_SPACE, np.zeros_like(INPUT_SPACE), color='tab:blue')
    ax_policy.set_ylim(0, 1.0)

    # ── diagnostics axes (entropy + regret) ──
    ax_diag.set_title('Entropy (blue) and Regret (red)')
    ax_diag.set_xlabel('time (s)')
    ax_diag.set_ylabel('value')
    line_entropy, = ax_diag.plot([], [], color='tab:blue', lw=1)
    line_regret, = ax_diag.plot([], [], color='tab:red', lw=1)
    ax_diag.grid(True)

    # ── control trace axes (reuse small area in state subplot bottom) ──
    ctrl_time = np.arange(len(control_history)) * TIMESTEP
    ax_ctrl = ax_state.twinx()
    ax_ctrl.set_ylabel('u (N)')
    line_u, = ax_ctrl.plot([], [], color='purple', lw=0.8)

    # ── animation update ──
    # prepare metric arrays
    policy_hist = None
    logw_hist = None
    entropy_hist = None
    regret_hist = None
    if metrics is not None:
        policy_hist = metrics.get('policy')
        logw_hist = metrics.get('log_weights')
        entropy_hist = metrics.get('entropy')
        regret_hist = metrics.get('regret')

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

        # update policy bar chart and diagnostics if available
        if policy_hist is not None and frame < len(policy_hist):
            p = policy_hist[frame]
            for rect, h in zip(bars, p):
                rect.set_height(h)
        elif policy_hist is not None and frame >= len(policy_hist):
            p = policy_hist[-1]
            for rect, h in zip(bars, p):
                rect.set_height(h)

        if entropy_hist is not None and regret_hist is not None:
            t = np.arange(min(frame + 1, len(entropy_hist))) * TIMESTEP
            ent = entropy_hist[: frame + 1] if frame < len(entropy_hist) else entropy_hist
            reg = regret_hist[: frame + 1] if frame < len(regret_hist) else regret_hist
            line_entropy.set_data(t, ent)
            line_regret.set_data(t, reg)
            ax_diag.set_xlim(0, time_vec[-1])
            # autoscale y for diagnostics
            allv = np.concatenate([ent, reg]) if len(ent) and len(reg) else np.array([0.0])
            ymin, ymax = np.min(allv), np.max(allv)
            pad = max(1e-6, 0.1 * (ymax - ymin))
            ax_diag.set_ylim(ymin - pad, ymax + pad)

        return cart_patch, pole_line, force_arrow, info_text, line_x, line_theta, line_u, bars

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

    state_hist, ctrl_hist, metrics = simulation()
    animate_cart_pole(state_hist, ctrl_hist, metrics=metrics)