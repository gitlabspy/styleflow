"""
  conditional ffjord
"""
import tensorflow.compat.v2 as tf
from tensorflow_probability.python import math as tfp_math
from tensorflow_probability.python.bijectors import bijector
from tensorflow_probability.python.internal import cache_util
from tensorflow_probability.python.internal import prefer_static

def trace_jacobian_hutchinson(
    ode_fn,
    state_shape,
    dtype,
    sample_fn=tf.random.normal,
    num_samples=1,
    seed=None):
  random_samples = sample_fn(
      prefer_static.concat([[num_samples], state_shape], axis=0),
      dtype=dtype, seed=seed)

  def augmented_ode_fn(time, state_log_det_jac):
    state, _ , condition= state_log_det_jac
    with tf.GradientTape(persistent=True,
                         watch_accessed_variables=False) as tape:
      tape.watch(state)
      tape.watch(condition) # ?
      state_time_derivative = ode_fn(time, state, condition)

    def estimate_trace(random_sample):
      jvp = tape.gradient(state_time_derivative, state, random_sample)
      return random_sample * jvp

    results = tf.map_fn(estimate_trace, random_samples)
    trace_estimates = tf.reduce_mean(results, axis=0)
    return state_time_derivative, trace_estimates, condition

  return augmented_ode_fn

class FFJORD(bijector.Bijector):
  _cache = cache_util.BijectorCacheWithGreedyAttrs(
      forward_name='_augmented_forward',
      inverse_name='_augmented_inverse')

  def __init__(
      self,
      state_time_derivative_fn,
      ode_solve_fn=None,
      trace_augmentation_fn=trace_jacobian_hutchinson,
      initial_time=0.,
      final_time=1.,
      validate_args=False,
      dtype=tf.float32,
      name='ffjord'):
    parameters = dict(locals())
    with tf.name_scope(name) as name:
      self._initial_time = initial_time
      self._final_time = final_time
      self._ode_solve_fn = ode_solve_fn
      if self._ode_solve_fn is None:
        self._ode_solver = tfp_math.ode.DormandPrince()
        self._ode_solve_fn = self._ode_solver.solve
      self._trace_augmentation_fn = trace_augmentation_fn
      self._state_time_derivative_fn = state_time_derivative_fn

      def inverse_state_time_derivative(time, state, condition):
        return -state_time_derivative_fn(self._final_time - time, state, condition)

      self._inv_state_time_derivative_fn = inverse_state_time_derivative
      super(FFJORD, self).__init__(
          forward_min_event_ndims=0,
          dtype=dtype,
          validate_args=validate_args,
          parameters=parameters,
          name=name)

  def _solve_ode(self, ode_fn, state):
    integration_result = self._ode_solve_fn(
        ode_fn=ode_fn,
        initial_time=self._initial_time,
        initial_state=state,
        solution_times=[self._final_time])
    final_state = tf.nest.map_structure(
        lambda x: x[-1], integration_result.states)
    return final_state

  def _augmented_forward(self, x, condition):
    """Computes forward and forward_log_det_jacobian transformations."""
    augmented_ode_fn = self._trace_augmentation_fn(
        self._state_time_derivative_fn, x.shape, x.dtype)
    augmented_x = (x, tf.zeros(shape=x.shape, dtype=x.dtype), condition)
    y, fldj, _ = self._solve_ode(augmented_ode_fn, augmented_x)
    return y, {'ildj': -fldj, 'fldj': fldj}

  def _augmented_inverse(self, y, condition):
    """Computes inverse and inverse_log_det_jacobian transformations."""
    augmented_inv_ode_fn = self._trace_augmentation_fn(
        self._inv_state_time_derivative_fn, y.shape, y.dtype)
    augmented_y = (y, tf.zeros(shape=y.shape, dtype=y.dtype), condition)
    x, ildj, _ = self._solve_ode(augmented_inv_ode_fn, augmented_y)
    return x, {'ildj': ildj, 'fldj': -ildj}

  def _forward(self, x, condition):
    y, _ = self._augmented_forward(x, condition)
    return y

  def _inverse(self, y, condition):
    x, _ = self._augmented_inverse(y, condition)
    return x

  def _forward_log_det_jacobian(self, x, condition):
    cached = self._cache.forward_attributes(x, condition)
    # If LDJ isn't in the cache, call forward once.
    if 'fldj' not in cached:
      _, attrs = self._augmented_forward(x, condition)
      cached.update(attrs)
    return cached['fldj']

  def _inverse_log_det_jacobian(self, y, condition):
    cached = self._cache.inverse_attributes(y, condition)
    # If LDJ isn't in the cache, call inverse once.
    if 'ildj' not in cached:
      _, attrs = self._augmented_inverse(y, condition)
      cached.update(attrs)
    return cached['ildj']
