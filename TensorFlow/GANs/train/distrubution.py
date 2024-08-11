import tensorflow as tf
import os
from enum import Enum

def get_strategy():
    """
    Returns a TensorFlow distribution strategy based on the available hardware.

    This function first checks if the environment variable "COLAB_TPU_ADDR" is present in the system.
    If it is, it creates a TPU cluster resolver and initializes the TPU system. It then returns a TPUStrategy using the TPU cluster.

    If the environment variable is not present, the function checks if there are any GPUs available.
    If there are more than one GPUs, it returns a MirroredStrategy to distribute the computation across multiple GPUs.
    If there is only one GPU, it returns a MirroredStrategy using that GPU.
    If there are no GPUs, it returns the default distribution strategy.

    If any errors occur during the process, it returns the default distribution strategy.

    Returns:
        tf.distribute.Strategy: The TensorFlow distribution strategy based on the available hardware.
    """
    try:
        tpu_url = None
        if "COLAB_TPU_ADDR" in os.environ:
            tpu_url = "grpc://" + os.environ["COLAB_TPU_ADDR"]
        if tpu_url:
            tpu = tf.distribute.cluster_resolver.TPUClusterResolver(tpu_url)
            tf.config.experimental_connect_to_cluster(tpu)
            tf.tpu.experimental.initialize_tpu_system(tpu)
            strategy = tf.distribute.TPUStrategy(tpu)
            print("Using TPU strategy")
        else:
            gpus = tf.config.experimental.list_physical_devices("GPU")
            if len(gpus) > 1:
                strategy = tf.distribute.MirroredStrategy()
                print("Using multi-GPU strategy")
            elif len(gpus) == 1:
                strategy = tf.distribute.MirroredStrategy(devices=["/gpu:0"])
                print("Using single-GPU strategy")
            else:
                strategy = tf.distribute.get_strategy()
                print("Using default strategy")
    except (RuntimeError, ValueError) as e:
        print(f"Error occurred: {e}")
        strategy = tf.distribute.get_strategy()
        print("Using default strategy due to error")

    return strategy


class Reduction(Enum):
    NONE = 0
    SUM = 1
    MEAN = 2
    CONCAT = 3 

def distributed(strategy=None, *reduction_flags):
    """
    Decorator function that applies a distribution strategy to a given function.

    Args:
        strategy (tf.distribute.Strategy, optional): The distribution strategy to use. Defaults to None.
        *reduction_flags (Reduction): The reduction flag(s) to apply to the output of the function.

    Returns:
        function: The decorated function with the distribution strategy applied.

    Raises:
        NotImplementedError: If an invalid reduction flag is provided.

    Note:
        The decorated function must return a tensor or a list of tensors.

    Example:
        @distributed(tf.distribute.MirroredStrategy(), Reduction.SUM, Reduction.MEAN)
        def my_function(x):
            return x * 2, x ** 2

        result1, result2 = my_function(tf.ones((2, 3)))
        # result1 is the sum of the elements in x * 2 across all replicas
        # result2 is the mean of the elements in x ** 2 across all replicas
    """
    def _decorator(func):
        def pre_replica_reduction(z, flag):
            """
            Applies the specified reduction operation to the given tensor.

            Args:
                z (tf.Tensor): The tensor to be reduced.
                flag (Reduction): The reduction operation to be applied.

            Returns:
                tf.Tensor: The reduced tensor.

            Raises:
                NotImplementedError: If an invalid reduction flag is provided.
            """
            if flag == Reduction.NONE:
                return z
            elif flag == Reduction.SUM:
                return strategy.reduce(tf.distribute.ReduceOp.SUM, z, axis=None) if strategy else tf.reduce_sum(z)
            elif flag == Reduction.MEAN:
                return strategy.reduce(tf.distribute.ReduceOp.MEAN, z, axis=None) if strategy else tf.reduce_mean(z)
            elif flag == Reduction.CONCAT:
                return tf.concat(strategy.experimental_local_results(z), 0) if strategy else z
            else:
                raise NotImplementedError(f"Reduction flag {flag} is not implemented")

        @tf.function
        def _decorated(*args, **kwargs):
            """
            Executes the decorated function using the distribution strategy and applies the specified reduction operations.

            Args:
                *args: Positional arguments to be passed to the decorated function.
                **kwargs: Keyword arguments to be passed to the decorated function.

            Returns:
                The result of the decorated function after applying the distribution strategy and reduction flags.

            Raises:
                AssertionError: If the function result is not a tensor or a tuple of tensors.
                NotImplementedError: If an invalid reduction flag is provided.
            """
            if strategy:
                fun_result = strategy.run(func, args=args, kwargs=kwargs)
            else:
                fun_result = func(*args, **kwargs)

            if len(reduction_flags) == 0:
                return fun_result
            elif len(reduction_flags) == 1:
                assert isinstance(fun_result, tf.Tensor) or fun_result is None
                return pre_replica_reduction(fun_result, reduction_flags[0])
            else:
                assert isinstance(fun_result, tuple) or fun_result is None
                return tuple(pre_replica_reduction(z, flag) for z, flag in zip(fun_result, reduction_flags))
                
        return _decorated
    return _decorator


# # Example usage:
# strategy = get_strategy()

# @distributed(strategy, Reduction.SUM, Reduction.MEAN)
# def my_function(x):
#     return x * 2, x ** 2

# result1, result2 = my_function(tf.ones((2, 3)))
# print("Result 1:", result1)
# print("Result 2:", result2)
