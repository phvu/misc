# Training MNIST distributedly with tensorflow

In `distributed_tf_sample.py` is a simple script to train a 2-hidden layer network distributedly.
This is an implementation of "synchronous" training, where the parameters are hold
in a parameter server, and each worker has a separated model.

## How to run

Start the server with the following commands:

    $ ./bazel-bin/tensorflow/core/distributed_runtime/rpc/grpc_tensorflow_server \
        --cluster_spec='ps|localhost:2222,worker|localhost:2223;localhost:2224' --job_name=ps --task_id=0 &

    $ ./bazel-bin/tensorflow/core/distributed_runtime/rpc/grpc_tensorflow_server \
        --cluster_spec='ps|localhost:2222,worker|localhost:2223;localhost:2224' --job_name=worker --task_id=0 &

    $ ./bazel-bin/tensorflow/core/distributed_runtime/rpc/grpc_tensorflow_server \
        --cluster_spec='ps|localhost:2222,worker|localhost:2223;localhost:2224' --job_name=worker --task_id=1 &

Then you can run the sample:

    $ python distributed_tf_sample.py
    
Note that how we use the same `cluster_spec`, but different values for `job_name` and `task_id`. 
In this setting, we use 1 task for _parameter server_, and 2 tasks for workers.

# Notes
This is just a sample implementation. There are a lot of things that are not optimal:

- We use `feeding` to feed training batches. A proper implementation should 
use [input pipeline](https://www.tensorflow.org/versions/r0.7/how_tos/reading_data/index.html).
- Model checkpointing is not implemented.

