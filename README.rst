

Distributed-DL-training
=======

.. inclusion-marker-start-do-not-remove

|

Distributed-DL-training is a distributed training framework for TensorFlow, Keras, PyTorch, and MXNet. The goal of Distributed-DL-training is to make
distributed Deep Learning fast and easy to use.


|

.. contents::



|


Why not traditional Distributed TensorFlow?
-------------------------------------------

The primary motivation for this project is to make it easy to take a single-GPU TensorFlow program and successfully train
it on many GPUs faster. This has two aspects:

1. How much modification does one have to make to a program to make it distributed, and how easy is it to run it?
2. How much faster would it run in distributed mode?

Internally at Uber we found the MPI model to be much more straightforward and require far less code changes than the
Distributed TensorFlow with parameter servers. See the `Usage <#usage>`__ section for more details.

In addition to being easy to use, Distributed-DL-training is fast. Below is a chart representing the benchmark that was done on 128
servers with 4 Pascal GPUs each connected by RoCE-capable 25 Gbit/s network:

.. image:: https://user-images.githubusercontent.com/16640218/38965607-bf5c46ca-4332-11e8-895a-b9c137e86013.png
   :alt: 512-GPU Benchmark

Distributed-DL-training achieves 90% scaling efficiency for both Inception V3 and ResNet-101, and 68% scaling efficiency for VGG-16.

While installing MPI and NCCL itself may seem like an extra hassle, it only needs to be done once by the team dealing
with infrastructure, while everyone else in the company who builds the models can enjoy the simplicity of training them at
scale.


Install
-------

To install Distributed-DL-training:

1. Install `Open MPI <https://www.open-mpi.org/>`_ or another MPI implementation. Learn how to install Open MPI `on this page <https://www.open-mpi.org/faq/?category=building#easy-build>`_.

   **Note**: Open MPI 3.1.3 has an issue that may cause hangs.  The recommended fix is to
   downgrade to Open MPI 3.1.2 or upgrade to Open MPI 4.0.0.

.. raw:: html

    <p/>

2. If you've installed TensorFlow from `PyPI <https://pypi.org/project/tensorflow>`__, make sure that the ``g++-4.8.5`` or ``g++-4.9`` is installed.

   If you've installed PyTorch from `PyPI <https://pypi.org/project/torch>`__, make sure that the ``g++-4.9`` or above is installed.

   If you've installed either package from `Conda <https://conda.io>`_, make sure that the ``gxx_linux-64`` Conda package is installed.

.. raw:: html

    <p/>

3. Install the ``Distributed-DL-training`` .

.. code-block:: bash


This basic installation is good for laptops and for getting to know Distributed-DL-training.


Concepts
--------

Distributed-DL-training core principles are based on `MPI <http://mpi-forum.org/>`_ concepts such as *size*, *rank*,
*local rank*, **allreduce**, **allgather** and, *broadcast*. 

Supported frameworks
--------------------
Tensorflow, keras, Pytorch,MXNET 

Usage
-----

To use Distributed-DL-training, make the following additions to your program. This example uses TensorFlow.

1. Run ``hvd.init()``.

2. Pin a server GPU to be used by this process using ``config.gpu_options.visible_device_list``.
   With the typical setup of one GPU per process, this can be set to *local rank*. In that case, the first process on
   the server will be allocated the first GPU, second process will be allocated the second GPU and so forth.

3. Scale the learning rate by number of workers. Effective batch size in synchronous distributed training is scaled by
   the number of workers. An increase in learning rate compensates for the increased batch size.

4. Wrap optimizer in ``hvd.DistributedOptimizer``.  The distributed optimizer delegates gradient computation
   to the original optimizer, averages gradients using **allreduce** or **allgather**, and then applies those averaged
   gradients.

5. Add ``hvd.BroadcastGlobalVariablesHook(0)`` to broadcast initial variable states from rank 0 to all other processes.
   This is necessary to ensure consistent initialization of all workers when training is started with random weights or
   restored from a checkpoint. Alternatively, if you're not using ``MonitoredTrainingSession``, you can simply execute
   the ``hvd.broadcast_global_variables`` op after global variables have been initialized.

6. Modify your code to save checkpoints only on worker 0 to prevent other workers from corrupting them.
   This can be accomplished by passing ``checkpoint_dir=None`` to ``tf.train.MonitoredTrainingSession`` if
   ``hvd.rank() != 0``.


.. code-block:: python

    import tensorflow as tf
    import Distributed-DL-training.tensorflow as hvd


    # Initialize Distributed-DL-training
    hvd.init()

    # Pin GPU to be used to process local rank (one GPU per process)
    config = tf.ConfigProto()
    config.gpu_options.visible_device_list = str(hvd.local_rank())

    # Build model...
    loss = ...
    opt = tf.train.AdagradOptimizer(0.01 * hvd.size())

    # Add Distributed-DL-training Distributed Optimizer
    opt = hvd.DistributedOptimizer(opt)

    # Add hook to broadcast variables from rank 0 to all other processes during
    # initialization.
    hooks = [hvd.BroadcastGlobalVariablesHook(0)]

    # Make training operation
    train_op = opt.minimize(loss)

    # Save checkpoints only on worker 0 to prevent other workers from corrupting them.
    checkpoint_dir = '/tmp/train_logs' if hvd.rank() == 0 else None

    # The MonitoredTrainingSession takes care of session initialization,
    # restoring from a checkpoint, saving to a checkpoint, and closing when done
    # or an error occurs.
    with tf.train.MonitoredTrainingSession(checkpoint_dir=checkpoint_dir,
                                           config=config,
                                           hooks=hooks) as mon_sess:
      while not mon_sess.should_stop():
        # Perform synchronous training.
        mon_sess.run(train_op)


Running Distributed-DL-training
---------------


1. To run on a machine with 4 GPUs:

.. code-block:: bash

     $ Distributed-DL-trainingrun -np 4 -H localhost:4 python train.py

2. To run on 4 machines with 4 GPUs each:

.. code-block:: bash

    $ Distributed-DL-trainingrun -np 16 -H server1:4,server2:4,server3:4,server4:4 python train.py




Gloo
----
`Gloo <https://github.com/facebookincubator/gloo>`_ is an open source collective communications library developed by Facebook.

Gloo comes included with Distributed-DL-training, and allows users to run Distributed-DL-training without requiring MPI to be installed. Gloo support only requires
that you have `CMake <https://cmake.org/>`_ installed, and is only supported on Linux at this time.

For environments that have support both MPI and Gloo, you can choose to use Gloo at runtime by passing the ``--gloo`` argument to ``Distributed-DL-trainingrun``:

.. code-block:: bash

     $ Distributed-DL-trainingrun --gloo -np 2 python train.py

Gloo support is still early in its development, and more features are coming soon.

mpi4py
------
Distributed-DL-training supports mixing and matching Distributed-DL-training collectives with other MPI libraries, such as `mpi4py <https://mpi4py.scipy.org>`_,
provided that the MPI was built with multi-threading support.

You can check for MPI multi-threading support by querying the ``hvd.mpi_threads_supported()`` function.

.. code-block:: python

    import Distributed-DL-training.tensorflow as hvd

    # Initialize Distributed-DL-training
    hvd.init()

    # Verify that MPI multi-threading is supported.
    assert hvd.mpi_threads_supported()

    from mpi4py import MPI
    assert hvd.size() == MPI.COMM_WORLD.Get_size()

You can also initialize Distributed-DL-training with an `mpi4py` sub-communicator, in which case each sub-communicator
will run an independent Distributed-DL-training training.

.. code-block:: python

    from mpi4py import MPI
    import Distributed-DL-training.tensorflow as hvd

    # Split COMM_WORLD into subcommunicators
    subcomm = MPI.COMM_WORLD.Split(color=MPI.COMM_WORLD.rank % 2,
                                   key=MPI.COMM_WORLD.rank)

    # Initialize Distributed-DL-training
    hvd.init(comm=subcomm)

    print('COMM_WORLD rank: %d, Distributed-DL-training rank: %d' % (MPI.COMM_WORLD.rank, hvd.rank()))


Inference
---------
Learn how to optimize your model for inference and remove Distributed-DL-training operations from the graph `here <docs/inference.rst>`_.


Tensor Fusion
-------------
One of the unique things about Distributed-DL-training is its ability to interleave communication and computation coupled with the ability
to batch small **allreduce** operations, which results in improved performance. We call this batching feature Tensor Fusion.



Analyzing Distributed-DL-training Performance
-----------------------------
Distributed-DL-training has the ability to record the timeline of its activity, called Distributed-DL-training Timeline.

.. image:: https://user-images.githubusercontent.com/16640218/29735271-9e148da0-89ac-11e7-9ae0-11d7a099ac89.png
   :alt: Distributed-DL-training Timeline



Automated Performance Tuning
----------------------------
Selecting the right values to efficiently make use of Tensor Fusion and other advanced Distributed-DL-training features can involve
a good amount of trial and error. We provide a system to automate this performance optimization process called
**autotuning**, which you can enable with a single command line argument to ``Distributed-DL-trainingrun``.



