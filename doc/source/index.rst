:no-toc:
:no-localtoc:
:no-pagination:

.. tlda documentation

.. only:: html

   .. raw:: html

      <br/><br/>

.. only:: html

   .. raw:: html 
   
      <div class="has-text-centered">
         <h2> Tensor LDA in PyTorch </h2>
      </div>
      <br/><br/>

.. only:: latex

   Tensor LDA in PyTorch
   =====================


A scalable, GPU-accelerated online tensor LDA, built on TensorLy and PyTorch.


.. image:: /_static/overview-tlda.png
   :align: center
   :width: 800


As batches of documents arrive online, they are first pre-processed. 
The resulting document term matrix is centered. 

The whitening transformation is updated online and used to whiten :math:`\matrix{X}`. Finally, the third order moment is updated, directly in factorized form.
This learned factorization can be directly unwhiten and uncentered to recover the classic solution to the tensor LDA 
and recover the topics.


.. only:: html

   .. raw:: html

      <br/> <br/>
      <br/>

      <div class="container has-text-centered">
      <a class="button is-large is-dark is-primary" href="install.html">
         Get started
      </a>
      </div>


.. toctree::
   :maxdepth: 1
   :hidden:

   install
   user_guide/index
   modules/api
