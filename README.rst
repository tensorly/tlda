==================================
Tensor Latent Dirichlet Allocation
==================================

A scalable, GPU-accelerated online tensor LDA, built on TensorLy and PyTorch.

.. image:: https://tensorly.org/tlda/dev/_images/overview-tlda.png

As batches of documents arrive online, they are first pre-processed. The resulting document term matrix is centered.
The whitening transformation is updated online and used to whiten. Finally, the third order moment is updated, directly in factorized form. This learned factorization can be directly unwhiten and uncentered to recover the classic solution to the tensor LDA and recover the topics.

Citation
========

If you use this work, please cite our paper [1]_, published in Political Analysis, and available on Arxiv ([2]_)::

  @article{Kangaslahti_Ebanks_Kossaifi_Liu_Alvarez_Anandkumar_2026,
    title={Analyzing Political Text at Scale with Online Tensor LDA},
    volume={34}, 
    DOI={10.1017/pan.2025.10024},
    number={1},
    journal={Political Analysis}, 
    author={Kangaslahti, Sara and Ebanks, Danny and Kossaifi, Jean and Liu, Anqi and Alvarez, R. Michael and Anandkumar, Animashree},
    year={2026}, 
    pages={53–77}
  }


.. [1] Kangaslahti S, Ebanks D, Kossaifi J, Liu A, Alvarez RM, Anandkumar A. Analyzing Political Text at Scale with Online Tensor LDA. Political Analysis. 2026;34(1):53-77. doi:10.1017/pan.2025.10024

.. [2] arxiv version: https://arxiv.org/abs/2511.07809
