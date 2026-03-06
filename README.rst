``pytest-mpl-oggm``
===================

``pytest-mpl`` is a `pytest <https://docs.pytest.org>`__ plugin to facilitate image
comparison for `Matplotlib <http://www.matplotlib.org>`__ figures.

``pytest-mpl-oggm`` is a shallow fork that adds a few OGGM specific features
and enhancements.

For each figure to test, an image is generated and then subtracted from an existing reference image.
If the RMS of the residual is larger than a user-specified tolerance, the test will fail.
Alternatively, the generated image can be hashed and compared to an expected value.

For more information about the original ``pytest-mpl``, see the
`pytest-mpl documentation <https://pytest-mpl.readthedocs.io>`__.

For more information about this fork, visit https://github.com/OGGM/pytest-mpl-oggm

Installation
------------
.. code-block:: bash

   pip install pytest-mpl-oggm


Usage
-----
First, write test functions that create a figure.
These image comparison tests are decorated with ``@pytest.mark.mpl_image_compare`` and return the figure for testing:

.. code-block:: python

   import matplotlib.pyplot as plt
   import pytest

   @pytest.mark.mpl_image_compare
   def test_plot():
       fig, ax = plt.subplots()
       ax.plot([1, 2])
       return fig

Then, run the test suite as usual, but pass ``--mpl-oggm`` to compare the returned figures to the reference images:

.. code-block:: bash

   pytest --mpl-oggm

By also passing ``--mpl-generate-summary=html``, a summary of the image comparison results will be generated in HTML format:

+---------------+---------------+---------------+
| |html all|    | |html filter| | |html result| |
+---------------+---------------+---------------+

For more information on how to configure and use ``pytest-mpl``, see the `pytest-mpl documentation <https://pytest-mpl.readthedocs.io>`__.

.. |html all| image:: docs/images/html_all.png
.. |html filter| image:: docs/images/html_filter.png
.. |html result| image:: docs/images/html_result.png
