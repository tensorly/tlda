from ..cumulant_gradient import cumulant_gradient
import tensorly as tl
from tensorly.testing import assert_array_almost_equal

def test_gradient():
    a = tl.tensor([[1., 2., 1.],
                   [-1., 0., 4.],
                   [0., -1., -3.],
                   [0., 1., -2.]])

    fac = tl.tensor([[0.8, 0.28, 0.96],
                    [0.6, 0., 0.224],
                    [0., 0.96, 0.168]])

    grad_true = tl.tensor([[45.10146632, 41.48045819, 49.48543984],
                        [7.81949602, 5.80957422, 13.50289805],
                        [-4.60580231, -12.24053294, 8.70265979]])

    grad_calc = cumulant_gradient(fac, a, alpha = 1)

    assert_array_almost_equal(grad_calc, grad_true)