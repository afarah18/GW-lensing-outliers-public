import numpy as np
from scipy.integrate import cumtrapz

def inverse_transform_sample(pdf, x_limits, rng, N=1000, num_interp_points=1000, **pdf_kwargs):
    """Generates N samples from the supplied pdf using inverse transform sampling
    pdf (callable): the probability distribution function defined on x
    x_limits (iterable, length 2): the bounds between which to evaluate the pdf
        and draw samples because many pdfs don't just go to zero by themselves.
    N (int): the number of samples to draw, defaults to 1k,
    num_interp_points (int): number of points on which to evaluate the CDF for 
        linear interpolation.
    pdf_kwargs (kwargs): any keword argument to be passed to the pdf function.
    """

    # get the CDF of the PDF
    xs = np.linspace(x_limits[0], x_limits[1], num=num_interp_points)
    if type(pdf)==type(np.array(1)):
        f_of_xs = pdf
    else:
        f_of_xs = pdf(xs, **pdf_kwargs)
    cdf = cumtrapz(f_of_xs, xs, initial=0)
    cdf /= cdf[-1] # normalize

    # choose random points along that CDF
    y = rng.uniform(size=N)

    # convert those to samples by evaluating the value of x
    # required to get you that value of the CDF
    samples = np.interp(y, cdf, xs)

    return samples
