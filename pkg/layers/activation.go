package layers

import "gonum.org/v1/gonum/mat"

type Activation struct {
	activation      ActivationFn
	activationPrime ActivationFnPrime
	input           *mat.Dense
	output          *mat.Dense
}

var _ Layer = (*Activation)(nil)

type ActivationFn func(*mat.Dense) *mat.Dense
type ActivationFnPrime func(*mat.Dense) *mat.Dense

func NewActivation(activation ActivationFn, activationPrime ActivationFnPrime) *Activation {
	return &Activation{
		activation:      activation,
		activationPrime: activationPrime,
	}
}

func (a *Activation) Forward(input *mat.Dense) *mat.Dense {
	a.input = mat.DenseCopyOf(input)
	a.output = a.activation(input)

	return a.output
}

func (a *Activation) Backward(outputError *mat.Dense, alpha float64) *mat.Dense {
	res := a.activationPrime(a.input)
	res.MulElem(res, outputError)

	return res
}
