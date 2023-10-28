package layers

import (
	"github.com/NiklasTreml/go-nn/pkg/util"
	"gonum.org/v1/gonum/mat"
)

var _ Layer = (*Dense)(nil)

type Dense struct {
	weights *mat.Dense
	bias    *mat.Dense
	input   *mat.Dense
	output  *mat.Dense
}

func NewDense(inputSize int, outputSize int) *Dense {

	layer := &Dense{
		weights: mat.NewDense(inputSize, outputSize, util.RandomData(inputSize*outputSize)),
		input:   mat.NewDense(inputSize, outputSize, util.RandomData(inputSize*outputSize)),
		bias:    mat.NewDense(1, inputSize, util.RandomData(inputSize)),
		output:  mat.NewDense(1, outputSize, nil),
	}

	return layer
}

func (d *Dense) Forward(input *mat.Dense) *mat.Dense {
	// Y = XW + B
	d.input = input
	d.output.Mul(d.input, d.weights)
	d.output.Add(d.input, d.bias)

	return d.output
}

func (d *Dense) Backward(outputError *mat.Dense, alpha float64) *mat.Dense {
	// Computes dE/dW for the outputError=dE/dY. Returns dE/dX
	inputError := mat.NewDense(1, 1, nil)
	weightsError := mat.NewDense(1, 1, nil)

	// dE/dX = dE/dY * W^T
	inputError.Mul(outputError, d.weights.T())
	// dE/dW = X^T * dE/dY
	weightsError.Mul(d.input.T(), outputError)

	// Update parameters
	// scale the weights and output error with the learning rate
	weightsError.Scale(alpha, weightsError)
	outputError.Scale(alpha, outputError)

	// d.weights = d.weights - alpha * weightsError
	d.weights.Sub(d.weights, weightsError)
	// d.bias = d.bias - alpha * outputError
	d.bias.Sub(d.bias, outputError)

	return inputError
}
